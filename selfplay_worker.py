"""
selfplay_worker.py
==================
Self-play with repetition hard-blocked at the game loop level.

Key change
----------
Previously repetition was only discouraged via reward shaping.
The model could still *choose* a repeating move if the policy was
confident enough.  Now the game loop itself blocks 3-fold repetitions:

    if after pushing a move the position would be seen for the 3rd time,
    that move is removed from the candidate list before sampling.

This is done in both MCTS mode and fast (policy-only) mode.

Reward shaping changes (all English, all previous logic kept)
-------------------------------------------------------------
- Checkmate:            +5.0
- Stalemate:            -0.5
- Safe check bonus:     +0.06  (only if SEE says the piece survives)
- Unsafe check:          0.0   (no bonus -- was causing blunders)
- Castling bonus:       +0.02
- Repetition penalty:   -0.10 (2-fold) / -0.30 (3-fold attempt, shouldn't happen now)
- Hanging piece penalty: proportional to piece value
- Game result blend:    0.8 * outcome + 0.2 * shaped  (outcome dominates)
"""

import multiprocessing as mp
import numpy as np
import chess
import torch
import time
from typing import List, Tuple, Optional

from board import encode_board, material_balance, display_board, clear_screen, MATERIAL_VALUES
from moves import move_to_id, id_to_legal_move, legal_move_ids, MOVE_SPACE
from config import cfg
from logger import get_logger

log = get_logger("selfplay")

TrajectoryStep = Tuple[np.ndarray, np.ndarray, float]


# ---------------------------------------------------------------------------
# SEE helpers
# ---------------------------------------------------------------------------

def _see(board: chess.Board, sq: int,
         target_value: int, attacker_color: chess.Color) -> bool:
    least_val = None
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]:
        ats = board.pieces(pt, attacker_color) & board.attackers(attacker_color, sq)
        if ats:
            v = MATERIAL_VALUES[pt]
            if least_val is None or v < least_val:
                least_val = v
    if least_val is None:
        return False
    return least_val <= target_value


def _is_safe_check(board: chess.Board, move: chess.Move) -> bool:
    if not board.gives_check(move):
        return False
    board_after = board.copy()
    board_after.push(move)
    checking_piece = board_after.piece_at(move.to_square)
    if checking_piece is None:
        return True
    piece_value = MATERIAL_VALUES.get(checking_piece.piece_type, 0)
    opp_color   = board_after.turn
    if not board_after.is_attacked_by(opp_color, move.to_square):
        return True
    return not _see(board_after, move.to_square, piece_value, opp_color)


def _hanging_penalty(board: chess.Board, move: chess.Move) -> float:
    board_after = board.copy()
    board_after.push(move)
    our_color = not board_after.turn
    opp_color = board_after.turn
    penalty   = 0.0
    for sq in chess.SQUARES:
        piece = board_after.piece_at(sq)
        if piece is None or piece.color != our_color:
            continue
        val = MATERIAL_VALUES.get(piece.piece_type, 0)
        if val < 3:
            continue
        if (board_after.is_attacked_by(opp_color, sq) and
                not board_after.is_attacked_by(our_color, sq)):
            penalty += val * 0.04
    return -penalty


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_reward(board: chess.Board, move: chess.Move) -> float:
    board_copy = board.copy()
    board_copy.push(move)

    if board_copy.is_checkmate():
        return 5.0
    if board_copy.is_stalemate():
        return -0.5

    before = material_balance(board)
    after  = material_balance(board_copy)
    delta  = after - before
    if not board.turn:
        delta = -delta
    reward = delta * cfg.material_reward_factor

    if board.gives_check(move):
        if _is_safe_check(board, move):
            reward += 0.06

    if board.fullmove_number <= 20 and board.is_castling(move):
        reward += cfg.castle_bonus

    try:
        if board_copy.is_repetition(2):
            reward -= 0.30
        elif board_copy.is_repetition(1):
            reward -= 0.10
    except Exception:
        pass

    reward += _hanging_penalty(board, move)
    return reward


# ---------------------------------------------------------------------------
# Repetition filter  (THE KEY FIX)
# ---------------------------------------------------------------------------

def _filter_repeating_moves(board: chess.Board,
                             legal_ids: List[int]) -> Tuple[List[int], List[int]]:
    """
    Split legal_ids into (allowed, forbidden).

    forbidden = moves that would cause a 3-fold repetition.
    allowed   = everything else.

    If ALL moves are forbidden (extremely rare fortress), return all as allowed
    to avoid having zero candidates.
    """
    allowed   = []
    forbidden = []

    for mid in legal_ids:
        mv = id_to_legal_move(mid, board)
        if mv is None:
            allowed.append(mid)
            continue
        tmp = board.copy()
        tmp.push(mv)
        try:
            if tmp.is_repetition(3):
                forbidden.append(mid)
            else:
                allowed.append(mid)
        except Exception:
            allowed.append(mid)

    if not allowed:
        return forbidden, []   # forced -- allow all

    return allowed, forbidden


def _apply_soft_repeat_penalty(logits: np.ndarray,
                                legal_ids: List[int],
                                board: chess.Board,
                                penalty: float = 3.0) -> np.ndarray:
    """Subtract `penalty` from logit of any move that revisits a position."""
    logits = logits.copy()
    for i, mid in enumerate(legal_ids):
        mv = id_to_legal_move(mid, board)
        if mv is None:
            continue
        tmp = board.copy()
        tmp.push(mv)
        try:
            if tmp.is_repetition(2):
                logits[i] -= penalty
            elif tmp.is_repetition(1):
                logits[i] -= penalty * 0.4
        except Exception:
            pass
    return logits


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def play_one_game(
    model_state_dict,
    device: str,
    watch: bool = False,
    fast: bool = False,
) -> List[TrajectoryStep]:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from model import ChessNet
    from mcts import MCTS

    model = ChessNet()
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    board      = chess.Board()
    trajectory = []
    move_count = 0
    last_move  = None

    if not fast:
        mcts = MCTS(model, device, cfg)

    while not board.is_game_over() and move_count < cfg.max_game_length:

        # The game engine already handles 3-fold via is_game_over(),
        # but it only triggers *after* the move is pushed.
        # We block here *before* the move is chosen.
        try:
            if board.is_repetition(3):
                break
        except Exception:
            pass

        temperature = cfg.temperature if move_count < cfg.temperature_drop_move else 0.0
        state       = encode_board(board)

        all_legal_ids = legal_move_ids(board)
        if not all_legal_ids:
            break

        # --- hard-block 3-fold moves BEFORE feeding to model ---
        allowed_ids, forbidden_ids = _filter_repeating_moves(board, all_legal_ids)

        if fast:
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                policy_logits, _ = model(x)
                logits = policy_logits.squeeze(0).cpu().numpy()

            legal_logits = np.array([logits[mid] for mid in allowed_ids],
                                    dtype=np.float64)

            # Soft penalty on 2-fold moves within allowed set
            legal_logits = _apply_soft_repeat_penalty(
                legal_logits, allowed_ids, board)

            if temperature == 0:
                probs = np.zeros(len(allowed_ids))
                probs[np.argmax(legal_logits)] = 1.0
            else:
                legal_logits -= legal_logits.max()
                exp_l = np.exp(legal_logits)
                probs = exp_l / (exp_l.sum() + 1e-9)

            policy = np.zeros(MOVE_SPACE, dtype=np.float32)
            for mid, p in zip(allowed_ids, probs):
                policy[mid] = float(p)

            chosen_id = int(np.random.choice(allowed_ids, p=probs))

        else:
            # MCTS: search over full board, then filter results
            root = mcts.search(board, cfg.num_simulations)
            move_ids, probs = mcts.get_action_probs(root, temperature)

            if not move_ids:
                break

            # Remove forbidden moves from MCTS results
            allowed_set = set(allowed_ids)
            filtered_ids   = []
            filtered_probs = []
            for mid, p in zip(move_ids, probs):
                if mid in allowed_set:
                    filtered_ids.append(mid)
                    filtered_probs.append(p)

            # Soft penalty on 2-fold within filtered
            if filtered_ids:
                fp = np.array(filtered_probs, dtype=np.float64)
                for i, mid in enumerate(filtered_ids):
                    mv = id_to_legal_move(mid, board)
                    if mv is None:
                        continue
                    tmp = board.copy()
                    tmp.push(mv)
                    try:
                        if tmp.is_repetition(2):
                            fp[i] *= 0.05
                        elif tmp.is_repetition(1):
                            fp[i] *= 0.3
                    except Exception:
                        pass
                total = fp.sum()
                if total > 1e-9:
                    fp /= total
                else:
                    fp = np.ones(len(fp)) / len(fp)
                filtered_probs = fp.tolist()
            else:
                # Fallback: use all (should not happen)
                filtered_ids   = list(move_ids)
                filtered_probs = list(probs)

            policy = np.zeros(MOVE_SPACE, dtype=np.float32)
            for mid, p in zip(filtered_ids, filtered_probs):
                policy[mid] = float(p)

            if temperature == 0:
                chosen_id = filtered_ids[int(np.argmax(filtered_probs))]
            else:
                fp = np.array(filtered_probs, dtype=np.float64)
                fp /= fp.sum()
                chosen_id = int(np.random.choice(filtered_ids, p=fp))

        move = id_to_legal_move(chosen_id, board)
        if move is None:
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[0]

        shaped_reward = compute_reward(board, move)
        trajectory.append((state, policy, shaped_reward))
        board.push(move)
        last_move  = move
        move_count += 1

        if watch:
            clear_screen()
            side     = 'White' if board.turn == chess.BLACK else 'Black'
            mode_str = 'fast' if fast else f'MCTS ({cfg.num_simulations})'
            rep_warn = ''
            try:
                if board.is_repetition(1):
                    rep_warn = '  [position seen before]'
            except Exception:
                pass
            print(f'  Self-play | {mode_str} | Move {move_count}: '
                  f'{side} {move.uci()}{rep_warn}')
            print()
            print(display_board(board, last_move=last_move))
            print()
            if board.is_checkmate():
                print('  ** CHECKMATE **')
            elif board.is_check():
                print('  Check!')
            elif board.is_stalemate():
                print('  Stalemate')
            print()
            time.sleep(0.1)

    # --- Outcome ---
    outcome = board.outcome()
    if outcome is None:
        game_result = 0.0
    elif outcome.winner is None:
        game_result = 0.0
    elif outcome.winner == chess.WHITE:
        game_result = 1.0
    else:
        game_result = -1.0

    if watch:
        clear_screen()
        print(display_board(board, last_move=last_move))
        print()
        if outcome is None:
            print(f'  Draw (max moves/repetition) after {move_count} moves')
        elif outcome.winner is None:
            print(f'  Draw ({outcome.termination.name}) after {move_count} moves')
        elif outcome.winner == chess.WHITE:
            print(f'  White wins after {move_count} moves')
        else:
            print(f'  Black wins after {move_count} moves')
        print()

    # 0.8 outcome + 0.2 shaped (outcome dominates)
    samples = []
    for i, (s, pol, shaped) in enumerate(trajectory):
        perspective = 1.0 if i % 2 == 0 else -1.0
        value = max(-1.0, min(1.0,
                    0.8 * perspective * game_result + 0.2 * shaped))
        samples.append((s, pol, value))

    return samples


# ---------------------------------------------------------------------------
# Worker / manager (unchanged logic)
# ---------------------------------------------------------------------------

def worker_fn(worker_id: int, model_state_dict_queue: mp.Queue,
              result_queue: mp.Queue, games_per_worker: int, device: str):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _log = get_logger(f"worker-{worker_id}")
    _log.info(f"Worker {worker_id} started")
    while True:
        state = model_state_dict_queue.get()
        if state is None:
            break
        samples = []
        for g in range(games_per_worker):
            try:
                s = play_one_game(state, device, watch=False, fast=True)
                samples.extend(s)
                _log.info(f"W{worker_id} g{g+1}/{games_per_worker}: {len(s)} steps")
            except Exception as e:
                _log.warning(f"W{worker_id} game error: {e}")
        result_queue.put(samples)


class SelfPlayManager:
    def __init__(self, model, device: str = "cpu"):
        self.model  = model
        self.device = device

    def collect_games(self, n_games: int) -> List[TrajectoryStep]:
        num_workers      = min(cfg.num_workers, n_games)
        games_per_worker = max(1, n_games // num_workers)
        ctx              = mp.get_context("spawn")
        result_queue     = ctx.Queue()
        model_state      = {k: v.cpu() for k, v in self.model.state_dict().items()}
        queues, procs    = [], []
        for wid in range(num_workers):
            q = ctx.Queue()
            q.put(model_state)
            p = ctx.Process(target=worker_fn,
                            args=(wid, q, result_queue, games_per_worker, "cpu"),
                            daemon=True)
            queues.append(q); procs.append(p); p.start()
        all_samples = []
        for _ in range(num_workers):
            all_samples.extend(result_queue.get(timeout=600))
        for q in queues:
            q.put(None)
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        return all_samples


def collect_selfplay_single(
    model,
    n_games: int,
    device: str,
    watch: bool = False,
    fast: bool = False,
) -> List[TrajectoryStep]:
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    all_samples = []
    for g in range(n_games):
        try:
            if watch:
                print(f'\n  -- Game {g+1}/{n_games} --')
                time.sleep(0.4)
            s = play_one_game(model_state, device, watch=watch, fast=fast)
            all_samples.extend(s)
            if not watch:
                log.info(f"Game {g+1}/{n_games}: {len(s)} steps")
        except Exception as e:
            log.warning(f"Game {g+1} error: {e}")
    return all_samples
