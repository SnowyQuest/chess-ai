"""
selfplay_worker.py
==================
Self-play data collection with corrected reward shaping.

Changes from previous version
------------------------------
1. Check bonus is now CONDITIONAL (was unconditional).
   A check only earns a bonus when it is SAFE, meaning the checking piece
   is not immediately capturable at a loss.  This eliminates the pattern
   where the model sacrifices material just to deliver a check.

2. Checkmate reward: preserved at +5.0

3. Hanging-piece penalty: if after our move we leave a piece hanging
   with value >= 3 (knight/bishop or higher), apply a penalty.
   This is the primary driver of check-blunders: the model moved a piece
   to give check but left something hanging behind it.

4. Stalemate penalty: -0.5 (unchanged)

5. Repetition penalty: -0.10 / -0.30 (unchanged)

6. Trajectory value blend: game_result weight raised 0.7 -> 0.8
   (outcome matters more than shaped reward, reduces noise)
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
# SEE (Static Exchange Evaluation) helper
# ---------------------------------------------------------------------------

def _see(board: chess.Board, sq: int, target_value: int, attacker_color: chess.Color) -> bool:
    """
    Simplified SEE: can `attacker_color` capture on `sq` and come out ahead?
    Returns True if the capture is profitable (gain >= 0).
    """
    least_val = None
    least_sq  = None
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]:
        ats = board.pieces(pt, attacker_color) & board.attackers(attacker_color, sq)
        if ats:
            v = MATERIAL_VALUES[pt]
            if least_val is None or v < least_val:
                least_val = v
                least_sq  = ats.pop()

    if least_sq is None:
        return False  # not attacked

    # Profitable if attacker is worth less than what it captures
    return least_val <= target_value


def _is_safe_check(board: chess.Board, move: chess.Move) -> bool:
    """
    Return True if `move` gives check AND the checking piece is not
    immediately capturable at a loss by the opponent.

    Called BEFORE pushing the move.
    """
    if not board.gives_check(move):
        return False

    board_after = board.copy()
    board_after.push(move)

    # The piece that just moved is now on move.to_square
    checking_piece = board_after.piece_at(move.to_square)
    if checking_piece is None:
        return True  # edge case

    piece_value = MATERIAL_VALUES.get(checking_piece.piece_type, 0)
    opp_color   = board_after.turn   # it's now opponent's turn

    # Is the checking piece attacked by the opponent?
    if not board_after.is_attacked_by(opp_color, move.to_square):
        return True  # not attacked at all -> safe

    # Attacked: is the capture profitable for the opponent?
    profitable = _see(board_after, move.to_square, piece_value, opp_color)
    return not profitable   # safe if opponent can't profit from recapture


def _hanging_penalty(board: chess.Board, move: chess.Move) -> float:
    """
    Return a penalty if after playing `move` one of our pieces is left
    hanging (value >= 3).  This catches the main check-blunder pattern.
    """
    board_after = board.copy()
    board_after.push(move)

    our_color  = not board_after.turn   # we just moved
    opp_color  = board_after.turn

    penalty = 0.0
    for sq in chess.SQUARES:
        piece = board_after.piece_at(sq)
        if piece is None or piece.color != our_color:
            continue
        val = MATERIAL_VALUES.get(piece.piece_type, 0)
        if val < 3:
            continue  # ignore pawns for performance

        # Is it attacked by opponent and not defended?
        if (board_after.is_attacked_by(opp_color, sq) and
                not board_after.is_attacked_by(our_color, sq)):
            penalty += val * 0.04   # proportional to piece value

    return -penalty


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(board: chess.Board, move: chess.Move) -> float:
    """
    Shaped reward for a single move.

    Components
    ----------
    +5.0   checkmate
    -0.5   stalemate
    +material_delta * 0.05
    +0.06  safe check (was unconditional 0.05 -- now gated by SEE)
     0.0   unsafe check (piece hangs after check -- no bonus)
    +0.02  castling (opening only)
    -0.10 / -0.30  repetition
    -variable  leaving piece hanging after move
    """
    board_copy = board.copy()
    board_copy.push(move)

    # --- Terminal outcomes ---
    if board_copy.is_checkmate():
        return 5.0
    if board_copy.is_stalemate():
        return -0.5

    # --- Material delta ---
    before = material_balance(board)
    after  = material_balance(board_copy)
    delta  = after - before
    if not board.turn:
        delta = -delta
    reward = delta * cfg.material_reward_factor

    # --- Check bonus: ONLY for safe checks ---
    if board.gives_check(move):
        if _is_safe_check(board, move):
            reward += 0.06
        # unsafe check -> no bonus (was giving +0.05 unconditionally before)

    # --- Castling bonus ---
    if board.fullmove_number <= 20 and board.is_castling(move):
        reward += cfg.castle_bonus

    # --- Repetition penalty ---
    try:
        if board_copy.is_repetition(2):
            reward -= 0.30
        elif board_copy.is_repetition(1):
            reward -= 0.10
    except Exception:
        pass

    # --- Hanging piece penalty ---
    reward += _hanging_penalty(board, move)

    return reward


# ---------------------------------------------------------------------------
# Play one game
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

        # Early repetition exit
        try:
            if board.is_repetition(3):
                break
        except Exception:
            pass

        temperature = cfg.temperature if move_count < cfg.temperature_drop_move else 0.0
        state = encode_board(board)

        if fast:
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                policy_logits, _ = model(x)
                logits = policy_logits.squeeze(0).cpu().numpy()

            legal_ids = legal_move_ids(board)
            if not legal_ids:
                break

            legal_logits = np.array([logits[mid] for mid in legal_ids], dtype=np.float64)

            # Repetition penalty on logits
            for i, mid in enumerate(legal_ids):
                mv = id_to_legal_move(mid, board)
                if mv is None:
                    continue
                tmp = board.copy()
                tmp.push(mv)
                try:
                    if tmp.is_repetition(2):
                        legal_logits[i] -= 3.0
                    elif tmp.is_repetition(1):
                        legal_logits[i] -= 1.0
                except Exception:
                    pass

            if temperature == 0:
                probs = np.zeros(len(legal_ids))
                probs[np.argmax(legal_logits)] = 1.0
            else:
                legal_logits -= legal_logits.max()
                exp_l = np.exp(legal_logits)
                probs = exp_l / exp_l.sum()

            policy = np.zeros(MOVE_SPACE, dtype=np.float32)
            for mid, p in zip(legal_ids, probs):
                policy[mid] = float(p)
            chosen_id = int(np.random.choice(legal_ids, p=probs))

        else:
            root = mcts.search(board, cfg.num_simulations)
            move_ids, probs = mcts.get_action_probs(root, temperature)
            if not move_ids:
                break
            policy = np.zeros(MOVE_SPACE, dtype=np.float32)
            for mid, p in zip(move_ids, probs):
                policy[mid] = float(p)
            if temperature == 0:
                chosen_id = move_ids[int(np.argmax(probs))]
            else:
                chosen_id = int(np.random.choice(move_ids, p=probs))

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
            side = 'White' if board.turn == chess.BLACK else 'Black'
            mode_str = 'fast' if fast else f'MCTS ({cfg.num_simulations})'
            print(f'  Self-play | {mode_str} | Move {move_count}: {side} {move.uci()}')
            print()
            print(display_board(board, last_move=last_move))
            print()
            if board.is_checkmate():
                print('  ** CHECKMATE **')
            elif board.is_check():
                print('  Check!')
            elif board.is_stalemate():
                print('  Stalemate')
            try:
                if board.is_repetition(2):
                    print('  WARNING: 3rd repetition imminent')
            except Exception:
                pass
            print()
            time.sleep(0.1)

    # --- Determine outcome ---
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

    # --- Build training samples ---
    # Game result weighted more heavily (0.8) than shaped reward (0.2)
    # This reduces noise from imperfect reward shaping
    samples = []
    for i, (state, policy, shaped) in enumerate(trajectory):
        perspective = 1.0 if i % 2 == 0 else -1.0
        value = max(-1.0, min(1.0,
                    0.8 * perspective * game_result + 0.2 * shaped))
        samples.append((state, policy, value))

    return samples


# ---------------------------------------------------------------------------
# Worker & manager
# ---------------------------------------------------------------------------

def worker_fn(worker_id: int, model_state_dict_queue: mp.Queue,
              result_queue: mp.Queue, games_per_worker: int, device: str):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _log = get_logger(f"worker-{worker_id}")
    _log.info(f"Worker {worker_id} started")
    while True:
        model_state = model_state_dict_queue.get()
        if model_state is None:
            break
        all_samples = []
        for g in range(games_per_worker):
            try:
                s = play_one_game(model_state, device, watch=False, fast=True)
                all_samples.extend(s)
                _log.info(f"W{worker_id} game {g+1}/{games_per_worker}: {len(s)} steps")
            except Exception as e:
                _log.warning(f"W{worker_id} game error: {e}")
        result_queue.put(all_samples)


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
        worker_queues, processes = [], []
        for wid in range(num_workers):
            q = ctx.Queue()
            q.put(model_state)
            p = ctx.Process(
                target=worker_fn,
                args=(wid, q, result_queue, games_per_worker, "cpu"),
                daemon=True,
            )
            worker_queues.append(q)
            processes.append(p)
            p.start()
        all_samples = []
        for _ in range(num_workers):
            all_samples.extend(result_queue.get(timeout=600))
        for q in worker_queues:
            q.put(None)
        for p in processes:
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
                time.sleep(0.5)
            s = play_one_game(model_state, device, watch=watch, fast=fast)
            all_samples.extend(s)
            if not watch:
                log.info(f"Game {g+1}/{n_games}: {len(s)} steps")
        except Exception as e:
            log.warning(f"Game {g+1} error: {e}")
    return all_samples
