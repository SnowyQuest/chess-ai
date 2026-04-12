import multiprocessing as mp
import numpy as np
import chess
import torch
import time
from typing import List, Tuple, Optional

from board import encode_board, material_balance, display_board, clear_screen
from moves import move_to_id, id_to_legal_move, legal_move_ids, MOVE_SPACE
from config import cfg
from logger import get_logger

log = get_logger("selfplay")

TrajectoryStep = Tuple[np.ndarray, np.ndarray, float]


def compute_reward(board: chess.Board, move: chess.Move) -> float:
    """Hybrid reward: material change + check bonus + castling bonus."""
    before = material_balance(board)
    board_copy = board.copy()
    board_copy.push(move)
    after = material_balance(board_copy)
    delta = after - before
    if not board.turn:  # black to move -> negate
        delta = -delta
    reward = delta * cfg.material_reward_factor

    if board_copy.is_check():
        reward += cfg.check_bonus
    if board.fullmove_number <= 20 and board.is_castling(move):
        reward += cfg.castle_bonus

    return reward


def play_one_game(
    model_state_dict,
    device: str,
    watch: bool = False,
    fast: bool = False,
) -> List[TrajectoryStep]:
    """
    Play one self-play game and return trajectory.

    Args:
        model_state_dict: model weights dict
        device: torch device string
        watch: if True, print board to terminal after each move
        fast: if True, use policy-only (no MCTS) for speed
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from model import ChessNet
    from mcts import MCTS

    model = ChessNet()
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    board = chess.Board()
    trajectory = []
    move_count = 0
    last_move = None

    if not fast:
        mcts = MCTS(model, device, cfg)

    while not board.is_game_over() and move_count < cfg.max_game_length:
        temperature = cfg.temperature if move_count < cfg.temperature_drop_move else 0.0
        state = encode_board(board)

        if fast:
            # Policy-only: fast but weaker
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                policy_logits, _ = model(x)
                logits = policy_logits.squeeze(0).cpu().numpy()

            legal_ids = legal_move_ids(board)
            if not legal_ids:
                break

            legal_logits = np.array([logits[mid] for mid in legal_ids])

            if temperature == 0:
                probs = np.zeros(len(legal_ids))
                probs[np.argmax(legal_logits)] = 1.0
            else:
                legal_logits -= legal_logits.max()
                exp_logits = np.exp(legal_logits)
                probs = exp_logits / exp_logits.sum()

            policy = np.zeros(MOVE_SPACE, dtype=np.float32)
            for mid, p in zip(legal_ids, probs):
                policy[mid] = float(p)

            chosen_id = int(np.random.choice(legal_ids, p=probs))
        else:
            # MCTS
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
        last_move = move
        move_count += 1

        # --- Live watch ---
        if watch:
            clear_screen()
            side = 'White' if board.turn == chess.BLACK else 'Black'  # who just moved
            mode_str = 'fast (policy)' if fast else f'MCTS ({cfg.num_simulations} sims)'
            print(f'  Self-play | Mode: {mode_str} | Move {move_count}: {side} plays {move.uci()}')
            print()
            print(display_board(board, last_move=last_move))
            print()
            # Material
            bal = material_balance(board)
            print(f'  Material balance (white): {bal:+d}')
            # Check/mate status
            if board.is_checkmate():
                print('  ★ Checkmate!')
            elif board.is_check():
                print('  ⚡ Check!')
            elif board.is_stalemate():
                print('  = Stalemate')
            print()
            time.sleep(0.1)  # küçük gecikme - takip edilebilsin

    # Game result
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
            print(f'  Game over (max moves) — Draw after {move_count} moves')
        elif outcome.winner is None:
            print(f'  Draw! ({outcome.termination.name}) — {move_count} moves')
        elif outcome.winner == chess.WHITE:
            print(f'  ♙ White wins! — {move_count} moves')
        else:
            print(f'  ♟ Black wins! — {move_count} moves')
        print()

    # Assign values
    samples = []
    for i, (state, policy, shaped) in enumerate(trajectory):
        perspective = 1.0 if i % 2 == 0 else -1.0
        value = max(-1.0, min(1.0, perspective * game_result + shaped))
        samples.append((state, policy, value))

    return samples


def worker_fn(worker_id: int, model_state_dict_queue: mp.Queue,
               result_queue: mp.Queue, games_per_worker: int, device: str):
    """Worker process: play games and send results (no watch mode)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    log = get_logger(f"worker-{worker_id}")
    log.info(f"Worker {worker_id} started")

    while True:
        model_state = model_state_dict_queue.get()
        if model_state is None:
            break

        all_samples = []
        for g in range(games_per_worker):
            try:
                samples = play_one_game(model_state, device, watch=False, fast=True)
                all_samples.extend(samples)
                log.info(f"Worker {worker_id} game {g+1}/{games_per_worker}: {len(samples)} steps")
            except Exception as e:
                log.warning(f"Worker {worker_id} game error: {e}")

        result_queue.put(all_samples)


class SelfPlayManager:
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def collect_games(self, n_games: int) -> List[TrajectoryStep]:
        num_workers = min(cfg.num_workers, n_games)
        games_per_worker = max(1, n_games // num_workers)

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        worker_queues = []
        processes = []
        for wid in range(num_workers):
            q = ctx.Queue()
            q.put(model_state)
            p = ctx.Process(
                target=worker_fn,
                args=(wid, q, result_queue, games_per_worker, "cpu"),
                daemon=True
            )
            worker_queues.append(q)
            processes.append(p)
            p.start()

        all_samples = []
        for _ in range(num_workers):
            samples = result_queue.get(timeout=600)
            all_samples.extend(samples)

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
    """Single-process self-play. Supports watch mode."""
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    all_samples = []
    for g in range(n_games):
        try:
            if watch:
                print(f'\n  ── Game {g+1}/{n_games} ──')
                time.sleep(0.5)
            samples = play_one_game(model_state, device, watch=watch, fast=fast)
            all_samples.extend(samples)
            if not watch:
                log.info(f"Game {g+1}/{n_games}: {len(samples)} steps")
        except Exception as e:
            log.warning(f"Game {g+1} error: {e}")
    return all_samples
