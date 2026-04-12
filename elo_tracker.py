import chess
import numpy as np
import copy
from typing import Tuple

from config import cfg
from logger import get_logger

log = get_logger("elo")


class EloTracker:
    def __init__(self, initial_elo: float = 1000.0):
        self.current_elo: float = initial_elo
        self.best_elo: float = initial_elo
        self.history = []

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def update_elo(rating: float, score: float, expected: float, k: float = 32.0) -> float:
        return rating + k * (score - expected)

    def update(self, score: float, opponent_elo: float):
        """
        Update current Elo.
        score: 1.0 = win, 0.5 = draw, 0.0 = loss
        """
        expected = self.expected_score(self.current_elo, opponent_elo)
        self.current_elo = self.update_elo(self.current_elo, score, expected)
        self.history.append(self.current_elo)
        return self.current_elo


def play_eval_game(current_model, best_model, current_is_white: bool, device: str) -> float:
    """
    Play one game between current and best model.
    Returns score for current_model: 1.0 win, 0.5 draw, 0.0 loss.
    Uses fast policy-only play (no MCTS) for speed.
    """
    from board import encode_board
    import torch

    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < cfg.max_game_length:
        is_current_turn = (board.turn == chess.WHITE) == current_is_white

        model = current_model if is_current_turn else best_model
        model.eval()

        enc = encode_board(board)
        with torch.no_grad():
            x = torch.tensor(enc, dtype=torch.float32).unsqueeze(0).to(device)
            policy_logits, _ = model(x)
            logits = policy_logits.squeeze(0).cpu().numpy()

        # Mask illegal moves
        from moves import legal_move_ids, id_to_legal_move
        legal_ids = legal_move_ids(board)
        if not legal_ids:
            break

        legal_logits = np.array([logits[mid] for mid in legal_ids])
        best_idx = np.argmax(legal_logits)
        move = id_to_legal_move(legal_ids[best_idx], board)

        if move is None:
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[0]

        board.push(move)
        move_count += 1

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.5

    winner_is_white = outcome.winner == chess.WHITE
    current_won = (winner_is_white == current_is_white)
    return 1.0 if current_won else 0.0


def evaluate_models(current_model, best_model, elo_tracker: EloTracker,
                    device: str, n_games: int = None) -> Tuple[float, bool]:
    """
    Evaluate current model vs best model.
    Returns (win_rate, should_replace_best).
    """
    n = n_games or cfg.eval_games
    scores = []

    for g in range(n):
        current_is_white = (g % 2 == 0)
        score = play_eval_game(current_model, best_model, current_is_white, device)
        scores.append(score)
        log.info(f"Eval game {g+1}/{n}: {'win' if score==1 else 'draw' if score==0.5 else 'loss'} "
                 f"({'white' if current_is_white else 'black'})")

    win_rate = np.mean(scores)
    total_score = sum(scores)

    # Update Elo
    elo_tracker.update(total_score / n, elo_tracker.best_elo)
    log.info(f"Eval complete. Win rate: {win_rate:.2f} | Elo: {elo_tracker.current_elo:.0f}")

    # Replace best if current wins majority
    should_replace = win_rate > 0.55
    if should_replace:
        elo_tracker.best_elo = elo_tracker.current_elo
        log.info("Current model is new best!")

    return win_rate, should_replace
