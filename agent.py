import numpy as np
import chess
from typing import Optional

from mcts import MCTS
from config import cfg


class Agent:
    def __init__(self, model, device: str = "cpu", num_simulations: Optional[int] = None):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations or cfg.num_simulations
        self.mcts = MCTS(model, device, cfg)

    def select_move(self, board: chess.Board, temperature: float = 1.0) -> chess.Move:
        """Select a move using MCTS."""
        root = self.mcts.search(board, self.num_simulations)
        move_ids, probs = self.mcts.get_action_probs(root, temperature)

        if not move_ids:
            # Fallback: random legal move
            legal = list(board.legal_moves)
            if not legal:
                raise ValueError("No legal moves available")
            return np.random.choice(legal)

        if temperature == 0:
            chosen_id = move_ids[np.argmax(probs)]
        else:
            chosen_id = np.random.choice(move_ids, p=probs)

        from moves import id_to_legal_move
        move = id_to_legal_move(chosen_id, board)
        if move is None:
            # Fallback
            legal = list(board.legal_moves)
            return legal[0] if legal else None
        return move

    def get_policy_target(self, board: chess.Board, temperature: float = 1.0) -> np.ndarray:
        """
        Run MCTS and return full policy vector (4096,) from visit counts.
        Used as training target.
        """
        from moves import MOVE_SPACE
        root = self.mcts.search(board, self.num_simulations)
        move_ids, probs = self.mcts.get_action_probs(root, temperature)

        policy = np.zeros(MOVE_SPACE, dtype=np.float32)
        for mid, p in zip(move_ids, probs):
            policy[mid] = p
        return policy

    def greedy_move(self, board: chess.Board) -> chess.Move:
        """Select best move greedily (temperature=0)."""
        return self.select_move(board, temperature=0.0)
