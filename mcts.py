import math
import numpy as np
import chess
from typing import Dict, List, Optional, Tuple

from board import encode_board
from moves import move_to_id, id_to_legal_move, legal_move_ids, legal_moves_mask
from config import cfg


class Node:
    __slots__ = ["board", "parent", "move", "children", "visit_count",
                 "value_sum", "prior", "is_expanded"]

    def __init__(self, board: chess.Board, parent: Optional["Node"] = None,
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: Dict[int, "Node"] = {}  # move_id -> Node
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.is_expanded: bool = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value + exploration

    def select_child(self, c_puct: float) -> "Node":
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))

    def expand(self, policy_priors: np.ndarray):
        """Expand node with policy priors over legal moves."""
        legal_ids = legal_move_ids(self.board)
        if not legal_ids:
            self.is_expanded = True
            return

        # Mask and normalize priors over legal moves
        priors = np.array([policy_priors[mid] for mid in legal_ids], dtype=np.float64)
        priors = priors - priors.max()  # numerical stability
        priors = np.exp(priors)
        priors /= priors.sum() + 1e-8

        for i, mid in enumerate(legal_ids):
            move = id_to_legal_move(mid, self.board)
            if move is None:
                continue
            child_board = self.board.copy()
            child_board.push(move)
            self.children[mid] = Node(child_board, parent=self, move=move, prior=float(priors[i]))

        self.is_expanded = True

    def backup(self, value: float):
        """Propagate value up the tree (negate at each level for two-player)."""
        node = self
        sign = 1.0
        while node is not None:
            node.visit_count += 1
            node.value_sum += sign * value
            sign = -sign
            node = node.parent


class MCTS:
    def __init__(self, model, device: str = "cpu", cfg=cfg):
        self.model = model
        self.device = device
        self.cfg = cfg

    def _evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """Evaluate position using neural network."""
        enc = encode_board(board)
        policy_logits, value = self.model.predict(enc, self.device)
        return policy_logits, value

    def _is_terminal(self, board: chess.Board) -> Tuple[bool, float]:
        """Check if position is terminal and return value."""
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return True, 0.0
            # Value from perspective of the side to move
            # If the game is over, current side just got mated or similar
            winner = outcome.winner
            if winner == board.turn:
                return True, 1.0
            else:
                return True, -1.0
        if board.is_game_over():
            return True, 0.0
        return False, 0.0

    def search(self, root_board: chess.Board, num_simulations: Optional[int] = None) -> Node:
        """Run MCTS and return root node with visit counts."""
        n_sims = num_simulations or self.cfg.num_simulations
        root = Node(root_board.copy())

        # Expand root immediately
        policy_logits, value = self._evaluate(root_board)
        root.expand(policy_logits)
        root.visit_count = 1

        for _ in range(n_sims):
            node = root

            # --- Selection ---
            while node.is_expanded and node.children:
                node = node.select_child(self.cfg.c_puct)

            # --- Terminal check ---
            terminal, term_value = self._is_terminal(node.board)
            if terminal:
                node.backup(term_value)
                continue

            # --- Expansion & Evaluation ---
            policy_logits, value = self._evaluate(node.board)
            node.expand(policy_logits)
            node.backup(value)

        return root

    def get_action_probs(self, root: Node, temperature: float = 1.0) -> Tuple[List[int], np.ndarray]:
        """
        Return (move_ids, probabilities) based on visit counts.
        """
        move_ids = list(root.children.keys())
        if not move_ids:
            return [], np.array([])

        visits = np.array([root.children[mid].visit_count for mid in move_ids], dtype=np.float64)

        if temperature == 0 or temperature < 1e-6:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()

        return move_ids, probs
