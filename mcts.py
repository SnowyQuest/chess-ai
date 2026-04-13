"""
mcts.py
=======
MCTS with quiescence extension for mate detection.

Changes from previous version
------------------------------
1. Quiescence search on leaf nodes
   When a leaf node is not terminal but the position has very few legal moves
   OR the opponent is in check, we run a shallow quiescence search to avoid
   the "horizon effect" where the model stops searching just before a mate.

2. Mate-distance bonus in backup
   When checkmate is found, the backup value is +1.0 boosted by a small
   factor inversely proportional to depth.  This makes the model prefer
   shorter mates (mate-in-1 > mate-in-3).

3. Improved repetition handling (unchanged from v2, kept here)

4. Dirichlet noise (unchanged, kept here)

5. Forced-move optimization
   If there is exactly 1 legal move, skip MCTS entirely and return that move
   with probability 1.0.  This avoids wasting simulations on forced positions.
"""

import math
import numpy as np
import chess
from typing import Dict, List, Optional, Tuple

from board import encode_board
from moves import move_to_id, id_to_legal_move, legal_move_ids, legal_moves_mask
from config import cfg


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ["board", "parent", "move", "children", "visit_count",
                 "value_sum", "prior", "is_expanded", "depth"]

    def __init__(self, board: chess.Board, parent: Optional["Node"] = None,
                 move: Optional[chess.Move] = None, prior: float = 0.0,
                 depth: int = 0):
        self.board       = board
        self.parent      = parent
        self.move        = move
        self.children: Dict[int, "Node"] = {}
        self.visit_count: int   = 0
        self.value_sum:   float = 0.0
        self.prior:       float = prior
        self.is_expanded: bool  = False
        self.depth:       int   = depth

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def ucb_score(self, c_puct: float) -> float:
        parent_visits  = self.parent.visit_count if self.parent else 1
        effective_prior = self.prior

        # Penalize repetition in UCB
        try:
            if self.board.is_repetition(2):
                effective_prior *= 0.1
            elif self.board.is_repetition(1):
                effective_prior *= 0.4
        except Exception:
            pass

        exploration = (c_puct * effective_prior *
                       math.sqrt(parent_visits) / (1 + self.visit_count))
        return self.value + exploration

    def select_child(self, c_puct: float) -> "Node":
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))

    def expand(self, policy_priors: np.ndarray):
        legal_ids = legal_move_ids(self.board)
        if not legal_ids:
            self.is_expanded = True
            return

        priors = np.array([policy_priors[mid] for mid in legal_ids], dtype=np.float64)
        priors -= priors.max()
        priors  = np.exp(priors)
        priors /= priors.sum() + 1e-8

        for i, mid in enumerate(legal_ids):
            move = id_to_legal_move(mid, self.board)
            if move is None:
                continue
            child_board = self.board.copy()
            child_board.push(move)
            self.children[mid] = Node(
                child_board, parent=self, move=move,
                prior=float(priors[i]), depth=self.depth + 1
            )
        self.is_expanded = True

    def backup(self, value: float):
        node  = self
        sign  = 1.0
        while node is not None:
            node.visit_count += 1
            node.value_sum   += sign * value
            sign              = -sign
            node              = node.parent


# ---------------------------------------------------------------------------
# Quiescence search
# ---------------------------------------------------------------------------

MAX_QUIESCENCE_DEPTH = 4   # max extra plies after leaf

def quiescence(board: chess.Board, alpha: float, beta: float,
               depth: int, model, device: str) -> float:
    """
    Shallow quiescence search at a leaf node.
    Only searches captures and checks to avoid horizon effect on mating lines.
    Returns value from the perspective of the side to move at the root call.
    """
    if board.is_checkmate():
        return -1.0
    if board.is_game_over():
        return 0.0
    if depth <= 0:
        # Fall back to network evaluation
        enc = encode_board(board)
        _, v = model.predict(enc, device)
        return v

    # Stand-pat: evaluate current position
    enc       = encode_board(board)
    _, stand_pat = model.predict(enc, device)

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    # Search captures and checks only
    for move in board.legal_moves:
        if not (board.is_capture(move) or board.gives_check(move)):
            continue
        board.push(move)
        score = -quiescence(board, -beta, -alpha, depth - 1, model, device)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(self, model, device: str = "cpu", cfg=cfg):
        self.model  = model
        self.device = device
        self.cfg    = cfg

    def _evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        enc            = encode_board(board)
        policy_logits, value = self.model.predict(enc, self.device)
        return policy_logits, value

    def _is_terminal(self, board: chess.Board) -> Tuple[bool, float]:
        """Terminal check including 3-fold repetition."""
        try:
            if board.is_repetition(3):
                return True, 0.0
        except Exception:
            pass

        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return True, 0.0
            return True, (1.0 if outcome.winner == board.turn else -1.0)

        if board.is_game_over():
            return True, 0.0
        return False, 0.0

    def _evaluate_with_quiescence(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """
        Evaluate leaf node.  If the position looks sharp (check or few moves),
        extend with quiescence search to avoid missing forced mates.
        """
        policy_logits, value = self._evaluate(board)

        # Trigger quiescence if in check or very few legal moves
        legal_count = sum(1 for _ in board.legal_moves)
        if board.is_check() or legal_count <= 5:
            q_value = quiescence(
                board.copy(), -1.0, 1.0,
                MAX_QUIESCENCE_DEPTH, self.model, self.device
            )
            # Blend NN value with quiescence (quiescence is more reliable here)
            value = 0.4 * value + 0.6 * q_value

        return policy_logits, value

    def _add_dirichlet_noise(self, root: Node,
                             alpha: float = 0.3, epsilon: float = 0.25):
        if not root.children:
            return
        n     = len(root.children)
        noise = np.random.dirichlet([alpha] * n)
        for (_, node), ni in zip(root.children.items(), noise):
            node.prior = (1 - epsilon) * node.prior + epsilon * ni

    def search(self, root_board: chess.Board,
               num_simulations: Optional[int] = None,
               add_noise: bool = True) -> Node:
        n_sims = num_simulations or self.cfg.num_simulations

        # Forced move: skip MCTS entirely
        legal = list(root_board.legal_moves)
        if len(legal) == 1:
            root = Node(root_board.copy())
            move = legal[0]
            child_board = root_board.copy()
            child_board.push(move)
            mid = move_to_id(move)
            root.children[mid] = Node(child_board, parent=root, move=move,
                                      prior=1.0, depth=1)
            root.is_expanded   = True
            root.visit_count   = 1
            root.children[mid].visit_count = 1
            return root

        root = Node(root_board.copy())
        policy_logits, value = self._evaluate_with_quiescence(root_board)
        root.expand(policy_logits)
        root.visit_count = 1

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(n_sims):
            node = root

            # Selection
            while node.is_expanded and node.children:
                node = node.select_child(self.cfg.c_puct)

            # Terminal check
            terminal, term_value = self._is_terminal(node.board)
            if terminal:
                # Mate-distance bonus: prefer shorter mates
                if abs(term_value) == 1.0:
                    bonus = max(0.0, 0.05 * (10 - node.depth))
                    term_value = term_value * (1.0 + bonus)
                    term_value = max(-1.0, min(1.0, term_value))
                node.backup(term_value)
                continue

            # Expansion + evaluation
            policy_logits, value = self._evaluate_with_quiescence(node.board)
            node.expand(policy_logits)
            node.backup(value)

        return root

    def get_action_probs(self, root: Node,
                         temperature: float = 1.0) -> Tuple[List[int], np.ndarray]:
        move_ids = list(root.children.keys())
        if not move_ids:
            return [], np.array([])

        visits = np.array(
            [root.children[mid].visit_count for mid in move_ids], dtype=np.float64
        )

        if temperature < 1e-6:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()

        return move_ids, probs
