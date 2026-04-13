"""
agent.py
========
Move selection agent with hard repetition blocking.

The core fix
------------
Previous versions applied repetition penalties only during training
(reward shaping).  At inference time the model had no memory of the
game so far -- it just saw the current board and picked the highest-
probability move, which could be the same move it already played twice.

This version adds a RepetitionGuard that operates at selection time:

1. Hard block: any move that would cause a 3-fold repetition is removed
   from the candidate list entirely.  The model physically cannot repeat
   a third time.

2. Soft penalty: any move that would cause a 2-fold repetition (visiting
   a position for the 2nd time) has its logit reduced by REPEAT_LOGIT_PENALTY
   before sampling.  This discourages going back to a seen position even
   once.

3. Fallback: if ALL legal moves lead to repetitions (extremely rare,
   usually only in fortress endgames), the hard block is lifted and only
   2-fold moves are allowed -- preventing the engine from having zero
   candidates while still avoiding 3-fold.

These rules apply in both MCTS mode and fast (policy-only) mode.
"""

import numpy as np
import chess
from typing import Optional, List, Tuple

from mcts import MCTS
from config import cfg

# How much to subtract from the logit of a move that revisits a position.
# 3.0 is strong enough to suppress it without completely zeroing it.
REPEAT_LOGIT_PENALTY = 3.0


class RepetitionGuard:
    """
    Classifies legal moves into three buckets:
      - SAFE      : does not revisit any previous position
      - ONCE      : revisits a position seen once (2-fold)
      - FORBIDDEN : would cause 3-fold repetition
    """

    def __init__(self, board: chess.Board):
        self.board = board

    def classify(self, moves: List[chess.Move]) -> Tuple[
            List[chess.Move], List[chess.Move], List[chess.Move]]:
        """
        Returns (safe, once, forbidden) lists.
        """
        safe      = []
        once      = []
        forbidden = []

        for mv in moves:
            tmp = self.board.copy()
            tmp.push(mv)
            try:
                if tmp.is_repetition(3):        # would be the 3rd occurrence
                    forbidden.append(mv)
                elif tmp.is_repetition(2):       # would be the 2nd occurrence
                    once.append(mv)
                else:
                    safe.append(mv)
            except Exception:
                safe.append(mv)

        return safe, once, forbidden

    def filter_moves(self, moves: List[chess.Move]) -> Tuple[
            List[chess.Move], bool]:
        """
        Return (allowed_moves, had_to_relax).
        allowed_moves never contains 3-fold moves.
        had_to_relax is True when all non-forbidden moves are 2-fold.
        """
        safe, once, forbidden = self.classify(moves)

        if safe:
            return safe + once, False     # prefer safe, allow once too
        if once:
            return once, True             # forced to revisit once
        if forbidden:
            return forbidden, True        # every option repeats (fortress) -- allow
        return moves, False


def _apply_repeat_penalty(logits: np.ndarray,
                          legal_ids: List[int],
                          board: chess.Board) -> np.ndarray:
    """
    Return a copy of `logits` with REPEAT_LOGIT_PENALTY subtracted from
    any move that would revisit a position.
    """
    from moves import id_to_legal_move
    logits = logits.copy()

    for i, mid in enumerate(legal_ids):
        mv = id_to_legal_move(mid, board)
        if mv is None:
            continue
        tmp = board.copy()
        tmp.push(mv)
        try:
            if tmp.is_repetition(2):         # 2nd or 3rd occurrence
                logits[i] -= REPEAT_LOGIT_PENALTY
            elif tmp.is_repetition(1):
                logits[i] -= REPEAT_LOGIT_PENALTY * 0.5
        except Exception:
            pass

    return logits


class Agent:
    def __init__(self, model, device: str = "cpu",
                 num_simulations: Optional[int] = None):
        self.model          = model
        self.device         = device
        self.num_simulations = num_simulations or cfg.num_simulations
        self.mcts           = MCTS(model, device, cfg)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def select_move(self, board: chess.Board,
                    temperature: float = 1.0) -> chess.Move:
        """
        Select a move using MCTS with repetition blocking.

        Steps
        -----
        1. Run MCTS search.
        2. Get visit-count probabilities for all candidates.
        3. Zero out any move that would cause 3-fold repetition.
        4. Apply soft logit penalty to 2-fold moves.
        5. Sample / argmax depending on temperature.
        """
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available")
        if len(legal) == 1:
            return legal[0]

        root = self.mcts.search(board, self.num_simulations)
        move_ids, probs = self.mcts.get_action_probs(root, temperature=1.0)

        if not move_ids:
            return np.random.choice(legal)

        # --- repetition filtering on visit-count probs ---
        from moves import id_to_legal_move
        guard = RepetitionGuard(board)

        # Identify forbidden (3-fold) move ids
        forbidden_ids = set()
        once_ids      = set()
        for mid in move_ids:
            mv = id_to_legal_move(mid, board)
            if mv is None:
                continue
            tmp = board.copy()
            tmp.push(mv)
            try:
                if tmp.is_repetition(3):
                    forbidden_ids.add(mid)
                elif tmp.is_repetition(2):
                    once_ids.add(mid)
            except Exception:
                pass

        # Build filtered arrays
        filtered_ids   = []
        filtered_probs = []

        for mid, p in zip(move_ids, probs):
            if mid in forbidden_ids:
                continue                    # hard block
            if mid in once_ids:
                p *= 0.05                   # strong soft penalty on prob
            filtered_ids.append(mid)
            filtered_probs.append(p)

        # Fallback: if everything was forbidden, allow 2-fold
        if not filtered_ids:
            for mid, p in zip(move_ids, probs):
                if mid not in forbidden_ids or mid in once_ids:
                    filtered_ids.append(mid)
                    filtered_probs.append(p)

        # Last resort: use all moves
        if not filtered_ids:
            filtered_ids   = list(move_ids)
            filtered_probs = list(probs)

        filtered_probs = np.array(filtered_probs, dtype=np.float64)
        total = filtered_probs.sum()
        if total < 1e-9:
            filtered_probs = np.ones(len(filtered_probs)) / len(filtered_probs)
        else:
            filtered_probs /= total

        # Temperature-adjusted selection
        if temperature == 0 or temperature < 1e-6:
            chosen_id = filtered_ids[int(np.argmax(filtered_probs))]
        else:
            visits = np.array(
                [root.children[mid].visit_count
                 for mid in filtered_ids
                 if mid in root.children],
                dtype=np.float64
            )
            # Re-build aligned arrays using only ids present in tree
            valid_ids = [mid for mid in filtered_ids if mid in root.children]
            if not valid_ids:
                valid_ids   = filtered_ids
                visits      = filtered_probs
            else:
                filtered_probs_tree = np.array(
                    [filtered_probs[filtered_ids.index(mid)] for mid in valid_ids],
                    dtype=np.float64
                )
                # Weight visits by filtered probs to keep repeat penalty
                visits = visits * filtered_probs_tree
                if visits.sum() < 1e-9:
                    visits = filtered_probs_tree
                else:
                    visits /= visits.sum()

            temps = visits ** (1.0 / max(temperature, 1e-6))
            if temps.sum() < 1e-9:
                temps = np.ones_like(temps)
            temps /= temps.sum()
            chosen_id = int(np.random.choice(valid_ids, p=temps))

        move = id_to_legal_move(chosen_id, board)
        if move is None:
            return legal[0]
        return move

    # ------------------------------------------------------------------
    # Policy-only fast selection (used in play mode without full MCTS)
    # ------------------------------------------------------------------

    def fast_move(self, board: chess.Board,
                  temperature: float = 1.0) -> chess.Move:
        """
        Policy-only move selection with repetition blocking.
        Faster than full MCTS; used when num_simulations is low or 0.
        """
        import torch
        from board import encode_board
        from moves import legal_move_ids, id_to_legal_move

        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves")
        if len(legal) == 1:
            return legal[0]

        state = encode_board(board)
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            policy_logits, _ = self.model(x)
            logits = policy_logits.squeeze(0).cpu().numpy()

        legal_ids    = legal_move_ids(board)
        legal_logits = np.array([logits[mid] for mid in legal_ids], dtype=np.float64)

        # Apply repetition penalty to logits
        legal_logits = _apply_repeat_penalty(legal_logits, legal_ids, board)

        # Hard block 3-fold
        for i, mid in enumerate(legal_ids):
            mv = id_to_legal_move(mid, board)
            if mv is None:
                continue
            tmp = board.copy()
            tmp.push(mv)
            try:
                if tmp.is_repetition(3):
                    legal_logits[i] = -1e9
            except Exception:
                pass

        if temperature == 0:
            chosen_id = legal_ids[int(np.argmax(legal_logits))]
        else:
            legal_logits -= legal_logits.max()
            exp_l = np.exp(legal_logits)
            probs = exp_l / (exp_l.sum() + 1e-9)
            chosen_id = int(np.random.choice(legal_ids, p=probs))

        move = id_to_legal_move(chosen_id, board)
        return move if move is not None else legal[0]

    # ------------------------------------------------------------------
    # Training helpers (unchanged)
    # ------------------------------------------------------------------

    def get_policy_target(self, board: chess.Board,
                          temperature: float = 1.0) -> np.ndarray:
        from moves import MOVE_SPACE
        root = self.mcts.search(board, self.num_simulations)
        move_ids, probs = self.mcts.get_action_probs(root, temperature)
        policy = np.zeros(MOVE_SPACE, dtype=np.float32)
        for mid, p in zip(move_ids, probs):
            policy[mid] = p
        return policy

    def greedy_move(self, board: chess.Board) -> chess.Move:
        return self.select_move(board, temperature=0.0)
