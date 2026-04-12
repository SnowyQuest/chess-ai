import chess
from typing import List, Optional


# Move encoding: from_square * 64 + to_square
# This gives 4096 possible move IDs (sufficient for all legal moves)
MOVE_SPACE = 4096


def move_to_id(move: chess.Move) -> int:
    """Encode a move to an integer ID."""
    return move.from_square * 64 + move.to_square


def id_to_move(move_id: int) -> chess.Move:
    """Decode an integer ID to a Move (no promotion info)."""
    from_sq = move_id // 64
    to_sq = move_id % 64
    return chess.Move(from_sq, to_sq)


def legal_move_ids(board: chess.Board) -> List[int]:
    """Return list of legal move IDs for current board position."""
    return [move_to_id(m) for m in board.legal_moves]


def legal_moves_mask(board: chess.Board) -> 'np.ndarray':
    """Return boolean mask of shape (4096,) for legal moves."""
    import numpy as np
    mask = np.zeros(MOVE_SPACE, dtype=np.float32)
    for m in board.legal_moves:
        mask[move_to_id(m)] = 1.0
    return mask


def id_to_legal_move(move_id: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Convert a move ID to a legal chess.Move, handling promotions.
    If the decoded move matches a legal move (possibly with promotion), return it.
    """
    from_sq = move_id // 64
    to_sq = move_id % 64
    for legal in board.legal_moves:
        if legal.from_square == from_sq and legal.to_square == to_sq:
            return legal
    return None
