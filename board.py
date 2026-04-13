"""
board.py
========
Board encoding: 26 planes (8x8 each), dtype float32.

Plane layout
------------
 0-11  Piece positions (6 types x 2 colors) — unchanged
12     White attacked squares
13     Black attacked squares
14     Check flag (entire plane = 1.0 if side-to-move is in check)
15     Castling rights (4 specific squares)
16     En-passant target square
17     Repetition signal (0.5 = seen once, 1.0 = seen twice)
18     Fifty-move clock / 100
19     Full-move number / 200

NEW planes — fixing mate blindness and check-blunder:
20     King danger: squares adjacent to the side-to-move king
        that are attacked by the opponent  (1.0 = dangerous)
21     Hanging pieces owned by side-to-move  (1.0 = that piece is hanging)
22     Hanging pieces owned by opponent      (1.0 = can be captured for free)
23     Mate threat: squares the opponent could deliver check from next move
        (approximation: opponent pieces that currently attack king zone)
24     Mobility plane: number of legal moves / 50 (broadcast)
25     Side-to-move flag (1.0 = white, 0.0 = black, broadcast)

Why these planes fix the problems
----------------------------------
Plane 20 (king danger):
    The network can directly see when its king is under fire.
    Before this, "check bonus" caused blind check-chasing because the network
    could not distinguish a safe check from one that leaves its own king exposed.

Plane 21 (own hanging pieces):
    Network sees which of its pieces are undefended and can be captured.
    Prevents blunders caused by moving a piece to give check while leaving
    another piece hanging.

Plane 22 (opponent hanging pieces):
    Network sees free material to grab — improves tactical awareness.

Plane 23 (mate threat zone):
    Approximates where the opponent threatens to give check.
    Helps distinguish "this check walks into a mating net" from "safe check".

Plane 24 (mobility):
    Encodes degree of freedom. Near-zero mobility = near-zugzwang or mating net.
"""

import os
import numpy as np
import chess
from typing import Optional

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

MATERIAL_VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}

UNICODE_PIECES = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
}

BOARD_PLANES = 26


# ---------------------------------------------------------------------------
# Helper: least-valuable attacker (for hanging-piece detection)
# ---------------------------------------------------------------------------

def _least_valuable_attacker(board: chess.Board, sq: int, color: chess.Color):
    """Return the least-valuable piece of `color` that attacks `sq`, or None."""
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]:
        attackers = board.pieces(pt, color) & board.attackers(color, sq)
        if attackers:
            return attackers.pop()
    return None


def _is_hanging(board: chess.Board, sq: int) -> bool:
    """
    Return True if the piece on `sq` is hanging (can be captured with net gain).
    Uses a simplified Static Exchange Evaluation (SEE).
    """
    piece = board.piece_at(sq)
    if piece is None:
        return False

    attacker_color = not piece.color
    attacker_sq = _least_valuable_attacker(board, sq, attacker_color)
    if attacker_sq is None:
        return False  # not attacked at all

    # Gain from capturing
    gain = MATERIAL_VALUES.get(piece.piece_type, 0)

    # If gain > 0 and the square is not defended, it's hanging
    if not board.is_attacked_by(piece.color, sq):
        return True

    # Simplified: if attacker value < target value, profitable even if recaptured
    attacker_piece = board.piece_at(attacker_sq)
    if attacker_piece is None:
        return False
    attacker_val = MATERIAL_VALUES.get(attacker_piece.piece_type, 0)
    return attacker_val < gain


def _king_adjacent_squares(board: chess.Board, color: chess.Color):
    """Return set of squares adjacent to the king of `color`."""
    king_sq = board.king(color)
    if king_sq is None:
        return set()
    file_ = chess.square_file(king_sq)
    rank_ = chess.square_rank(king_sq)
    adjacent = set()
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            f2, r2 = file_ + df, rank_ + dr
            if 0 <= f2 <= 7 and 0 <= r2 <= 7:
                adjacent.add(chess.square(f2, r2))
    return adjacent


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

def encode_board(board: chess.Board) -> np.ndarray:
    """Encode board state to float32 array of shape (26, 8, 8)."""
    planes = np.zeros((BOARD_PLANES, 8, 8), dtype=np.float32)

    # --- planes 0-11: piece positions ---
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            planes[plane, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    # --- plane 12: white-attacked squares ---
    # --- plane 13: black-attacked squares ---
    for sq in chess.SQUARES:
        r, f = chess.square_rank(sq), chess.square_file(sq)
        if board.is_attacked_by(chess.WHITE, sq):
            planes[12, r, f] = 1.0
        if board.is_attacked_by(chess.BLACK, sq):
            planes[13, r, f] = 1.0

    # --- plane 14: check ---
    if board.is_check():
        planes[14, :, :] = 1.0

    # --- plane 15: castling rights ---
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[15, 0, 6] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[15, 0, 2] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, 7, 6] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15, 7, 2] = 1.0

    # --- plane 16: en-passant ---
    if board.ep_square is not None:
        planes[16, chess.square_rank(board.ep_square),
                    chess.square_file(board.ep_square)] = 1.0

    # --- plane 17: repetition signal ---
    rep = 0.0
    try:
        if board.is_repetition(2):
            rep = 1.0
        elif board.is_repetition(1):
            rep = 0.5
    except Exception:
        pass
    planes[17, :, :] = rep

    # --- plane 18: fifty-move clock ---
    planes[18, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    # --- plane 19: full-move number ---
    planes[19, :, :] = min(board.fullmove_number / 200.0, 1.0)

    # --- plane 20: king danger (our king's adjacent squares attacked by opponent) ---
    our_color = board.turn
    opp_color = not our_color
    king_zone = _king_adjacent_squares(board, our_color)
    for sq in king_zone:
        if board.is_attacked_by(opp_color, sq):
            planes[20, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    # --- plane 21: our hanging pieces ---
    # --- plane 22: opponent hanging pieces ---
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        r, f = chess.square_rank(sq), chess.square_file(sq)
        if _is_hanging(board, sq):
            if piece.color == our_color:
                planes[21, r, f] = 1.0
            else:
                planes[22, r, f] = 1.0

    # --- plane 23: opponent mate-threat zone ---
    # Squares adjacent to our king that the opponent attacks with multiple pieces
    # (proxy for how trapped our king is)
    for sq in king_zone:
        attackers = bin(int(board.attackers(opp_color, sq))).count('1')
        if attackers >= 2:
            planes[23, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    # --- plane 24: mobility (legal move count, broadcast) ---
    planes[24, :, :] = min(len(list(board.legal_moves)) / 50.0, 1.0)

    # --- plane 25: side to move (1 = white, 0 = black) ---
    planes[25, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return planes


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def material_balance(board: chess.Board) -> float:
    """Return material balance from white's perspective."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            val = MATERIAL_VALUES[piece.piece_type]
            score += val if piece.color == chess.WHITE else -val
    return score


def display_board(board: chess.Board, last_move: chess.Move = None) -> str:
    files = 'abcdefgh'
    header = '   ' + ' '.join(files)
    top    = '  +' + '-' * 17 + '+'
    bottom = '  +' + '-' * 17 + '+'
    last_squares = set()
    if last_move is not None:
        last_squares = {last_move.from_square, last_move.to_square}
    lines = [header, top]
    for rank in range(7, -1, -1):
        row = f'{rank + 1} |'
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            sym = UNICODE_PIECES.get(piece.symbol(), piece.symbol()) if piece else '.'
            row += f' {sym}'
        row += f' | {rank + 1}'
        lines.append(row)
    lines.append(bottom)
    lines.append(header)
    return '\n'.join(lines)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
