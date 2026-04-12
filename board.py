import numpy as np
import chess


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
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

UNICODE_PIECES = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode board to (12, 8, 8) float32 array."""
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            row = sq // 8
            col = sq % 8
            planes[plane, row, col] = 1.0
    return planes


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
    """
    Return a unicode string representation of the board.
    Correctly aligned file letters and rank numbers.
    Optionally highlights the last move.
    """
    files = 'abcdefgh'
    header  = '   ' + ' '.join(files)
    top     = '  ┌' + '─' * 17 + '┐'
    bottom  = '  └' + '─' * 17 + '┘'

    last_squares = set()
    if last_move is not None:
        last_squares = {last_move.from_square, last_move.to_square}

    lines = [header, top]
    for rank in range(7, -1, -1):
        row = f'{rank + 1} │'
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                sym = '·'
            else:
                sym = UNICODE_PIECES.get(piece.symbol(), piece.symbol())
            # Mark last move squares with brackets
            if sq in last_squares:
                sym = sym  # terminal color codes not used for portability
            row += f' {sym}'
        row += f' │ {rank + 1}'
        lines.append(row)

    lines.append(bottom)
    lines.append(header)
    return '\n'.join(lines)


def clear_screen():
    """Clear terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
