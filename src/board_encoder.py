import chess
import numpy as np
import torch

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a chess.Board to a [18, 8, 8] tensor.
    Planes:
    - 0-5: White pieces (P, N, B, R, Q, K)
    - 6-11: Black pieces (p, n, b, r, q, k)
    - 12: White castling rights (OO)
    - 13: White castling rights (OOO)
    - 14: Black castling rights (oo)
    - 15: Black castling rights (ooo)
    - 16: En passant square (if any)
    - 17: Turn (1 for white, 0 for black)
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    
    # Piece planes
    for color in [chess.WHITE, chess.BLACK]:
        color_offset = 0 if color == chess.WHITE else 6
        for piece_type in range(1, 7):
            squares = board.pieces(piece_type, color)
            for sq in squares:
                rank, file = divmod(sq, 8)
                tensor[color_offset + piece_type - 1, rank, file] = 1.0
                
    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
        
    # En passant
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        tensor[16, rank, file] = 1.0
        
    # Turn
    if board.turn == chess.WHITE:
        tensor[17, :, :] = 1.0
        
    return torch.from_numpy(tensor)
