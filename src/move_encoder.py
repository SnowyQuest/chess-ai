import chess

# The AlphaZero encoding for moves is 8x8x73.
# 8x8 squares of origin.
# 73 planes:
# - 56 planes for queen-like moves: 7 squares in 8 directions.
# - 8 planes for knight moves.
# - 9 planes for underpromotions: (3 directions: left-diagonal, forward, right-diagonal) x (3 piece types: Knight, Bishop, Rook).
# Total = 8 * 8 * 73 = 4672.

def get_move_index(move: chess.Move, color: chess.Color) -> int:
    """
    Encodes a chess.Move into an index 0-4671.
    Note: For simplicity and consistency, we always encode from the perspective of the side to move.
    However, the requirement says "all legal UCI moves". 
    Actually, a simpler way to map 4672 moves is to use the standard AlphaZero mapping.
    """
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    # Flip squares if black to move? 
    # Standard approach is to flip the board so it's always "white" to move.
    # But let's stick to absolute squares for now if the requirement implies it.
    # Wait, the requirement says "4672 possible moves".
    
    # Let's implement a robust mapping.
    from_rank, from_file = divmod(from_square, 8)
    to_rank, to_file = divmod(to_square, 8)
    
    dr = to_rank - from_rank
    df = to_file - from_file
    
    # Underpromotions
    if promotion and promotion != chess.QUEEN:
        # 9 planes for underpromotions
        # Directions: df in [-1, 0, 1]
        # Pieces: Knight, Bishop, Rook
        promo_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[promotion]
        direction_idx = df + 1
        plane = 64 + promo_idx * 3 + direction_idx
    elif (abs(dr) == abs(df) or dr == 0 or df == 0) and not (promotion and promotion != chess.QUEEN):
        # Queen moves
        # 8 directions, up to 7 squares
        # Directions: (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)
        directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
        dist = max(abs(dr), abs(df))
        direction = (dr // dist, df // dist)
        direction_idx = directions.index(direction)
        plane = direction_idx * 7 + (dist - 1)
    else:
        # Knight moves
        # 8 directions
        knight_moves = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]
        plane = 56 + knight_moves.index((dr, df))
        
    return from_square * 73 + plane

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Decodes an index 0-4671 into a chess.Move.
    """
    from_square = index // 73
    plane = index % 73
    
    from_rank, from_file = divmod(from_square, 8)
    
    if plane < 56:
        # Queen moves
        direction_idx = plane // 7
        dist = (plane % 7) + 1
        directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
        dr, df = directions[direction_idx]
        to_rank, to_file = from_rank + dr * dist, from_file + df * dist
        promotion = None
        # Check for promotion
        if to_rank in [0, 7]:
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                promotion = chess.QUEEN
    elif plane < 64:
        # Knight moves
        knight_idx = plane - 56
        knight_moves = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]
        dr, df = knight_moves[knight_idx]
        to_rank, to_file = from_rank + dr, from_file + df
        promotion = None
    else:
        # Underpromotions
        under_idx = plane - 64
        promo_type_idx = under_idx // 3
        direction_idx = under_idx % 3
        df = direction_idx - 1
        dr = 1 if board.turn == chess.WHITE else -1
        to_rank, to_file = from_rank + dr, from_file + df
        promotion = [chess.KNIGHT, chess.BISHOP, chess.ROOK][promo_type_idx]
        
    if 0 <= to_rank < 8 and 0 <= to_file < 8:
        to_square = to_rank * 8 + to_file
        return chess.Move(from_square, to_square, promotion)
    return None # Invalid move representation
