import chess
import torch
from .engine import select_move

def test_checkmate_ability(model, device='cpu'):
    # King + Queen vs King (White to move)
    # White King: e1, White Queen: d1, Black King: e8
    board = chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1")
    moves = 0
    while not board.is_game_over() and moves < 20:
        if board.turn == chess.WHITE:
            move = select_move(model, board, device=device)
        else:
            # Random move for black
            move = list(board.legal_moves)[0]
        board.push(move)
        moves += 1
    
    is_mate = board.is_checkmate()
    return is_mate, moves

def test_check_detection(model, device='cpu'):
    # Position where checking is strong/only good move
    # White to move: Rook on e1, Black King on e8
    # 1. Re1+
    test_positions = [
        "4k3/8/8/8/8/8/8/4R2K w - - 0 1", # Re1+ is check
        "3k4/8/8/3Q4/8/8/8/3K4 w - - 0 1", # Qd7+ or Qd5+
    ]
    
    pass_count = 0
    for fen in test_positions:
        board = chess.Board(fen)
        move = select_move(model, board, device=device)
        if board.is_check(): # This checks if the position AFTER the move is a check? No, board.is_check() checks if side to move is in check.
            # We want to see if the move GIVES check.
            board.push(move)
            if board.is_check():
                pass_count += 1
                
    return pass_count / len(test_positions)

def run_tests(model, device='cpu'):
    print("Running Verification Tests...")
    mate_pass, mate_moves = test_checkmate_ability(model, device)
    check_rate = test_check_detection(model, device)
    
    print(f"Checkmate Test: {'PASS' if mate_pass else 'FAIL'} (Moves: {mate_moves})")
    print(f"Check Detection Rate: {check_rate*100:.1f}%")
    return mate_pass, check_rate
