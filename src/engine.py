import torch
import chess
from .board_encoder import board_to_tensor
from .move_encoder import get_move_index, index_to_move

def get_legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(4672, dtype=torch.bool)
    for move in board.legal_moves:
        idx = get_move_index(move, board.turn)
        mask[idx] = True
    return mask

def evaluate_board(model, board, device):
    """Returns the win probability from the perspective of the current player."""
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    _, value = model(tensor)
    return value.item()

def minimax(model, board, depth, alpha, beta, maximizing_player, device):
    if depth == 0 or board.is_game_over():
        # Evaluate from perspective of the side who just moved
        # Value head is trained to output from perspective of side to move
        val = evaluate_board(model, board, device)
        # If it's not the maximizing player's turn, we need to negate the value?
        # Actually, let's keep it simple: Value head returns V for the player to move.
        # If we reached here and it's maximizing_player's turn, we want to maximize that V.
        return val if maximizing_player else -val

    if maximizing_player:
        max_eval = -float('inf')
        # To keep "style", we could sort moves by policy head, but for depth 2-3, 
        # full legal moves is fine.
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(model, board, depth - 1, alpha, beta, False, device)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(model, board, depth - 1, alpha, beta, True, device)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def select_move(model, board, device='cpu', epsilon=0.0, depth=2):
    """
    Selects a move using a hybrid approach:
    1. Get top moves from Policy Head (Magnus Style)
    2. Use Minimax (depth 2) to verify they aren't blunders.
    """
    model.eval()
    with torch.no_grad():
        # Policy Head for "Instinct"
        tensor = board_to_tensor(board).unsqueeze(0).to(device)
        policy_logits, _ = model(tensor)
        
        mask = get_legal_move_mask(board).to(device)
        policy_logits[0, ~mask] = -1e9
        
        # Get top 5 moves to consider (keeps the "Magnus style")
        probs = torch.softmax(policy_logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(5, board.legal_moves.count()))
        
        best_move = None
        max_val = -float('inf')
        
        # Use Minimax to pick the best among the top "stylish" moves
        for idx in top_indices[0]:
            move = index_to_move(idx.item(), board)
            if move is None: continue
            
            board.push(move)
            # We want to minimize the opponent's response
            val = minimax(model, board, depth - 1, -float('inf'), float('inf'), False, device)
            board.pop()
            
            if val > max_val:
                max_val = val
                best_move = move
                
        return best_move if best_move else list(board.legal_moves)[0]
