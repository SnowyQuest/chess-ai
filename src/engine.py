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

def minimax(model, board, depth, alpha, beta, maximizing_player, device, top_k=5):
    if depth == 0 or board.is_game_over():
        return evaluate_board(model, board, device)

    # Get policy logits to order/filter moves
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    policy_logits, _ = model(tensor)
    mask = get_legal_move_mask(board).to(device)
    policy_logits[0, ~mask] = -1e9
    
    # Filter to top_k moves
    probs = torch.softmax(policy_logits, dim=1)
    k = min(top_k, board.legal_moves.count())
    _, top_indices = torch.topk(probs, k=k)

    if maximizing_player:
        max_eval = -float('inf')
        for idx in top_indices[0]:
            move = index_to_move(idx.item(), board)
            if move is None: continue
            board.push(move)
            eval = minimax(model, board, depth - 1, alpha, beta, False, device, top_k=max(2, top_k-1))
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for idx in top_indices[0]:
            move = index_to_move(idx.item(), board)
            if move is None: continue
            board.push(move)
            eval = minimax(model, board, depth - 1, alpha, beta, True, device, top_k=max(2, top_k-1))
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def select_move(model, board, device='cpu', epsilon=0.0, depth=2):
    """
    Selects a move using a high-speed hybrid approach:
    - Branching is limited to top-k 'Magnus' moves.
    - Depth is kept shallow for human-like response times.
    """
    model.eval()
    with torch.no_grad():
        # Get moves from the perspective of the player to move
        tensor = board_to_tensor(board).unsqueeze(0).to(device)
        policy_logits, _ = model(tensor)
        mask = get_legal_move_mask(board).to(device)
        policy_logits[0, ~mask] = -1e9
        
        probs = torch.softmax(policy_logits, dim=1)
        # Search the top 5 moves for the best tactical result
        k = min(5, board.legal_moves.count())
        _, top_indices = torch.topk(probs, k=k)
        
        best_move = None
        max_val = -float('inf')
        
        for idx in top_indices[0]:
            move = index_to_move(idx.item(), board)
            if move is None: continue
            
            board.push(move)
            # Minimize opponent's best response (within their top moves)
            val = minimax(model, board, depth - 1, -float('inf'), float('inf'), False, device, top_k=3)
            board.pop()
            
            if val > max_val:
                max_val = val
                best_move = move
                
        return best_move if best_move else list(board.legal_moves)[0]
