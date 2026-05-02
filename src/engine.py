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

def negamax(model, board, depth, alpha, beta, device, top_k=5):
    if depth == 0 or board.is_game_over():
        # evaluate_board returns score for player to move.
        # In negamax, we return the value relative to the player to move.
        return evaluate_board(model, board, device)

    # 1. Get Policy Suggestions
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    policy_logits, _ = model(tensor)
    mask = get_legal_move_mask(board).to(device)
    policy_logits[0, ~mask] = -1e9
    
    probs = torch.softmax(policy_logits, dim=1)
    k = min(top_k, board.legal_moves.count())
    _, top_indices = torch.topk(probs, k=k)

    # 2. Candidate Moves = Top Policy Moves + ALL Captures
    # (Captures ensure we don't miss "eating" blundered pieces)
    candidate_moves = []
    seen_moves = set()
    
    # Add top policy moves
    for idx in top_indices[0]:
        move = index_to_move(idx.item(), board)
        if move and move in board.legal_moves:
            candidate_moves.append(move)
            seen_moves.add(move.uci())
            
    # Add any capture moves that weren't in the top-k
    for move in board.legal_moves:
        if board.is_capture(move) and move.uci() not in seen_moves:
            candidate_moves.append(move)
            seen_moves.add(move.uci())

    max_val = -float('inf')
    for move in candidate_moves:
        board.push(move)
        # Value is negated for the next player
        val = -negamax(model, board, depth - 1, -beta, -alpha, device, max(2, top_k - 1))
        board.pop()
        
        if val > max_val:
            max_val = val
        alpha = max(alpha, val)
        if alpha >= beta:
            break
            
    return max_val

def select_move(model, board, device='cpu', epsilon=0.0, depth=2):
    model.eval()
    with torch.no_grad():
        # Root search
        tensor = board_to_tensor(board).unsqueeze(0).to(device)
        policy_logits, _ = model(tensor)
        mask = get_legal_move_mask(board).to(device)
        policy_logits[0, ~mask] = -1e9
        
        probs = torch.softmax(policy_logits, dim=1)
        k = min(5, board.legal_moves.count())
        _, top_indices = torch.topk(probs, k=k)
        
        # Build root candidate list: Top Policy + All Captures
        candidate_moves = []
        seen_moves = set()
        for idx in top_indices[0]:
            move = index_to_move(idx.item(), board)
            if move:
                candidate_moves.append(move)
                seen_moves.add(move.uci())
        for move in board.legal_moves:
            if board.is_capture(move) and move.uci() not in seen_moves:
                candidate_moves.append(move)
        
        best_move = None
        max_val = -float('inf')
        
        for move in candidate_moves:
            board.push(move)
            # We are currently searching for the best response to 'move'
            val = -negamax(model, board, depth - 1, -float('inf'), float('inf'), device, top_k=3)
            board.pop()
            
            if val > max_val:
                max_val = val
                best_move = move
                
        return best_move if best_move else list(board.legal_moves)[0]
