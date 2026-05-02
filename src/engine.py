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

def select_move(model, board, device='cpu', epsilon=0.0):
    model.eval()
    with torch.no_grad():
        tensor = board_to_tensor(board).unsqueeze(0).to(device)
        policy_logits, _ = model(tensor)
        
        mask = get_legal_move_mask(board).to(device)
        # Apply mask: set illegal move logits to a very small number
        policy_logits[0, ~mask] = -1e9
        
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            # Epsilon-greedy or sampling can be used
            # For pure strength, argmax is better, but for evolution/training, sampling might help
            probs = torch.softmax(policy_logits, dim=1)
            move_idx = torch.multinomial(probs, 1).item()
        else:
            move_idx = torch.argmax(policy_logits, dim=1).item()
            
        move = index_to_move(move_idx, board)
        return move
