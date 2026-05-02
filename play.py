import chess
import torch
import argparse
from src.model import ChessNet
from src.engine import select_move

def play(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(num_residual_blocks=args.res_blocks, channels=args.channels).to(device)
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    board = chess.Board()
    ai_color = chess.WHITE if args.color == 'white' else chess.BLACK

    while not board.is_game_over():
        print("\n", board)
        if board.turn == ai_color:
            move = select_move(model, board, device=device)
            print(f"\nAI moves: {move}")
            board.push(move)
        else:
            move_str = input("\nEnter your move (UCI): ")
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move!")
            except:
                print("Invalid format!")
    
    print("\nGame Over!")
    print("Result:", board.result())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"])
    parser.add_argument("--res_blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    args = parser.parse_args()
    play(args)
