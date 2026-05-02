import chess
import chess.pgn
import torch
import argparse
import os
from src.model import ChessNet
from src.engine import select_move

def generate_self_play_games(model, num_games, output_pgn, device='cpu'):
    with open(output_pgn, "a") as f:
        for _ in range(num_games):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Self Play"
            node = game
            
            moves = 0
            while not board.is_game_over() and moves < 200:
                move = select_move(model, board, device=device, epsilon=0.1)
                if move is None: break
                board.push(move)
                node = node.add_main_variation(move)
                moves += 1
                
            game.headers["Result"] = board.result()
            print(game, file=f, end="\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/self_play.pgn")
    parser.add_argument("--res_blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(num_residual_blocks=args.res_blocks, channels=args.channels).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    
    print(f"Generating {args.num_games} self-play games to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_self_play_games(model, args.num_games, args.output, device=device)
    print("Done.")
