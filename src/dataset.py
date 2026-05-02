import chess.pgn
import torch
from torch.utils.data import Dataset
from .board_encoder import board_to_tensor
from .move_encoder import get_move_index
import os

class ChessDataset(Dataset):
    def __init__(self, pgn_files, max_samples=None, deduplicate=True, cache_path=None):
        self.samples = []
        
        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading dataset from cache: {cache_path}")
            self.samples = torch.load(cache_path)
            if max_samples:
                self.samples = self.samples[:max_samples]
            return

        print("Parsing PGN files (this may take a while)...")
        seen_fens = set() if deduplicate else None
        for pgn_file in pgn_files:
            if not os.path.exists(pgn_file):
                print(f"Warning: {pgn_file} not found.")
                continue
            with open(pgn_file) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    outcome = game.headers.get("Result")
                    if outcome == "1-0":
                        res = 1.0
                    elif outcome == "0-1":
                        res = -1.0
                    else:
                        res = 0.0
                        
                    board = game.board()
                    for move in game.mainline_moves():
                        # Use first 4 fields of FEN
                        fen_parts = board.fen().split()
                        clean_fen = " ".join(fen_parts[:4]) + " 0 1"
                        
                        if not deduplicate or clean_fen not in seen_fens:
                            current_outcome = res if board.turn == chess.WHITE else -res
                            
                            try:
                                move_idx = get_move_index(move, board.turn)
                                # Store as tuple of (fen, move_idx, outcome)
                                self.samples.append((clean_fen, move_idx, current_outcome))
                                if deduplicate:
                                    seen_fens.add(clean_fen)
                            except Exception:
                                pass
                            
                        board.push(move)
                        if max_samples and len(self.samples) >= max_samples:
                            break
                    if max_samples and len(self.samples) >= max_samples:
                        break
        
        # Save to cache if requested
        if cache_path:
            print(f"Saving dataset to cache: {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.samples, cache_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, move_idx, outcome = self.samples[idx]
        board = chess.Board(fen)
        tensor = board_to_tensor(board)
        return tensor, move_idx, torch.tensor([outcome], dtype=torch.float32)
