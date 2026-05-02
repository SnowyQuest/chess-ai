from src.train import train
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--res_blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache", type=str, default=None, help="Path to dataset cache file (.pt)")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing model checkpoint (.pt) to continue training")
    args = parser.parse_args()
    
    os.makedirs("checkpoints", exist_ok=True)
    train(args)
