import torch
import argparse
from src.model import ChessNet
from src.tests import run_tests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--res_blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(num_residual_blocks=args.res_blocks, channels=args.channels).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    
    run_tests(model, device=device)
