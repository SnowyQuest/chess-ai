import argparse
import torch
import os
from src.evolution import EvolutionManager
from src.tests import run_tests
from src.model import ChessNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--games_per_pair", type=int, default=5)
    parser.add_argument("--res_blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--load_best", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evolution starting on {device}")

    manager = EvolutionManager(
        population_size=args.population,
        res_blocks=args.res_blocks,
        channels=args.channels,
        device=device
    )

    if args.load_best and os.path.exists(args.load_best):
        print(f"Loading initial agent from {args.load_best}")
        state_dict = torch.load(args.load_best, map_location=device)
        for net in manager.population:
            net.load_state_dict(state_dict)

    os.makedirs("checkpoints", exist_ok=True)

    for gen in range(args.generations):
        print(f"\n--- Generation {gen} ---")
        best_agent = manager.run_generation(gen, games_per_pair=args.games_per_pair)
        
        # Save best agent
        torch.save(best_agent.state_dict(), f"checkpoints/best_gen_{gen}.pt")
        torch.save(best_agent.state_dict(), f"checkpoints/best_evolved.pt")
        
        # Verification
        mate_pass, check_rate = run_tests(best_agent, device=device)
        
        if mate_pass and check_rate >= 0.8:
            print(f"Generation {gen} is VIABLE!")

if __name__ == "__main__":
    main()
