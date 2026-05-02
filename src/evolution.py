import torch
import copy
import random
import chess
from .model import ChessNet
from .engine import select_move
import tqdm

class EvolutionManager:
    def __init__(self, population_size=8, elite_fraction=0.25, res_blocks=10, channels=128, device='cpu'):
        self.device = device
        self.population = [ChessNet(res_blocks, channels).to(device) for _ in range(population_size)]
        self.elite_fraction = elite_fraction
        self.population_size = population_size

    def play_game(self, agent_white, agent_black, max_moves=200):
        board = chess.Board()
        moves = 0
        while not board.is_game_over() and moves < max_moves:
            agent = agent_white if board.turn == chess.WHITE else agent_black
            move = select_move(agent, board, device=self.device)
            if move is None or move not in board.legal_moves:
                # This should not happen with masking, but as a fallback:
                return -1 if board.turn == chess.WHITE else 1 # Loss for the one who made invalid move
            board.push(move)
            moves += 1
        
        result = board.result()
        if result == "1-0": return 1
        if result == "0-1": return -1
        return 0

    def evaluate_population(self, games_per_pair=1):
        fitness = [0.0] * self.population_size
        indices = list(range(self.population_size))
        
        pairs = []
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                pairs.append((i, j))
        
        for i, j in tqdm.tqdm(pairs, desc="Evaluating population"):
            for _ in range(games_per_pair):
                # i as White
                res = self.play_game(self.population[i], self.population[j])
                if res == 1: fitness[i] += 1.0
                elif res == -1: fitness[j] += 1.0
                else: fitness[i] += 0.5; fitness[j] += 0.5
                
                # j as White
                res = self.play_game(self.population[j], self.population[i])
                if res == 1: fitness[j] += 1.0
                elif res == -1: fitness[i] += 1.0
                else: fitness[i] += 0.5; fitness[j] += 0.5
                
        return fitness

    def mutate(self, net, std=0.02):
        with torch.no_grad():
            for param in net.parameters():
                param.add_(torch.randn_like(param) * std)

    def crossover(self, parent_a, parent_b, alpha=0.5):
        child = copy.deepcopy(parent_a)
        with torch.no_grad():
            for cp, pb in zip(child.parameters(), parent_b.parameters()):
                cp.data = alpha * cp.data + (1 - alpha) * pb.data
        return child

    def select_and_reproduce(self, fitness):
        # Sort by fitness
        ranked_indices = sorted(range(self.population_size), key=lambda i: fitness[i], reverse=True)
        num_elites = int(self.population_size * self.elite_fraction)
        elites = [self.population[i] for i in ranked_indices[:num_elites]]
        
        new_population = []
        # Elitism
        new_population.extend([copy.deepcopy(e) for e in elites])
        
        # Fill remaining slots
        while len(new_population) < self.population_size:
            if random.random() < 0.5:
                # Crossover
                p1, p2 = random.sample(elites, 2)
                child = self.crossover(p1, p2)
            else:
                # Mutation
                parent = random.choice(elites)
                child = copy.deepcopy(parent)
                self.mutate(child)
            new_population.append(child)
            
        self.population = new_population
        return ranked_indices[0] # Index of the best agent

    def run_generation(self, gen_idx, games_per_pair=1):
        fitness = self.evaluate_population(games_per_pair)
        best_idx = self.select_and_reproduce(fitness)
        print(f"Gen {gen_idx} complete. Best fitness: {fitness[best_idx]}")
        return self.population[0] # Return the best (first in new pop due to elitism)
