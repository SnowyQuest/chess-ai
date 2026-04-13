from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Model
    num_res_blocks: int = 4
    num_filters: int = 64
    policy_output_size: int = 4096  # 64 * 64

    # MCTS
    num_simulations: int = 100
    c_puct: float = 1.4
    temperature: float = 1.0
    temperature_drop_move: int = 20  # after this move, use greedy

    # Self-play
    num_workers: int = 2
    max_game_length: int = 200
    games_per_iteration: int = 10

    # Replay buffer
    buffer_size: int = 50000
    min_buffer_size: int = 512

    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_steps_per_iter: int = 50
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Reward shaping
    material_reward_factor: float = 0.05
    check_bonus: float = 0.01
    castle_bonus: float = 0.02

    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 100  # steps

    # Elo evaluation
    eval_interval: int = 200  # steps
    eval_games: int = 10
    initial_elo: float = 1000.0

    # Loop
    total_iterations: int = 10000
    device: str = "cuda"

    # Play mode
    play_simulations: int = 50


cfg = Config()
