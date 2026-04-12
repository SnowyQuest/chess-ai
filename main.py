"""
Chess AI - Main Entry Point

Usage:
    python main.py play               # Human vs AI
    python main.py selfplay           # Run self-play data collection
    python main.py train              # Train from replay buffer
    python main.py loop               # Full training loop
    python main.py train_pgn file.pgn # Supervised PGN pretraining
"""

import sys
import os
import copy
import signal
import argparse

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import chess

from config import cfg
from model import ChessNet
from replay_buffer import ReplayBuffer
from trainer import Trainer
from checkpoint import save_checkpoint, load_checkpoint, save_best_model, load_best_model
from elo_tracker import EloTracker, evaluate_models
from logger import get_logger

log = get_logger("main")



# ─────────────────────────────────────────────────
# Data: Kayıtlı veriyi listele
# ─────────────────────────────────────────────────

def cmd_data(args):
    from data_store import list_data, load_samples
    list_data()


# ─────────────────────────────────────────────────
# Play Mode: Human vs AI
# ─────────────────────────────────────────────────

def cmd_play(args):
    from board import display_board
    from agent import Agent

    device = cfg.device
    model = ChessNet().to(device)
    load_checkpoint(model)

    agent = Agent(model, device, num_simulations=cfg.play_simulations)
    board = chess.Board()

    print("\n♟  Chess AI - Human vs AI  ♟")
    print("Enter moves in UCI format (e.g. e2e4, e1g1 for castling)")
    print("Type 'quit' to exit, 'board' to redisplay\n")

    human_color = chess.WHITE
    ai_color = chess.BLACK

    while not board.is_game_over():
        print(display_board(board))
        print()

        if board.turn == human_color:
            # Human move
            while True:
                try:
                    raw = input("Your move: ").strip().lower()
                except EOFError:
                    print("Goodbye!")
                    return

                if raw == "quit":
                    print("Goodbye!")
                    return
                if raw == "board":
                    print(display_board(board))
                    continue

                try:
                    move = chess.Move.from_uci(raw)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print(f"Illegal move: {raw}. Legal moves: "
                              f"{', '.join(m.uci() for m in list(board.legal_moves)[:10])}")
                except ValueError:
                    print(f"Invalid UCI format: {raw}")
        else:
            # AI move
            print("AI is thinking...")
            temperature = 0.0  # greedy in play mode
            move = agent.select_move(board, temperature=temperature)
            print(f"AI plays: {move.uci()}")
            board.push(move)

    print(display_board(board))
    outcome = board.outcome()
    if outcome is None:
        print("Game over (max moves reached) - Draw")
    elif outcome.winner is None:
        print(f"Draw! Reason: {outcome.termination.name}")
    elif outcome.winner == human_color:
        print("You win! 🎉")
    else:
        print("AI wins! 🤖")


# ─────────────────────────────────────────────────
# Self-play data collection
# ─────────────────────────────────────────────────

def cmd_selfplay(args):
    from selfplay_worker import collect_selfplay_single

    device = cfg.device
    model = ChessNet().to(device)
    load_checkpoint(model)
    model.eval()

    n_games = getattr(args, "games", cfg.games_per_iteration)
    watch   = getattr(args, "watch", False)
    fast    = getattr(args, "fast", False)

    # Hız ipuçları
    if not fast and not watch:
        log.info("İpucu: --fast ile policy-only mod kullanarak çok daha hızlı çalışır.")
        log.info("       --watch ile tahtayı canlı takip edebilirsin.")

    mode_str = "fast (policy-only)" if fast else f"MCTS ({cfg.num_simulations} sim)"
    log.info(f"Running {n_games} self-play game(s) | mode={mode_str} | watch={watch}")

    samples = collect_selfplay_single(model, n_games, device, watch=watch, fast=fast)
    log.info(f"Collected {len(samples)} training samples from {n_games} games")

    # Diske kaydet
    from data_store import save_samples
    save_samples(samples)


# ─────────────────────────────────────────────────
# Training from replay buffer
# ─────────────────────────────────────────────────

def cmd_train(args):
    from selfplay_worker import collect_selfplay_single

    device = cfg.device
    model = ChessNet().to(device)
    trainer = Trainer(model, device)
    trainer.step = load_checkpoint(model, trainer.optimizer)

    replay_buffer = ReplayBuffer(cfg.buffer_size)

    # Önce diskten yükle
    from data_store import save_samples, load_samples
    saved = load_samples()
    if saved:
        replay_buffer.add_many(saved)
        log.info(f"Diskten {len(saved)} örnek yüklendi → buffer: {len(replay_buffer)}")

    # Yetmiyorsa self-play çalıştır
    if not replay_buffer.is_ready(cfg.min_buffer_size):
        log.info("Buffer yetersiz, self-play başlatılıyor...")
        from selfplay_worker import collect_selfplay_single
        while not replay_buffer.is_ready(cfg.min_buffer_size):
            more = collect_selfplay_single(model, cfg.games_per_iteration, device)
            replay_buffer.add_many(more)
            save_samples(more)
            log.info(f"Buffer size: {len(replay_buffer)}")

    # Train
    n_steps = getattr(args, "steps", cfg.train_steps_per_iter)
    log.info(f"Training for {n_steps} steps...")
    pl, vl = trainer.train_n_steps(replay_buffer, n_steps)
    log.info(f"Training done. Policy loss: {pl:.4f} | Value loss: {vl:.4f}")

    save_checkpoint(model, trainer.optimizer, trainer.step)


# ─────────────────────────────────────────────────
# PGN Supervised Pretraining
# ─────────────────────────────────────────────────

def cmd_train_pgn(args):
    device = cfg.device
    model = ChessNet().to(device)
    trainer = Trainer(model, device)
    trainer.step = load_checkpoint(model, trainer.optimizer)

    pgn_path = args.pgn_file
    if not os.path.exists(pgn_path):
        log.error(f"PGN file not found: {pgn_path}")
        sys.exit(1)

    epochs = getattr(args, "epochs", 1)
    watch  = getattr(args, "watch", False)
    trainer.train_pgn(pgn_path, epochs=epochs, watch=watch)
    save_checkpoint(model, trainer.optimizer, trainer.step, tag="pgn_pretrained")
    log.info("PGN training complete. Checkpoint saved.")


# ─────────────────────────────────────────────────
# Full Training Loop
# ─────────────────────────────────────────────────

def cmd_loop(args):
    from selfplay_worker import collect_selfplay_single

    device = cfg.device

    # Current model
    model = ChessNet().to(device)
    trainer = Trainer(model, device)
    start_step = load_checkpoint(model, trainer.optimizer)
    trainer.step = start_step

    # Best model (clone)
    best_model = ChessNet().to(device)
    if not load_best_model(best_model):
        best_model.load_state_dict(copy.deepcopy(model.state_dict()))
        save_best_model(best_model)

    replay_buffer = ReplayBuffer(cfg.buffer_size)
    elo_tracker = EloTracker(cfg.initial_elo)

    # Graceful shutdown
    shutdown_requested = False
    def handle_sigint(sig, frame):
        nonlocal shutdown_requested
        log.info("Shutdown requested (Ctrl+C). Finishing current iteration...")
        shutdown_requested = True
    signal.signal(signal.SIGINT, handle_sigint)

    log.info(f"Starting training loop from step {trainer.step}")

    iteration = 0
    while not shutdown_requested:
        iteration += 1
        log.info(f"\n{'='*50}")
        log.info(f"Iteration {iteration}")

        # --- Self-play data collection ---
        log.info(f"Collecting {cfg.games_per_iteration} self-play games...")
        try:
            samples = collect_selfplay_single(model, cfg.games_per_iteration, device)
            replay_buffer.add_many(samples)
            log.info(f"Buffer: {len(replay_buffer)} samples")
        except Exception as e:
            log.warning(f"Self-play error: {e}")

        # --- Training ---
        if replay_buffer.is_ready(cfg.min_buffer_size):
            pl, vl = trainer.train_n_steps(replay_buffer, cfg.train_steps_per_iter)
            log.info(f"Step {trainer.step} | PL={pl:.4f} | VL={vl:.4f}")
        else:
            log.info(f"Buffer not ready ({len(replay_buffer)}/{cfg.min_buffer_size}), skipping train")

        # --- Checkpoint ---
        if trainer.step > 0 and trainer.step % cfg.checkpoint_interval == 0:
            save_checkpoint(model, trainer.optimizer, trainer.step)

        # --- Elo evaluation ---
        if trainer.step > 0 and trainer.step % cfg.eval_interval == 0:
            log.info("Evaluating current vs best model...")
            try:
                win_rate, should_replace = evaluate_models(
                    model, best_model, elo_tracker, device)
                if should_replace:
                    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    save_best_model(best_model)
                    log.info(f"New best model! Win rate: {win_rate:.2f}")
            except Exception as e:
                log.warning(f"Evaluation error: {e}")

        if shutdown_requested:
            break

    # Final save
    log.info("Saving final checkpoint...")
    save_checkpoint(model, trainer.optimizer, trainer.step, tag="final")
    log.info("Training loop finished.")


# ─────────────────────────────────────────────────
# CLI Setup
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chess AI - AlphaZero-style training system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command")

    # data
    p_data = subparsers.add_parser("data", help="Kaydedilmiş self-play verilerini listele")
    p_data.set_defaults(func=cmd_data)

    # play
    p_play = subparsers.add_parser("play", help="Human vs AI")
    p_play.set_defaults(func=cmd_play)

    # selfplay
    p_sp = subparsers.add_parser("selfplay", help="Run self-play data collection")
    p_sp.add_argument("--games", type=int, default=cfg.games_per_iteration,
                      help="Number of games to play")
    p_sp.add_argument("--watch", action="store_true",
                      help="Tahtayi canli olarak terminalde takip et")
    p_sp.add_argument("--fast", action="store_true",
                      help="MCTS yerine policy-only kullan (cok daha hizli)")

    p_sp.set_defaults(func=cmd_selfplay)

    # train
    p_train = subparsers.add_parser("train", help="Train from self-play data")
    p_train.add_argument("--steps", type=int, default=cfg.train_steps_per_iter,
                         help="Training steps")
    p_train.set_defaults(func=cmd_train)

    # loop
    p_loop = subparsers.add_parser("loop", help="Full training loop (self-play + train + eval)")
    p_loop.set_defaults(func=cmd_loop)

    # train_pgn
    p_pgn = subparsers.add_parser("train_pgn", help="Supervised PGN pretraining")
    p_pgn.add_argument("pgn_file", help="Path to PGN file")
    p_pgn.add_argument("--epochs", type=int, default=1, help="Training epochs")
    p_pgn.add_argument("--watch", action="store_true",
                       help="Her hamleyi tahtada canlı göster")
    p_pgn.set_defaults(func=cmd_train_pgn)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
