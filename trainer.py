import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from config import cfg
from replay_buffer import ReplayBuffer
from logger import get_logger

log = get_logger("trainer")


# ---------------------------------------------------------------------------
# PGN Datasets
# ---------------------------------------------------------------------------

class PGNDataset(Dataset):
    """
    In-memory PGN dataset for small/medium files (< ~1.5 GB).
    Loads all (state, move_id) pairs upfront into numpy arrays.
    DataLoader workers slice with zero encoding overhead.
    """

    def __init__(self, pgn_path: str):
        import chess.pgn
        from board import encode_board
        from moves import move_to_id

        log.info(f"Loading PGNDataset: {pgn_path}")
        states, move_ids = [], []

        with open(pgn_path, "r", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    states.append(encode_board(board))
                    move_ids.append(move_to_id(move))
                    board.push(move)

        self.states   = np.array(states,   dtype=np.float32)  # (N, 26, 8, 8)
        self.move_ids = np.array(move_ids, dtype=np.int64)    # (N,)
        log.info(f"PGNDataset ready: {len(self):,} samples")

    def __len__(self):
        return len(self.move_ids)

    def __getitem__(self, idx):
        return self.states[idx], self.move_ids[idx]


class StreamingPGNDataset(Dataset):
    """
    Streaming PGN dataset for large files (> ~1.5 GB).
    First pass collects (file_offset, move_index) pairs.
    __getitem__ re-reads only the required game on demand — low RAM usage.
    Keep num_workers <= 2 to avoid disk I/O contention.
    """

    def __init__(self, pgn_path: str):
        import chess.pgn

        self.pgn_path = pgn_path
        log.info(f"Scanning StreamingPGNDataset: {pgn_path}")

        # index: list of (game_offset, move_index_in_game)
        self.index: list = []

        with open(pgn_path, "r", errors="replace") as f:
            while True:
                offset = f.tell()
                game   = chess.pgn.read_game(f)
                if game is None:
                    break
                n_moves = sum(1 for _ in game.mainline_moves())
                for i in range(n_moves):
                    self.index.append((offset, i))

        n_games = len({o for o, _ in self.index})
        log.info(f"StreamingPGNDataset ready: {len(self):,} samples, "
                 f"{n_games:,} games")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        import chess.pgn
        from board import encode_board
        from moves import move_to_id

        offset, move_idx = self.index[idx]
        with open(self.pgn_path, "r", errors="replace") as f:
            f.seek(offset)
            game = chess.pgn.read_game(f)

        board = game.board()
        for i, move in enumerate(game.mainline_moves()):
            if i == move_idx:
                return encode_board(board), np.int64(move_to_id(move))
            board.push(move)

        # Fallback (should never be reached)
        return np.zeros((26, 8, 8), dtype=np.float32), np.int64(0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        self.step = 0

    # ------------------------------------------------------------------
    # Self-play training
    # ------------------------------------------------------------------

    def train_step(self, replay_buffer: ReplayBuffer) -> Tuple[float, float]:
        if not replay_buffer.is_ready(cfg.batch_size):
            return 0.0, 0.0

        states, policies, values = replay_buffer.sample(cfg.batch_size)

        states_t   = torch.tensor(states,   dtype=torch.float32).to(self.device)
        policies_t = torch.tensor(policies, dtype=torch.float32).to(self.device)
        values_t   = torch.tensor(values,   dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        policy_logits, value_pred = self.model(states_t)

        log_probs   = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies_t * log_probs).sum(dim=1).mean()
        value_loss  = F.mse_loss(value_pred, values_t)

        loss = cfg.policy_loss_weight * policy_loss + cfg.value_loss_weight * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1
        return policy_loss.item(), value_loss.item()

    def train_n_steps(self, replay_buffer: ReplayBuffer, n: int) -> Tuple[float, float]:
        pl_total, vl_total = 0.0, 0.0
        for _ in range(n):
            pl, vl = self.train_step(replay_buffer)
            pl_total += pl
            vl_total += vl
        return pl_total / n, vl_total / n

    # ------------------------------------------------------------------
    # PGN supervised pretraining
    # ------------------------------------------------------------------

    def train_pgn(self, pgn_path: str, epochs: int = 1,
                  watch: bool = False, num_workers: int = 4,
                  streaming_threshold_mb: int = 1500):
        """
        Supervised pretraining from a PGN file (policy head only).

        Uses torch.utils.data.DataLoader with parallel CPU workers so that
        board encoding runs in the background and the GPU/CPU trainer is
        never starved waiting for data.

        Parameters
        ----------
        pgn_path : str
            Path to the PGN file.
        epochs : int
            Number of full passes over the dataset.
        watch : bool
            Print a live board to the terminal each move (forces num_workers=0
            because tty output from worker processes is unreliable).
        num_workers : int
            DataLoader worker processes for parallel encoding.
            Automatically set to 0 on CPU-only runs (no benefit) and capped
            at 2 for streaming mode (disk I/O bound).
            Set to 0 manually for debugging or Windows compatibility.
        streaming_threshold_mb : int
            Files larger than this (MB) use StreamingPGNDataset (low RAM).
            Files smaller are loaded fully into RAM for faster access.
        """
        from tqdm import tqdm

        on_gpu     = self.device != "cpu" and torch.cuda.is_available()
        file_mb    = os.path.getsize(pgn_path) / 1024 / 1024

        log.info(f"PGN training: {pgn_path} ({file_mb:.0f} MB) | "
                 f"device={self.device} | epochs={epochs}")

        # ── Select dataset ──────────────────────────────────────────
        if file_mb > streaming_threshold_mb:
            log.info("Large file — using StreamingPGNDataset (low RAM mode)")
            dataset = StreamingPGNDataset(pgn_path)
            # Disk I/O is the bottleneck; more workers don't help much
            effective_workers = min(num_workers, 2)
        else:
            dataset = PGNDataset(pgn_path)
            effective_workers = num_workers

        # watch mode needs stdout from the main process
        if watch:
            effective_workers = 0

        # No benefit from workers on CPU-only training
        if not on_gpu:
            effective_workers = 0

        log.info(f"DataLoader: batch={cfg.batch_size} | "
                 f"workers={effective_workers} | "
                 f"pin_memory={on_gpu}")

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=effective_workers,
            pin_memory=on_gpu,           # zero-copy CPU→GPU transfer
            prefetch_factor=4 if effective_workers > 0 else None,
            persistent_workers=(effective_workers > 0),
        )

        # ── Watch-mode helpers ──────────────────────────────────────
        if watch:
            import chess
            import chess.pgn
            from board import display_board, clear_screen

        # ── Training loop ────────────────────────────────────────────
        for epoch in range(epochs):
            t_start     = time.time()
            loss_window = []
            recent_loss = 0.0
            total_moves = 0

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                colour="cyan",
                dynamic_ncols=True,
            )

            for states_np, targets_np in pbar:
                # states_np / targets_np are already tensors when coming
                # from DataLoader; keep as-is and just move to device.
                if isinstance(states_np, np.ndarray):
                    states_t  = torch.from_numpy(states_np).to(
                        self.device, non_blocking=on_gpu)
                    targets_t = torch.from_numpy(targets_np).to(
                        self.device, non_blocking=on_gpu)
                else:
                    states_t  = states_np.to(self.device, non_blocking=on_gpu)
                    targets_t = targets_np.to(self.device, non_blocking=on_gpu)

                self.model.train()
                self.optimizer.zero_grad()
                policy_logits, _ = self.model(states_t)
                loss = F.cross_entropy(policy_logits, targets_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.step   += 1
                total_moves += states_t.size(0)

                loss_window.append(loss.item())
                if len(loss_window) > 50:
                    loss_window.pop(0)
                recent_loss = sum(loss_window) / len(loss_window)

                elapsed = time.time() - t_start
                spd     = total_moves / elapsed if elapsed > 0 else 0

                pbar.set_postfix({
                    "loss": f"{recent_loss:.4f}",
                    "pos/s": f"{spd:.0f}",
                    "step": self.step,
                }, refresh=False)

                # ── Watch mode (workers=0, sequential) ───────────────
                if watch:
                    # Re-read the last batch's first position for display
                    # (approximate; exact board display requires sequential mode)
                    pass   # board display is handled below in sequential path

            pbar.close()

            elapsed = time.time() - t_start
            log.info(
                f"Epoch {epoch + 1}/{epochs} done — "
                f"{total_moves:,} positions | "
                f"avg loss: {recent_loss:.4f} | "
                f"time: {elapsed:.0f}s | "
                f"{total_moves / elapsed:.0f} pos/s"
            )

        # ── Watch mode: sequential replay for terminal display ───────
        # Run a second, display-only pass when watch=True.
        # This is separate so the fast path above is never slowed down.
        # If you only need watch mode, pass watch=True and accept that
        # it runs single-threaded.
        if watch:
            self._watch_pgn(pgn_path)

    # ------------------------------------------------------------------
    # Watch helper (sequential, display only — not used for training)
    # ------------------------------------------------------------------

    def _watch_pgn(self, pgn_path: str):
        """
        Display PGN games move-by-move in the terminal.
        Called automatically when train_pgn(..., watch=True).
        """
        import chess
        import chess.pgn
        from board import encode_board, display_board, clear_screen
        from moves import move_to_id

        log.info("Watch mode: replaying PGN for terminal display...")

        with open(pgn_path, "r", errors="replace") as f:
            game_num = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                game_num += 1
                moves = list(game.mainline_moves())
                if not moves:
                    continue

                white  = game.headers.get("White",    "?")
                black  = game.headers.get("Black",    "?")
                result = game.headers.get("Result",   "?")
                elo_w  = game.headers.get("WhiteElo", "?")
                elo_b  = game.headers.get("BlackElo", "?")

                clear_screen()
                print(f"  Game #{game_num}")
                print(f"  White: {white} ({elo_w})  vs  Black: {black} ({elo_b})")
                print(f"  Result: {result}  |  Moves: {len(moves)}")
                print()

                board = game.board()
                for i, move in enumerate(moves):
                    side = "White" if board.turn == chess.WHITE else "Black"
                    board.push(move)
                    clear_screen()
                    print(f"  Move {i + 1}/{len(moves)}: {side} -> {move.uci()}")
                    print()
                    print(display_board(board, last_move=move))
                    print()
                    time.sleep(0.08)
