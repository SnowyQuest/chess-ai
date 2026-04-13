import os
import time
import pickle
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
    Lightweight in-memory PGN dataset.

    Stores only (fen, move_uci) string pairs in RAM — NOT encoded tensors.
    encode_board() is called lazily inside __getitem__, executed by DataLoader
    worker processes in parallel.

    RAM usage: ~200 bytes/sample instead of 6656 bytes/sample.
    Example: 89 MB PGN ~ 500K samples -> ~100 MB RAM (vs ~3.3 GB if pre-encoded).

    Cache: scan results are saved to <pgn_path>.cache.pkl so that the second
    run skips scanning entirely and loads in seconds.
    """

    def __init__(self, pgn_path: str):
        import chess.pgn
        from tqdm import tqdm

        cache_path = pgn_path + ".cache.pkl"

        # -- Try loading from cache first --------------------------------------
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            pgn_mtime   = os.path.getmtime(pgn_path)
            if cache_mtime >= pgn_mtime:
                log.info(f"Loading PGNDataset from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    self.samples = pickle.load(f)
                log.info(f"PGNDataset ready: {len(self):,} samples (from cache)")
                return
            else:
                log.info("Cache is outdated — rescanning PGN file")

        # -- Scan PGN with progress bar ----------------------------------------
        file_size = os.path.getsize(pgn_path)
        log.info(f"Scanning PGNDataset: {pgn_path} "
                 f"({file_size / 1024 / 1024:.0f} MB)")

        self.samples = []   # list of (fen: str, move_uci: str)

        with open(pgn_path, "r", errors="replace") as f:
            pbar = tqdm(
                total=file_size,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc="Scanning PGN",
                colour="yellow",
                dynamic_ncols=True,
            )
            last_pos = 0

            while True:
                game = chess.pgn.read_game(f)

                cur_pos = f.tell()
                pbar.update(cur_pos - last_pos)
                last_pos = cur_pos

                if game is None:
                    break

                board = game.board()
                for move in game.mainline_moves():
                    self.samples.append((board.fen(), move.uci()))
                    board.push(move)

            pbar.close()

        ram_mb = len(self.samples) * 200 / 1024 / 1024
        log.info(f"PGNDataset ready: {len(self):,} samples (~{ram_mb:.0f} MB RAM)")

        # -- Save cache --------------------------------------------------------
        log.info(f"Saving cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(self.samples, f)
        cache_mb = os.path.getsize(cache_path) / 1024 / 1024
        log.info(f"Cache saved ({cache_mb:.0f} MB) — next run will skip scanning")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import chess
        from board import encode_board
        from moves import move_to_id

        fen, move_uci = self.samples[idx]
        board = chess.Board(fen)
        move  = chess.Move.from_uci(move_uci)
        state = encode_board(board)
        return state, np.int64(move_to_id(move))


class StreamingPGNDataset(Dataset):
    """
    Streaming PGN dataset for very large files (> ~1.5 GB).

    First pass collects (file_offset, move_index_in_game) index pairs.
    __getitem__ seeks to the game on disk and re-reads only what it needs.
    Lowest possible RAM usage at the cost of more disk I/O.
    Keep num_workers <= 2 to avoid disk contention.

    Cache: the index list is saved to <pgn_path>.index.pkl so rescanning
    is skipped on subsequent runs.
    """

    def __init__(self, pgn_path: str):
        import chess.pgn
        from tqdm import tqdm

        self.pgn_path  = pgn_path
        cache_path     = pgn_path + ".index.pkl"

        # -- Try loading index from cache -------------------------------------
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            pgn_mtime   = os.path.getmtime(pgn_path)
            if cache_mtime >= pgn_mtime:
                log.info(f"Loading StreamingPGNDataset index from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    self.index = pickle.load(f)
                n_games = len({o for o, _ in self.index})
                log.info(f"StreamingPGNDataset ready: {len(self):,} samples, "
                         f"{n_games:,} games (from cache)")
                return
            else:
                log.info("Index cache is outdated — rescanning PGN file")

        # -- Scan PGN with progress bar ----------------------------------------
        file_size = os.path.getsize(pgn_path)
        log.info(f"Scanning StreamingPGNDataset: {pgn_path} "
                 f"({file_size / 1024 / 1024:.0f} MB)")

        # index: list of (game_byte_offset, move_index_in_game)
        self.index = []

        with open(pgn_path, "r", errors="replace") as f:
            pbar = tqdm(
                total=file_size,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc="Scanning PGN",
                colour="yellow",
                dynamic_ncols=True,
            )
            last_pos = 0

            while True:
                offset = f.tell()
                game   = chess.pgn.read_game(f)

                cur_pos = f.tell()
                pbar.update(cur_pos - last_pos)
                last_pos = cur_pos

                if game is None:
                    break

                n_moves = sum(1 for _ in game.mainline_moves())
                for i in range(n_moves):
                    self.index.append((offset, i))

            pbar.close()

        n_games = len({o for o, _ in self.index})
        log.info(f"StreamingPGNDataset ready: {len(self):,} samples, "
                 f"{n_games:,} games")

        # -- Save index cache --------------------------------------------------
        log.info(f"Saving index cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(self.index, f)
        log.info("Index cache saved — next run will skip scanning")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        import chess
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

        # Fallback — should never be reached
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

        DataLoader with parallel CPU workers encodes boards in the background
        so the trainer (GPU or CPU) is never waiting for data.

        Parameters
        ----------
        pgn_path : str
            Path to the PGN file.
        epochs : int
            Number of full passes over the dataset.
        watch : bool
            Replay games move-by-move in the terminal after training.
            Forces num_workers=0 during training (tty output from worker
            processes is unreliable).
        num_workers : int
            DataLoader worker processes for parallel board encoding.
            Auto-set to 0 on CPU-only runs and capped at 2 in streaming mode.
            Set to 0 manually for debugging or Windows compatibility.
        streaming_threshold_mb : int
            Files larger than this use StreamingPGNDataset (disk-based,
            very low RAM). Smaller files use PGNDataset (FEN strings in RAM).
        """
        from tqdm import tqdm

        on_gpu  = self.device != "cpu" and torch.cuda.is_available()
        file_mb = os.path.getsize(pgn_path) / 1024 / 1024

        log.info(f"PGN training: {pgn_path} ({file_mb:.0f} MB) | "
                 f"device={self.device} | epochs={epochs}")

        # -- Select dataset ----------------------------------------------------
        if file_mb > streaming_threshold_mb:
            log.info("Large file detected — using StreamingPGNDataset (low RAM mode)")
            dataset           = StreamingPGNDataset(pgn_path)
            effective_workers = min(num_workers, 2)   # disk I/O bound
        else:
            dataset           = PGNDataset(pgn_path)
            effective_workers = num_workers

        # watch mode requires stdout from the main process
        if watch:
            effective_workers = 0

        # Worker processes bring no benefit on CPU-only training
        if not on_gpu:
            effective_workers = 0

        log.info(f"DataLoader: batch_size={cfg.batch_size} | "
                 f"workers={effective_workers} | pin_memory={on_gpu}")

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=effective_workers,
            pin_memory=on_gpu,                               # zero-copy CPU->GPU transfer
            prefetch_factor=4 if effective_workers > 0 else None,
            persistent_workers=(effective_workers > 0),
        )

        # -- Training loop -----------------------------------------------------
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

            for states_batch, targets_batch in pbar:
                # Move tensors to device (non_blocking for async GPU transfer)
                if isinstance(states_batch, np.ndarray):
                    states_t  = torch.from_numpy(states_batch).to(
                        self.device, non_blocking=on_gpu)
                    targets_t = torch.from_numpy(targets_batch).to(
                        self.device, non_blocking=on_gpu)
                else:
                    states_t  = states_batch.to(self.device, non_blocking=on_gpu)
                    targets_t = targets_batch.to(self.device, non_blocking=on_gpu)

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
                    "loss":  f"{recent_loss:.4f}",
                    "pos/s": f"{spd:.0f}",
                    "step":  self.step,
                }, refresh=False)

            pbar.close()

            elapsed = time.time() - t_start
            log.info(
                f"Epoch {epoch + 1}/{epochs} done — "
                f"{total_moves:,} positions | "
                f"avg loss: {recent_loss:.4f} | "
                f"time: {elapsed:.0f}s | "
                f"{total_moves / elapsed:.0f} pos/s"
            )

        # -- Watch mode: terminal board replay after training ------------------
        if watch:
            self._watch_pgn(pgn_path)

    # ------------------------------------------------------------------
    # Terminal board display (watch mode)
    # ------------------------------------------------------------------

    def _watch_pgn(self, pgn_path: str):
        """
        Replay PGN games move-by-move in the terminal.
        Called automatically after train_pgn(..., watch=True).
        This is display-only — no training happens here.
        """
        import chess
        import chess.pgn
        from board import display_board, clear_screen

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
