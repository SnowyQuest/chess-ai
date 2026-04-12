import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional

from config import cfg
from replay_buffer import ReplayBuffer
from logger import get_logger

log = get_logger("trainer")


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

    # ──────────────────────────────────────────────────────────────
    # PGN supervised pretraining
    # ──────────────────────────────────────────────────────────────

    def train_pgn(self, pgn_path: str, epochs: int = 1, watch: bool = False):
        """
        Supervised pretraining on PGN file (policy head).
        - tqdm progress bar (dosya boyutuna göre gerçek %)
        - watch=True → her hamlede tahtayı terminalde göster
        """
        import chess
        import chess.pgn
        from tqdm import tqdm
        from board import encode_board, display_board, clear_screen
        from moves import move_to_id

        file_size = os.path.getsize(pgn_path)
        log.info(f"PGN training: {pgn_path}  ({file_size/1024/1024:.1f} MB)")

        UNICODE_PIECES = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
        }

        for epoch in range(epochs):
            batch_states  = []
            batch_targets = []
            total_games   = 0
            total_moves   = 0
            recent_loss   = 0.0
            loss_window   = []   # son 50 batch'in loss'u
            t_start       = time.time()

            # ── progress bar: dosya byte'larına göre ──
            pbar = tqdm(
                total=file_size,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"Epoch {epoch+1}/{epochs}",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                colour="cyan",
                dynamic_ncols=True,
            )

            last_pos = 0

            def flush_batch() -> float:
                if not batch_states:
                    return 0.0
                states_t  = torch.tensor(
                    np.array(batch_states, dtype=np.float32), dtype=torch.float32
                ).to(self.device)
                targets_t = torch.tensor(batch_targets, dtype=torch.long).to(self.device)

                self.model.train()
                self.optimizer.zero_grad()
                policy_logits, _ = self.model(states_t)
                loss = F.cross_entropy(policy_logits, targets_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.step += 1
                batch_states.clear()
                batch_targets.clear()
                return loss.item()

            with open(pgn_path, "r", errors="replace") as f:
                while True:
                    game = chess.pgn.read_game(f)

                    # Progress bar güncelle
                    cur_pos = f.tell()
                    pbar.update(cur_pos - last_pos)
                    last_pos = cur_pos

                    if game is None:
                        break

                    board = game.board()
                    game_moves = list(game.mainline_moves())
                    if not game_moves:
                        total_games += 1
                        continue

                    # ── Watch modu: oyun başlığını göster ──
                    if watch:
                        white = game.headers.get("White", "?")
                        black = game.headers.get("Black", "?")
                        result = game.headers.get("Result", "?")
                        elo_w  = game.headers.get("WhiteElo", "?")
                        elo_b  = game.headers.get("BlackElo", "?")
                        pbar.clear()
                        clear_screen()
                        print(f"  ╔══════════════════════════════════════╗")
                        print(f"  ║  PGN Eğitimi — Oyun #{total_games+1:<6}          ║")
                        print(f"  ║  ♙ {white:<16} ({elo_w:>4})        ║")
                        print(f"  ║  ♟ {black:<16} ({elo_b:>4})        ║")
                        print(f"  ║  Sonuç: {result:<6}  Hamle: {len(game_moves):<4}         ║")
                        print(f"  ╚══════════════════════════════════════╝")
                        print()

                    for move_num, move in enumerate(game_moves):
                        state = encode_board(board)
                        batch_states.append(state)
                        batch_targets.append(move_to_id(move))
                        board.push(move)
                        total_moves += 1

                        # ── Watch modu: her hamleyi göster ──
                        if watch:
                            side = "♙ Beyaz" if board.turn == chess.BLACK else "♟ Siyah"
                            print(f"\r  Hamle {move_num+1}/{len(game_moves)}: {side} → {move.uci()}  ", end="", flush=True)
                            print()
                            print(display_board(board, last_move=move))
                            print()
                            elapsed = time.time() - t_start
                            spd = total_moves / elapsed if elapsed > 0 else 0
                            recent = f"{recent_loss:.4f}" if recent_loss > 0 else "—"
                            print(f"  Step: {self.step:,}  |  Toplam hamle: {total_moves:,}  |  "
                                  f"Loss: {recent}  |  Hız: {spd:.0f} hamle/s")
                            time.sleep(0.08)

                        # Batch dolunca eğit
                        if len(batch_states) >= cfg.batch_size:
                            loss_val = flush_batch()
                            loss_window.append(loss_val)
                            if len(loss_window) > 50:
                                loss_window.pop(0)
                            recent_loss = sum(loss_window) / len(loss_window)

                            # Progress bar suffix
                            elapsed = time.time() - t_start
                            spd = total_moves / elapsed if elapsed > 0 else 0
                            pbar.set_postfix({
                                "oyun": f"{total_games:,}",
                                "hamle": f"{total_moves:,}",
                                "loss": f"{recent_loss:.4f}",
                                "h/s": f"{spd:.0f}",
                                "step": self.step,
                            }, refresh=False)

                    total_games += 1

            # Kalan batch
            if batch_states:
                loss_val = flush_batch()
                loss_window.append(loss_val)
                recent_loss = sum(loss_window) / len(loss_window)

            pbar.update(file_size - last_pos)  # kalan byte'ları kapat
            pbar.set_postfix({
                "oyun": f"{total_games:,}",
                "hamle": f"{total_moves:,}",
                "loss": f"{recent_loss:.4f}",
                "step": self.step,
            })
            pbar.close()

            elapsed = time.time() - t_start
            log.info(
                f"Epoch {epoch+1}/{epochs} tamamlandı — "
                f"{total_games:,} oyun | {total_moves:,} hamle | "
                f"avg loss: {recent_loss:.4f} | "
                f"süre: {elapsed:.0f}s"
            )
