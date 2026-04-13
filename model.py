"""
model.py
========
ChessNet with 26-channel input (upgraded from 20).

Changes from previous version
------------------------------
- INPUT_PLANES: 20 -> 26
- migrate_checkpoint(): handles 12-ch, 20-ch, and 26-ch checkpoints
  (preserves all existing weights, zero-initializes new channels)
- Slightly wider value head (256 units) for better positional evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from moves import MOVE_SPACE

INPUT_PLANES = 26


class ResBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        nf = cfg.num_filters

        self.input_conv = nn.Conv2d(INPUT_PLANES, nf, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(nf)

        self.res_blocks = nn.ModuleList(
            [ResBlock(nf) for _ in range(cfg.num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(nf, 32, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * 8 * 8, MOVE_SPACE)

        # Value head (wider hidden layer for better positional evaluation)
        self.value_conv  = nn.Conv2d(nf, 4, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(4)
        self.value_fc1   = nn.Linear(4 * 8 * 8, 256)
        self.value_fc2   = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value

    def predict(self, board_tensor, device: str = "cpu"):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)
            pol, val = self.forward(x)
            return pol.squeeze(0).cpu().numpy(), val.item()


# ---------------------------------------------------------------------------
# Checkpoint migration
# ---------------------------------------------------------------------------

def migrate_checkpoint(old_path: str, new_path: str):
    """
    Migrate a checkpoint from any previous channel count to INPUT_PLANES (26).

    Strategy
    ---------
    - input_conv.weight[:, :old_ch, :, :] = preserved weights
    - input_conv.weight[:, old_ch:, :, :]  = zero (new channels learned from scratch)
    - value head: if hidden size changed (64->256), reinitialize those layers only
    - All other layers: copied as-is

    Supports upgrading from 12, 20, or any other channel count.
    """
    state    = torch.load(old_path, map_location="cpu")
    old_sd   = state["model_state_dict"]

    # Detect old channel count
    old_ch = old_sd["input_conv.weight"].shape[1]
    print(f"  Source: {old_ch}-channel -> Target: {INPUT_PLANES}-channel")

    if old_ch == INPUT_PLANES:
        print("  Already at target channel count. Nothing to migrate.")
        return

    new_model = ChessNet()
    new_sd    = new_model.state_dict()
    migrated  = {}

    for key, param in old_sd.items():
        if key == "input_conv.weight":
            nf    = param.shape[0]
            new_w = torch.zeros(nf, INPUT_PLANES, 3, 3)
            new_w[:, :old_ch, :, :] = param
            migrated[key] = new_w
            print(f"  input_conv.weight: {list(param.shape)} -> {list(new_w.shape)}")
        elif key in new_sd:
            if new_sd[key].shape == param.shape:
                migrated[key] = param
            else:
                # Shape mismatch (e.g. value head widened): skip, use fresh init
                print(f"  Skipped (shape mismatch): {key} "
                      f"{list(param.shape)} vs {list(new_sd[key].shape)}")
        else:
            print(f"  Skipped (key not in new model): {key}")

    new_model.load_state_dict(migrated, strict=False)
    torch.save({
        "model_state_dict": new_model.state_dict(),
        "optimizer_state_dict": {},   # reset optimizer after arch change
        "step": state.get("step", 0),
    }, new_path)
    print(f"  Saved -> {new_path}")


if __name__ == "__main__":
    import numpy as np
    net = ChessNet()
    params = sum(p.numel() for p in net.parameters())
    print(f"ChessNet parameters: {params:,}")
    dummy  = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    pol, v = net.predict(dummy)
    print(f"Policy: {pol.shape}, Value: {v:.4f}")
