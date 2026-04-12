import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from moves import MOVE_SPACE


class ResBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        nf = cfg.num_filters

        # Input tower
        self.input_conv = nn.Conv2d(12, nf, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(nf)

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(nf) for _ in range(cfg.num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(nf, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, MOVE_SPACE)

        # Value head
        self.value_conv = nn.Conv2d(nf, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 12, 8, 8)
        Returns:
            policy_logits: (batch, 4096)
            value: (batch, 1) in [-1, 1]
        """
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

    def predict(self, board_tensor: 'np.ndarray', device: str = "cpu"):
        """
        Run inference on a single board state.
        board_tensor: np.ndarray (12, 8, 8)
        Returns:
            policy_logits: np.ndarray (4096,)
            value: float
        """
        import numpy as np
        self.eval()
        with torch.no_grad():
            x = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)
            pol, val = self.forward(x)
            return pol.squeeze(0).cpu().numpy(), val.item()
