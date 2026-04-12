import random
import numpy as np
from collections import deque
from typing import List, Tuple


# Each sample: (state: np.ndarray (12,8,8), policy: np.ndarray (4096,), value: float)
Sample = Tuple[np.ndarray, np.ndarray, float]


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer: deque = deque(maxlen=maxlen)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.buffer.append((state, policy, value))

    def add_many(self, samples: List[Sample]):
        for s in samples:
            self.buffer.append(s)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([s[0] for s in batch], dtype=np.float32)
        policies = np.array([s[1] for s in batch], dtype=np.float32)
        values = np.array([s[2] for s in batch], dtype=np.float32)
        return states, policies, values

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self.buffer) >= min_size
