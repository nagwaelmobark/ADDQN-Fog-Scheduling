"""
Experience replay buffer used for ADDQN training.
"""

from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:
    """Store and sample transitions for DQN training."""

    def __init__(self, capacity=100_000, seed=42, device=None):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

        random.seed(seed)

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state):
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)

        self.buffer.append(
            (
                state,
                int(action),
                float(reward),
                next_state,
            )
        )

    def sample(self, batch_size):
        batch = random.sample(
            self.buffer,
            batch_size,
        )

        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(
            np.stack(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            np.stack(next_states),
            dtype=torch.float32,
            device=self.device,
        )

        return states, actions, rewards, next_states