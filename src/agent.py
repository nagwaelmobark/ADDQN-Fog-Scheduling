"""
Double DQN agent used with the ADDQN architecture.
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .model import ADDQN
    from .replay_buffer import ReplayBuffer
except ImportError:
    from model import ADDQN
    from replay_buffer import ReplayBuffer


class ADDQNAgent:
    """Double DQN learning agent for fog scheduling."""

    def __init__(
        self,
        state_dim,
        action_dim,
        num_nodes=15,
        lr=3e-4,
        gamma=0.99,
        device=None,
        hidden_dim=128,
        use_lr_scheduler=True,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.num_nodes = num_nodes

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Online network
        self.policy_net = ADDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Target network
        self.target_net = ADDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )

        self.use_lr_scheduler = use_lr_scheduler

        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=100,
                gamma=0.95,
            )

        self.memory = ReplayBuffer(
            capacity=100_000,
            device=self.device,
        )

        self.loss_fn = nn.MSELoss()

        self.training_steps = 0
        self.losses = []

    def select_action(self, state, epsilon):
        """Select an action using epsilon-greedy exploration."""

        if random.random() < epsilon:
            return random.randint(
                0,
                self.action_dim - 1,
            )

        self.policy_net.eval()

        with torch.no_grad():
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            )

            q_values = self.policy_net(
                state_tensor
            )

            return int(
                q_values.argmax(dim=1).item()
            )

    def train(self, batch_size=64):
        """Perform one Double DQN optimization step."""

        if len(self.memory) < batch_size:
            return None

        self.policy_net.train()

        state, action, reward, next_state = (
            self.memory.sample(batch_size)
        )

        current_q = (
            self.policy_net(state)
            .gather(
                1,
                action.unsqueeze(1),
            )
            .squeeze(1)
        )

        # Double DQN target:
        # online network selects the next action,
        # target network evaluates it.
        with torch.no_grad():
            next_actions = (
                self.policy_net(next_state)
                .argmax(
                    dim=1,
                    keepdim=True,
                )
            )

            next_q = (
                self.target_net(next_state)
                .gather(
                    1,
                    next_actions,
                )
                .squeeze(1)
            )

            target = (
                reward
                + self.gamma * next_q
            )

        loss = self.loss_fn(
            current_q,
            target,
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.training_steps += 1
        self.losses.append(
            float(loss.item())
        )

        return float(
            loss.item()
        )

    def update_target(self):
        """Copy online-network weights to the target network."""

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def step_scheduler(self):
        if self.use_lr_scheduler:
            self.scheduler.step()

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_training_stats(self):
        if len(self.losses) == 0:
            return {
                "avg_loss": 0.0,
                "training_steps": 0,
            }

        return {
            "avg_loss": (
                sum(self.losses[-100:])
                / min(
                    100,
                    len(self.losses),
                )
            ),
            "training_steps": self.training_steps,
        }