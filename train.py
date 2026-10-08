"""
Training script for the 15-node ADDQN fog-scheduling experiment.
"""

import csv
import random

import numpy as np
import torch

from src.environment import FogSchedulingEnv
from src.agent import ADDQNAgent


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------------------------------------
# Environment and agent
# -------------------------------------------------

env = FogSchedulingEnv(
    num_nodes=15,
    max_steps=200,
    seed=SEED,
)

state_dim = len(env.reset())
action_dim = env.num_nodes

agent = ADDQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    num_nodes=env.num_nodes,
    lr=3e-4,
    gamma=0.99,
    hidden_dim=128,
    use_lr_scheduler=True,
)

print(
    f"Environment: {env.num_nodes} nodes | "
    f"state_dim={state_dim} | "
    f"action_dim={action_dim}"
)

print(
    "ADDQN parameters:",
    f"{sum(p.numel() for p in agent.policy_net.parameters()):,}",
)


# -------------------------------------------------
# Training configuration
# -------------------------------------------------

EPISODES = 500
BATCH_SIZE = 64

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

TARGET_UPDATE_FREQ = 5

csv_path = "training_metrics.csv"


# -------------------------------------------------
# CSV header
# -------------------------------------------------

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(
        [
            "episode",
            "total_reward",
            "avg_rt",
            "dmr",
            "deadline_met_ratio",
            "overload_ratio",
            "epsilon",
            "learning_rate",
            "avg_cpu",
            "avg_queue_ms",
            "avg_energy",
            "chosen_latency",
            "avg_loss",
        ]
    )


# -------------------------------------------------
# Training
# -------------------------------------------------

print(f"\nStarting ADDQN training for {EPISODES} episodes...\n")


for episode in range(EPISODES):

    state = env.reset()
    total_reward = 0.0

    response_times = []
    deadline_misses = []
    overloads = []

    avg_cpus = []
    avg_queues = []
    avg_energies = []
    chosen_latencies = []

    for _ in range(env.max_steps):

        action = agent.select_action(
            state,
            epsilon,
        )

        next_state, reward, done, info = env.step(
            action
        )

        agent.memory.add(
            state,
            action,
            reward,
            next_state,
        )

        agent.train(
            batch_size=BATCH_SIZE
        )

        state = next_state
        total_reward += reward

        response_times.append(
            info["response_time"]
        )

        deadline_misses.append(
            info["deadline_miss"]
        )

        overloads.append(
            info["overload"]
        )

        avg_cpus.append(
            info["avg_cpu"]
        )

        avg_queues.append(
            info["avg_queue_ms"]
        )

        avg_energies.append(
            info["avg_energy"]
        )

        chosen_latencies.append(
            info["latency"]
        )

        if done:
            break

    # Epsilon decay
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay,
    )

    # Target-network synchronization
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target()

    agent.step_scheduler()

    deadline_met_ratio = (
        1.0 - np.mean(deadline_misses)
    )

    overload_ratio = np.mean(
        overloads
    )

    # Log every 10 episodes
    if episode % 10 == 0:

        stats = agent.get_training_stats()

        with open(
            csv_path,
            "a",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    episode,
                    float(total_reward),
                    float(np.mean(response_times)),
                    float(np.mean(deadline_misses)),
                    float(deadline_met_ratio),
                    float(overload_ratio),
                    float(epsilon),
                    float(agent.get_current_lr()),
                    float(np.mean(avg_cpus)),
                    float(np.mean(avg_queues)),
                    float(np.mean(avg_energies)),
                    float(np.mean(chosen_latencies)),
                    float(stats["avg_loss"]),
                ]
            )

    # Console progress
    if episode % 50 == 0:

        print(
            f"Episode {episode:3d} | "
            f"Reward: {total_reward:7.1f} | "
            f"RT: {np.mean(response_times):6.1f} ms | "
            f"DMR: {np.mean(deadline_misses):.3f} | "
            f"Epsilon: {epsilon:.3f}"
        )


print("\nTraining complete.")
print(f"Metrics saved to: {csv_path}")