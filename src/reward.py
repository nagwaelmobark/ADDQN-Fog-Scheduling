"""
Reward function used by the ADDQN fog-scheduling environment.
"""

import numpy as np


class RewardCalculator:
    """Compute the shaped multi-objective reward."""

    def __init__(
        self,
        w_response_time=1.0,
        w_deadline=15.0,
        w_overload=5.0,
        w_energy=0.5,
        w_load_balance=3.0,
        overload_threshold_ms=1500.0,
        queue_cap_ms=2000.0,
    ):
        self.w_response_time = w_response_time
        self.w_deadline = w_deadline
        self.w_overload = w_overload
        self.w_energy = w_energy
        self.w_load_balance = w_load_balance

        self.overload_threshold_ms = overload_threshold_ms
        self.queue_cap_ms = queue_cap_ms

    def calculate(
        self,
        response_time,
        exec_time,
        deadline,
        queue_before,
        queue_after,
        energy_level,
        all_queues,
    ):
        # Deadline satisfaction
        deadline_margin = deadline - response_time

        if deadline_margin >= 0:
            deadline_reward = self.w_deadline * (
                1.0 + min(deadline_margin / deadline, 0.5)
            )
        else:
            miss_ratio = abs(deadline_margin) / deadline
            deadline_reward = -self.w_deadline * (1.0 + miss_ratio)

        # Response-time penalty
        rt_normalized = min(response_time / 500.0, 1.0)
        rt_reward = -self.w_response_time * rt_normalized

        # Overload penalty
        if queue_after >= self.overload_threshold_ms:
            overload_penalty = -self.w_overload * (
                queue_after / self.queue_cap_ms
            )
        else:
            overload_penalty = 0.0

        # Load-balancing terms
        queue_ratio_before = queue_before / self.queue_cap_ms
        load_penalty = -self.w_load_balance * (queue_ratio_before ** 2)

        queue_std = np.std(all_queues) / self.queue_cap_ms
        balance_bonus = (
            self.w_load_balance * 0.5 * (1.0 - queue_std)
        )

        # Remaining-energy bonus
        energy_bonus = self.w_energy * energy_level

        reward = (
            deadline_reward
            + rt_reward
            + overload_penalty
            + load_penalty
            + balance_bonus
            + energy_bonus
        )

        # Keep the reward within the range used during training
        reward = np.clip(reward, -50.0, 20.0)

        return float(reward)

    def get_info(self, reward, response_time, deadline):
        """Return a small reward summary for logging."""
        return {
            "reward": reward,
            "deadline_met": response_time <= deadline,
            "deadline_margin": deadline - response_time,
        }