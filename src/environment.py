"""
Fog-computing scheduling environment used in the ADDQN experiments.
"""

import numpy as np

try:
    from .reward import RewardCalculator
except ImportError:
    from reward import RewardCalculator


class FogSchedulingEnv:
    """
    Heterogeneous fog-scheduling environment.

    Each fog node is represented by five state features:
    1. Queue-derived utilization proxy
    2. Normalized queue occupancy
    3. Normalized communication latency
    4. Remaining normalized energy
    5. Normalized processing-capacity factor
    """

    def __init__(
        self,
        num_nodes=15,
        max_steps=200,
        seed=42,
        exec_time_min=50.0,
        exec_time_max=200.0,
        tight_deadline_rate=0.10,
        queue_cap_ms=2000.0,
        dt=1.0,
        overload_threshold_ms=1500.0,
    ):
        self.num_nodes = int(num_nodes)
        self.max_steps = int(max_steps)

        self.exec_time_min = float(exec_time_min)
        self.exec_time_max = float(exec_time_max)
        self.tight_deadline_rate = float(tight_deadline_rate)

        self.queue_cap_ms = float(queue_cap_ms)
        self.dt = float(dt)
        self.overload_threshold_ms = float(overload_threshold_ms)

        self.rng = np.random.default_rng(seed)

        # Fixed heterogeneous node characteristics
        self.capacity_factor = self.rng.uniform(
            0.8, 1.2, size=self.num_nodes
        ).astype(np.float32)

        self.base_latency = self.rng.uniform(
            5.0, 50.0, size=self.num_nodes
        ).astype(np.float32)

        self.reward_calc = RewardCalculator(
            w_response_time=1.0,
            w_deadline=15.0,
            w_overload=5.0,
            w_energy=0.5,
            w_load_balance=3.0,
            overload_threshold_ms=self.overload_threshold_ms,
            queue_cap_ms=self.queue_cap_ms,
        )

        self.reset()

    def reset(self):
        """Reset the dynamic state while preserving the node topology."""

        self.step_count = 0

        self.queue_ms = np.zeros(
            self.num_nodes,
            dtype=np.float32,
        )

        self.energy = np.ones(
            self.num_nodes,
            dtype=np.float32,
        )

        # Queue-derived utilization proxy
        self.cpu = np.zeros(
            self.num_nodes,
            dtype=np.float32,
        )

        self.latency = self.base_latency.copy()

        return self._get_state()

    def _get_state(self):
        """Return the flattened 5N-dimensional state representation."""

        cpu_n = np.clip(
            self.cpu,
            0.0,
            1.0,
        )

        q_n = np.clip(
            self.queue_ms / self.queue_cap_ms,
            0.0,
            1.0,
        )

        lat_n = np.clip(
            self.latency / 200.0,
            0.0,
            1.0,
        )

        energy_n = np.clip(
            self.energy,
            0.0,
            1.0,
        )

        capacity_n = np.clip(
            (self.capacity_factor - 0.8) / (1.2 - 0.8),
            0.0,
            1.0,
        )

        state = np.stack(
            [
                cpu_n,
                q_n,
                lat_n,
                energy_n,
                capacity_n,
            ],
            axis=1,
        ).reshape(-1)

        return state.astype(np.float32)

    def _sample_task(self):
        """Generate one execution time and its associated deadline."""

        exec_time = float(
            self.rng.uniform(
                self.exec_time_min,
                self.exec_time_max,
            )
        )

        if self.rng.random() < self.tight_deadline_rate:
            deadline = exec_time * float(
                self.rng.uniform(1.05, 1.20)
            )
        else:
            deadline = exec_time * float(
                self.rng.uniform(1.20, 2.00)
            )

        return exec_time, deadline

    def step(self, action):
        """Apply one scheduling decision and advance the environment."""

        action = int(action)

        if action < 0 or action >= self.num_nodes:
            raise ValueError(
                f"Invalid action {action}. "
                f"Expected an integer in [0, {self.num_nodes - 1}]."
            )

        self.step_count += 1

        # Generate the incoming task
        exec_time, deadline = self._sample_task()

        queue_before = float(
            self.queue_ms[action]
        )

        # Assign the task to the selected node
        self.queue_ms[action] += exec_time

        # Simulate processing during one environment step
        processed = (
            80.0
            * self.capacity_factor
            * self.dt
        )

        self.queue_ms = np.maximum(
            0.0,
            self.queue_ms - processed.astype(np.float32),
        )

        queue_after = float(
            self.queue_ms[action]
        )

        # Update queue-derived utilization
        self.cpu = np.clip(
            self.queue_ms / self.queue_cap_ms,
            0.0,
            1.0,
        ).astype(np.float32)

        # Queue-dependent communication latency
        congestion = (
            self.queue_ms / self.queue_cap_ms
        )

        self.latency = (
            self.base_latency
            * (1.0 + 1.5 * congestion)
        ).astype(np.float32)

        # Update normalized remaining energy
        energy_drop = (
            0.0005
            * (exec_time / 200.0)
            * (0.5 + self.cpu[action])
        )

        self.energy[action] = np.clip(
            self.energy[action] - energy_drop,
            0.0,
            1.0,
        )

        # Scheduling metrics
        waiting_time = queue_before

        response_time = (
            waiting_time
            + float(self.latency[action])
            + exec_time
        )

        overload = float(
            queue_after >= self.overload_threshold_ms
        )

        deadline_miss = float(
            response_time > deadline
        )

        reward = self.reward_calc.calculate(
            response_time=response_time,
            exec_time=exec_time,
            deadline=deadline,
            queue_before=queue_before,
            queue_after=queue_after,
            energy_level=float(self.energy[action]),
            all_queues=self.queue_ms,
        )

        done = (
            self.step_count >= self.max_steps
        )

        info = {
            "exec_time": exec_time,
            "deadline": deadline,
            "waiting_time": waiting_time,
            "latency": float(self.latency[action]),
            "response_time": response_time,
            "overload": overload,
            "deadline_miss": deadline_miss,
            "avg_cpu": float(np.mean(self.cpu)),
            "avg_queue_ms": float(np.mean(self.queue_ms)),
            "avg_energy": float(np.mean(self.energy)),
            "chosen_node": action,
            "reward_raw": reward,
        }

        return (
            self._get_state(),
            float(reward),
            bool(done),
            info,
        )