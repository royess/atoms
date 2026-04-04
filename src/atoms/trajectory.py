from __future__ import annotations

import numpy as np

from atoms.config import TrajectoryConfig


def build_time_grid(duration_s: float, time_step_s: float) -> np.ndarray:
    steps = int(np.round(duration_s / time_step_s))
    if steps < 1:
        raise ValueError("duration_s must be at least one time step")
    return np.linspace(0.0, duration_s, steps + 1)


def minimum_jerk_trajectory(config: TrajectoryConfig, time_grid_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = np.clip(time_grid_s / config.duration_s, 0.0, 1.0)
    x = config.distance_m * (10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)
    v = (config.distance_m / config.duration_s) * (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4)
    a = (config.distance_m / (config.duration_s**2)) * (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3)
    return x, v, a
