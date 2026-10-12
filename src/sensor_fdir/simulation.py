from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimulationConfig


def _first_order(command: np.ndarray, fs_hz: float, tau_s: float, x0: float = 0.0) -> np.ndarray:
    x = np.empty_like(command, dtype=float)
    x[0] = x0
    dt = 1.0 / fs_hz
    alpha = dt / tau_s
    for k in range(1, len(command)):
        x[k] = x[k - 1] + alpha * (command[k - 1] - x[k - 1])
    return x


def generate_baseline(cfg: SimulationConfig) -> pd.DataFrame:
    """Generate physically related healthy sensor data for a motor/rotor/cooling rig."""
    if cfg.fs_hz <= 0 or cfg.duration_s <= 0:
        raise ValueError("Sampling frequency and duration must be positive.")

    rng = np.random.default_rng(cfg.seed)
    fs = cfg.fs_hz
    t = np.arange(0.0, cfg.duration_s, 1.0 / fs)
    n = len(t)

    # Non-stationary speed command: ramp, multiple operating plateaus and load events.
    rpm_cmd = np.select(
        [t < 12, t < 70, t < 110, t < 170, t <= cfg.duration_s],
        [
            250.0 * t,
            3000.0 + 90.0 * np.sin(2 * np.pi * 0.035 * t),
            2700.0 + 130.0 * np.sin(2 * np.pi * 0.025 * t),
            3200.0 + 75.0 * np.sin(2 * np.pi * 0.045 * t),
            2850.0 + 110.0 * np.sin(2 * np.pi * 0.030 * t),
        ],
    )
    rpm_true = _first_order(rpm_cmd, fs, tau_s=0.65, x0=0.0)
    omega_true = rpm_true * 2 * np.pi / 60.0

    load = 0.38 + 0.10 * np.sin(2 * np.pi * 0.018 * t)
    load += np.where((t >= 80) & (t < 105), 0.30, 0.0)
    load += np.where((t >= 180) & (t < 205), 0.22, 0.0)

    ambient = 24.0
    thermal_target = ambient + 20.0 * load + 0.0010 * rpm_true
    temperature_true = _first_order(thermal_target, fs, tau_s=30.0, x0=ambient)

    pressure_true = 1.85 + 0.00030 * rpm_true + 0.24 * load + 0.035 * np.sin(2 * np.pi * 7 * t)
    voltage_true = 48.5 - 2.2 * load - 0.00016 * rpm_true + 0.16 * np.sin(2 * np.pi * 100 * t)

    rot_hz = rpm_true / 60.0
    phase = np.cumsum(2 * np.pi * rot_hz / fs)
    accel_x_true = (
        0.16 * np.sin(phase)
        + 0.055 * np.sin(2 * phase + 0.4)
        + 0.030 * np.sin(2 * np.pi * 110 * t)
    )
    accel_y_true = (
        0.11 * np.cos(phase + 0.2)
        + 0.045 * np.sin(3 * phase)
        + 0.022 * np.sin(2 * np.pi * 72 * t)
    )

    data = pd.DataFrame(
        {
            "time_s": t,
            "rpm_cmd": rpm_cmd,
            "load": load,
            "rpm_model": rpm_true,
            "temperature_model_C": temperature_true,
            "pressure_model_bar": pressure_true,
            "voltage_model_V": voltage_true,
            "rpm": rpm_true + rng.normal(0.0, 3.0, n),
            "gyro_z_rad_s": omega_true + rng.normal(0.0, 0.38, n),
            "temperature_C": temperature_true + rng.normal(0.0, 0.10, n),
            "pressure_bar": pressure_true + rng.normal(0.0, 0.010, n),
            "voltage_V": voltage_true + rng.normal(0.0, 0.040, n),
            "accel_x_g": accel_x_true + rng.normal(0.0, 0.025, n),
            "accel_y_g": accel_y_true + rng.normal(0.0, 0.025, n),
        }
    )
    return data
