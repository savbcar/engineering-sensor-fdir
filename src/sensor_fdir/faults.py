from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FaultSpec:
    name: str
    sensor: str
    fault_type: str
    start_s: float
    end_s: float
    magnitude: str


DEFAULT_FAULTS: tuple[FaultSpec, ...] = (
    FaultSpec("gyro_bias", "gyro_z_rad_s", "bias", 50.0, 65.0, "+12 rad/s"),
    FaultSpec("temperature_drift", "temperature_C", "drift", 75.0, 95.0, "0 to +6 C"),
    FaultSpec("pressure_stuck", "pressure_bar", "stuck", 105.0, 120.0, "held constant"),
    FaultSpec("voltage_dropout", "voltage_V", "dropout", 125.0, 128.0, "NaN"),
    FaultSpec("vibration_spike", "accel_x_g", "spike", 135.0, 140.0, "+/-1.3 g impulses"),
    FaultSpec("vibration_noise", "accel_y_g", "increased_noise", 150.0, 165.0, "+0.20 g sigma"),
    FaultSpec("rpm_gain", "rpm", "gain_error", 175.0, 190.0, "+6% gain"),
    FaultSpec("vibration_interference", "accel_x_g", "sinusoidal_interference", 200.0, 215.0, "90 Hz, 0.28 g"),
    FaultSpec("bearing_degradation", "accel_y_g", "slow_degradation", 220.0, 240.0, "ramped 1x/2x vibration"),
)


def fault_manifest(faults: Iterable[FaultSpec] = DEFAULT_FAULTS) -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in faults])


def inject_faults(
    baseline: pd.DataFrame,
    fs_hz: float,
    seed: int = 123,
    faults: Iterable[FaultSpec] = DEFAULT_FAULTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject deterministic fault families and return corrupted data plus sample labels."""
    df = baseline.copy(deep=True)
    t = df["time_s"].to_numpy()
    rng = np.random.default_rng(seed)
    labels = pd.DataFrame({"time_s": t})

    specs = {f.name: f for f in faults}
    for name, spec in specs.items():
        mask = (t >= spec.start_s) & (t < spec.end_s)
        labels[f"fault_{name}"] = mask.astype(np.int8)

        if name == "gyro_bias":
            df.loc[mask, spec.sensor] += 12.0
        elif name == "temperature_drift":
            frac = (t[mask] - spec.start_s) / (spec.end_s - spec.start_s)
            df.loc[mask, spec.sensor] += 6.0 * frac
        elif name == "pressure_stuck":
            idx = max(int(np.searchsorted(t, spec.start_s)) - 1, 0)
            df.loc[mask, spec.sensor] = float(df.iloc[idx][spec.sensor])
        elif name == "voltage_dropout":
            df.loc[mask, spec.sensor] = np.nan
        elif name == "vibration_spike":
            candidates = np.where(mask)[0]
            step = max(round(0.40 * fs_hz), 1)
            spike_idx = candidates[::step]
            df.loc[spike_idx, spec.sensor] += rng.choice([-1.0, 1.0], len(spike_idx)) * 1.3
            labels[f"fault_{name}"] = 0
            labels.loc[spike_idx, f"fault_{name}"] = 1
        elif name == "vibration_noise":
            df.loc[mask, spec.sensor] += rng.normal(0.0, 0.20, int(mask.sum()))
        elif name == "rpm_gain":
            df.loc[mask, spec.sensor] *= 1.06
        elif name == "vibration_interference":
            df.loc[mask, spec.sensor] += 0.28 * np.sin(2 * np.pi * 90.0 * t[mask])
        elif name == "bearing_degradation":
            frac = (t[mask] - spec.start_s) / (spec.end_s - spec.start_s)
            f1 = np.clip(df.loc[mask, "rpm_model"].to_numpy() / 60.0, 1e-6, None)
            phase = np.cumsum(2 * np.pi * f1 / fs_hz)
            extra = frac * (0.28 * np.sin(phase) + 0.10 * np.sin(2 * phase + 0.3))
            df.loc[mask, spec.sensor] += extra
        else:
            raise ValueError(f"Unsupported fault: {name}")

    fault_cols = [c for c in labels.columns if c.startswith("fault_")]
    labels["fault_any"] = (labels[fault_cols].sum(axis=1) > 0).astype(np.int8)
    return df, labels
