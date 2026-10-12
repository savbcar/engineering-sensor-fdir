from __future__ import annotations

import pandas as pd


class ThresholdRuleDetector:
    """Simple engineering thresholds; intentionally limited and transparent."""

    def __init__(self, fs_hz: float) -> None:
        self.fs_hz = fs_hz

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"time_s": df["time_s"].to_numpy()})
        w = max(int(round(self.fs_hz)), 2)
        p_std = df["pressure_bar"].rolling(w, min_periods=w).std()
        ay_std = df["accel_y_g"].rolling(w, min_periods=w).std()
        temp_rate = df["temperature_C"].diff().rolling(w, min_periods=w).mean() * self.fs_hz

        out["pressure_low_variance"] = (p_std < 0.003).fillna(False).astype(int)
        out["voltage_dropout"] = df["voltage_V"].isna().astype(int)
        out["vibration_spike"] = (df["accel_x_g"].abs() > 0.90).astype(int)
        out["vibration_high_noise"] = (ay_std > 0.16).fillna(False).astype(int)
        out["temperature_fast_change"] = (temp_rate.abs() > 0.60).fillna(False).astype(int)
        out["rpm_out_of_range"] = ((df["rpm"] < -100) | (df["rpm"] > 3600)).astype(int)
        flag_cols = [c for c in out.columns if c != "time_s"]
        out["det_any"] = (out[flag_cols].sum(axis=1) > 0).astype(int)
        return out
