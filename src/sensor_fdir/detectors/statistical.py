from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import SENSOR_COLUMNS


class StatisticalDetector:
    """Channel-wise robust z-score detector learned only from a healthy calibration interval."""

    def __init__(self, threshold: float = 6.0) -> None:
        self.threshold = threshold
        self.medians: dict[str, float] = {}
        self.scales: dict[str, float] = {}

    def fit(self, healthy: pd.DataFrame) -> StatisticalDetector:
        for col in SENSOR_COLUMNS:
            x = healthy[col].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            scale = max(1.4826 * mad, float(np.std(x)), 1e-6)
            self.medians[col] = med
            self.scales[col] = scale
        return self

    def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if not self.medians:
            raise RuntimeError("Detector must be fitted before predict")
        flags = pd.DataFrame({"time_s": df["time_s"].to_numpy()})
        max_score = np.zeros(len(df), dtype=float)
        for col in SENSOR_COLUMNS:
            x = df[col].to_numpy(dtype=float)
            z = np.abs((x - self.medians[col]) / self.scales[col])
            z[~np.isfinite(z)] = self.threshold + 1.0
            flags[f"{col}_z"] = (z > self.threshold).astype(int)
            max_score = np.maximum(max_score, z)
        cols = [c for c in flags.columns if c != "time_s"]
        flags["det_any"] = (flags[cols].sum(axis=1) > 0).astype(int)
        return flags, pd.Series(max_score, name="statistical_score")
