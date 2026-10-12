from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .signal_processing import welch_psd

SENSOR_COLUMNS = (
    "rpm",
    "gyro_z_rad_s",
    "temperature_C",
    "pressure_bar",
    "voltage_V",
    "accel_x_g",
    "accel_y_g",
)


def _safe_skew(x: np.ndarray) -> float:
    return float(stats.skew(x, bias=False)) if len(x) >= 3 and np.std(x) > 1e-12 else 0.0


def _safe_kurtosis(x: np.ndarray) -> float:
    return float(stats.kurtosis(x, fisher=True, bias=False)) if len(x) >= 4 and np.std(x) > 1e-12 else 0.0


def _band_power(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freq >= low) & (freq < high)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(psd[mask], freq[mask]))


def _spectral_features(x: np.ndarray, fs_hz: float, rpm_mean: float) -> dict[str, float]:
    finite = np.isfinite(x)
    x = x[finite]
    if len(x) < max(16, int(0.25 * fs_hz)):
        return {
            "dominant_frequency_hz": 0.0,
            "spectral_centroid_hz": 0.0,
            "bandpower_0_20": 0.0,
            "bandpower_20_80": 0.0,
            "bandpower_80_180": 0.0,
            "harmonic_power_ratio": 0.0,
        }
    freq, psd = welch_psd(x, fs_hz, nperseg=min(len(x), int(fs_hz)))
    if len(freq) <= 1:
        dom = 0.0
    else:
        dom = float(freq[1 + np.argmax(psd[1:])])
    total = float(np.trapezoid(psd, freq)) + 1e-15
    centroid = float(np.sum(freq * psd) / (np.sum(psd) + 1e-15))
    rot_hz = max(rpm_mean / 60.0, 0.0)
    harmonic_power = 0.0
    for multiple in (1.0, 2.0, 3.0):
        center = multiple * rot_hz
        harmonic_power += _band_power(freq, psd, max(0.0, center - 2.0), center + 2.0)
    return {
        "dominant_frequency_hz": dom,
        "spectral_centroid_hz": centroid,
        "bandpower_0_20": _band_power(freq, psd, 0.0, 20.0),
        "bandpower_20_80": _band_power(freq, psd, 20.0, 80.0),
        "bandpower_80_180": _band_power(freq, psd, 80.0, min(180.0, fs_hz / 2.0)),
        "harmonic_power_ratio": harmonic_power / total,
    }


def extract_window_features(
    df: pd.DataFrame,
    fs_hz: float,
    window_s: float = 2.0,
    stride_s: float = 0.5,
) -> pd.DataFrame:
    """Extract time- and frequency-domain features from overlapping windows."""
    win = int(round(window_s * fs_hz))
    stride = int(round(stride_s * fs_hz))
    if win < 8 or stride < 1:
        raise ValueError("Window/stride configuration is too small")
    if len(df) < win:
        raise ValueError("Input shorter than one feature window")

    rows: list[dict[str, float]] = []
    for start in range(0, len(df) - win + 1, stride):
        stop = start + win
        seg = df.iloc[start:stop]
        row: dict[str, float] = {
            "window_start_s": float(seg["time_s"].iloc[0]),
            "window_end_s": float(seg["time_s"].iloc[-1]),
            "window_mid_s": float((seg["time_s"].iloc[0] + seg["time_s"].iloc[-1]) / 2.0),
        }
        for col in SENSOR_COLUMNS:
            x = seg[col].to_numpy(dtype=float)
            finite = x[np.isfinite(x)]
            prefix = f"{col}__"
            row[prefix + "dropout_fraction"] = float(1.0 - len(finite) / len(x))
            if len(finite) == 0:
                for name in ("mean", "rms", "variance", "skewness", "kurtosis", "peak_to_peak", "derivative_rms", "slope"):
                    row[prefix + name] = 0.0
                continue
            row[prefix + "mean"] = float(np.mean(finite))
            row[prefix + "rms"] = float(np.sqrt(np.mean(finite**2)))
            row[prefix + "variance"] = float(np.var(finite, ddof=1)) if len(finite) > 1 else 0.0
            row[prefix + "skewness"] = _safe_skew(finite)
            row[prefix + "kurtosis"] = _safe_kurtosis(finite)
            row[prefix + "peak_to_peak"] = float(np.ptp(finite))
            row[prefix + "derivative_rms"] = float(np.sqrt(np.mean(np.diff(finite) ** 2))) if len(finite) > 1 else 0.0
            if len(finite) > 1:
                tt = np.linspace(0.0, window_s, len(finite), endpoint=False)
                row[prefix + "slope"] = float(np.polyfit(tt, finite, 1)[0])
            else:
                row[prefix + "slope"] = 0.0


        residual_defs = {
            "rpm": seg["rpm"].to_numpy(dtype=float) - seg["rpm_model"].to_numpy(dtype=float),
            "gyro": seg["gyro_z_rad_s"].to_numpy(dtype=float) - seg["rpm_model"].to_numpy(dtype=float) * 2 * np.pi / 60.0,
            "temperature": seg["temperature_C"].to_numpy(dtype=float) - seg["temperature_model_C"].to_numpy(dtype=float),
            "pressure": seg["pressure_bar"].to_numpy(dtype=float) - seg["pressure_model_bar"].to_numpy(dtype=float),
            "voltage": seg["voltage_V"].to_numpy(dtype=float) - seg["voltage_model_V"].to_numpy(dtype=float),
        }
        for name, r in residual_defs.items():
            finite_r = r[np.isfinite(r)]
            prefix = f"residual__{name}__"
            if len(finite_r) == 0:
                row[prefix + "mean"] = 0.0
                row[prefix + "rms"] = 0.0
                row[prefix + "std"] = 0.0
                row[prefix + "slope"] = 0.0
                continue
            row[prefix + "mean"] = float(np.mean(finite_r))
            row[prefix + "rms"] = float(np.sqrt(np.mean(finite_r**2)))
            row[prefix + "std"] = float(np.std(finite_r, ddof=1)) if len(finite_r) > 1 else 0.0
            if len(finite_r) > 1:
                tt_r = np.linspace(0.0, window_s, len(finite_r), endpoint=False)
                row[prefix + "slope"] = float(np.polyfit(tt_r, finite_r, 1)[0])
            else:
                row[prefix + "slope"] = 0.0

        rpm_mean = float(np.nanmean(seg["rpm"].to_numpy(dtype=float)))
        for col in ("accel_x_g", "accel_y_g"):
            spec = _spectral_features(seg[col].to_numpy(dtype=float), fs_hz, rpm_mean)
            for key, value in spec.items():
                row[f"{col}__{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def feature_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in features.columns if not c.startswith("window_")]
    return features[names].to_numpy(dtype=float), names


def diagnostic_feature_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Operating-point-compensated feature subset used by unsupervised detectors."""
    names = [
        c for c in features.columns
        if c.startswith("residual__")
        or c.startswith("accel_x_g__")
        or c.startswith("accel_y_g__")
        or c.endswith("__dropout_fraction")
    ]
    return features[names].to_numpy(dtype=float), names
