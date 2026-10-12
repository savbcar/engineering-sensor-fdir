from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    out: dict[str, float | int] = {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(score, dtype=float)))
    else:
        out["roc_auc"] = float("nan")
    return out


def first_detection_delay(time_s: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    time_s = np.asarray(time_s, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    true_idx = np.flatnonzero(y_true == 1)
    if len(true_idx) == 0:
        return float("nan")
    start = true_idx[0]
    candidates = np.flatnonzero((np.arange(len(y_pred)) >= start) & (y_pred == 1))
    if len(candidates) == 0:
        return float("nan")
    return max(0.0, float(time_s[candidates[0]] - time_s[start]))


def window_labels(labels: pd.DataFrame, features: pd.DataFrame, column: str = "fault_any") -> np.ndarray:
    t = labels["time_s"].to_numpy(dtype=float)
    y = labels[column].to_numpy(dtype=int)
    out = np.zeros(len(features), dtype=int)
    for i, row in features.iterrows():
        mask = (t >= float(row["window_start_s"])) & (t <= float(row["window_end_s"]))
        out[i] = int(np.any(y[mask] == 1))
    return out


def aggregate_sample_predictions(time_s: np.ndarray, pred: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    pred = np.asarray(pred, dtype=int)
    out = np.zeros(len(features), dtype=int)
    for i, row in features.iterrows():
        mask = (time_s >= float(row["window_start_s"])) & (time_s <= float(row["window_end_s"]))
        out[i] = int(np.any(pred[mask] == 1))
    return out


def fault_specific_metrics(labels: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "gyro_bias": "fault_gyro_bias",
        "temperature_drift": "fault_temperature_drift",
        "pressure_stuck": "fault_pressure_stuck",
        "voltage_dropout": "fault_voltage_dropout",
        "vibration_spike": "fault_vibration_spike",
        "vibration_noise": "fault_vibration_noise",
        "rpm_gain": "fault_rpm_gain",
        "vibration_interference": "fault_vibration_interference",
        "bearing_degradation": "fault_bearing_degradation",
    }
    rows: list[dict[str, float | int | str]] = []
    for pred_col, label_col in mapping.items():
        y = labels[label_col].to_numpy(dtype=int)
        yp = flags[pred_col].to_numpy(dtype=int)
        row: dict[str, float | int | str] = {"fault": pred_col}
        row.update(binary_metrics(y, yp))
        row["detection_delay_s"] = first_detection_delay(labels["time_s"].to_numpy(), y, yp)
        rows.append(row)
    return pd.DataFrame(rows)


def contiguous_events(time_s: np.ndarray, flags: pd.DataFrame, min_duration_s: float = 0.02) -> pd.DataFrame:
    """Convert fault-specific binary flags to an event list for operator-facing diagnosis."""
    t = np.asarray(time_s, dtype=float)
    rows: list[dict[str, float | str]] = []
    mapping = {
        "gyro_bias": ("gyro_z_rad_s", "bias"),
        "temperature_drift": ("temperature_C", "drift"),
        "pressure_stuck": ("pressure_bar", "stuck"),
        "voltage_dropout": ("voltage_V", "dropout"),
        "vibration_spike": ("accel_x_g", "spike"),
        "vibration_noise": ("accel_y_g", "increased_noise"),
        "rpm_gain": ("rpm", "gain_error"),
        "vibration_interference": ("accel_x_g", "sinusoidal_interference"),
        "bearing_degradation": ("accel_y_g", "slow_degradation"),
    }
    for col, (sensor, kind) in mapping.items():
        y = flags[col].to_numpy(dtype=int)
        padded = np.r_[0, y, 0]
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1) - 1
        for s, e in zip(starts, stops):
            duration = float(t[e] - t[s]) if e > s else 0.0
            if duration >= min_duration_s or kind in {"spike", "dropout"}:
                rows.append(
                    {
                        "sensor": sensor,
                        "fault_type": kind,
                        "start_s": float(t[s]),
                        "end_s": float(t[e]),
                        "duration_s": duration,
                    }
                )
    return pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True) if rows else pd.DataFrame()
