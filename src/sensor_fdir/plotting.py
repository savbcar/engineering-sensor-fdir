from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .signal_processing import alias_frequency, butter_response, fft_spectrum


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_signal(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(x, y, linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _finish(fig, path)


def plot_two_signals(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, labels: tuple[str, str], ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(x, y1, label=labels[0], linewidth=1.0)
    ax.plot(x, y2, label=labels[1], linewidth=1.0, alpha=0.85)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finish(fig, path)


def plot_fft(df: pd.DataFrame, fs_hz: float, path: Path) -> None:
    mask = (df["time_s"] >= 25.0) & (df["time_s"] < 35.0)
    freq, amp = fft_spectrum(df.loc[mask, "accel_x_g"].to_numpy(), fs_hz)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(freq, amp, linewidth=1.0)
    ax.set_xlim(0, min(200.0, fs_hz / 2.0))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [g]")
    ax.set_title("Hann-windowed vibration FFT (healthy steady-state segment)")
    ax.grid(True, alpha=0.3)
    _finish(fig, path)


def plot_filter_response(fs_hz: float, path: Path) -> None:
    f, h = butter_response(fs_hz, 80.0, order=4, btype="lowpass")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.semilogx(f[1:], 20 * np.log10(np.maximum(np.abs(h[1:]), 1e-12)))
    ax.axvline(80.0, linestyle="--", label="80 Hz cutoff")
    ax.axhline(-3.0103, linestyle=":", label="-3.01 dB")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("4th-order Butterworth low-pass response")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _finish(fig, path)


def plot_aliasing_demo(path: Path) -> None:
    fs = 100.0
    f_true = 70.0
    f_alias = alias_frequency(f_true, fs)
    t = np.arange(0.0, 0.25, 1.0 / fs)
    y_true_samples = np.sin(2 * np.pi * f_true * t)
    y_alias = np.sin(2 * np.pi * f_alias * t)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(t, y_true_samples, marker="o", label=f"Samples of {f_true:.0f} Hz")
    ax.plot(t, -y_alias, linestyle="--", label=f"Equivalent sampled alias magnitude: {f_alias:.0f} Hz")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Aliasing demonstration: 70 Hz sampled at 100 Hz")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finish(fig, path)


def plot_fault_timeline(labels: pd.DataFrame, path: Path) -> None:
    cols = [c for c in labels.columns if c.startswith("fault_") and c != "fault_any"]
    t = labels["time_s"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for i, col in enumerate(cols):
        y = labels[col].to_numpy(dtype=float)
        ax.fill_between(t, i, i + 0.8 * y, step="pre", alpha=0.65)
    ax.set_yticks(np.arange(len(cols)) + 0.4)
    ax.set_yticklabels([c.replace("fault_", "") for c in cols])
    ax.set_xlabel("Time [s]")
    ax.set_title("Ground-truth injected fault timeline")
    ax.grid(True, axis="x", alpha=0.3)
    _finish(fig, path)


def plot_score(time_s: np.ndarray, score: np.ndarray, threshold: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(time_s, score, linewidth=1.0)
    ax.axhline(threshold, linestyle="--", label="Decision threshold")
    ax.set_xlabel("Window midpoint [s]")
    ax.set_ylabel("Anomaly score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finish(fig, path)


def plot_f1_comparison(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(metrics["detector"], metrics["f1"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Window-level detector comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    _finish(fig, path)
