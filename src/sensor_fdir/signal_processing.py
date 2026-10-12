from __future__ import annotations

import numpy as np
from scipy import signal


def nyquist_frequency(fs_hz: float) -> float:
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    return fs_hz / 2.0


def alias_frequency(f_hz: float, fs_hz: float) -> float:
    """Return the baseband alias frequency in [0, fs/2]."""
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    return float(abs(((f_hz + fs_hz / 2.0) % fs_hz) - fs_hz / 2.0))


def butter_filter(
    x: np.ndarray,
    fs_hz: float,
    cutoff_hz: float | tuple[float, float],
    order: int = 4,
    btype: str = "lowpass",
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        raise ValueError("butter_filter does not accept NaNs; impute or segment first")
    sos = signal.butter(order, cutoff_hz, btype=btype, fs=fs_hz, output="sos")
    return signal.sosfiltfilt(sos, x)


def butter_response(
    fs_hz: float,
    cutoff_hz: float | tuple[float, float],
    order: int = 4,
    btype: str = "lowpass",
    n: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    sos = signal.butter(order, cutoff_hz, btype=btype, fs=fs_hz, output="sos")
    f, h = signal.sosfreqz(sos, worN=n, fs=fs_hz)
    return f, h


def fft_spectrum(x: np.ndarray, fs_hz: float, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be one-dimensional with at least two samples")
    x = x - np.nanmean(x)
    if np.isnan(x).any():
        raise ValueError("fft_spectrum does not accept NaNs")
    win = signal.get_window(window, len(x), fftbins=True)
    spectrum = np.fft.rfft(x * win)
    freq = np.fft.rfftfreq(len(x), d=1.0 / fs_hz)
    amp = 2.0 * np.abs(spectrum) / np.sum(win)
    return freq, amp


def welch_psd(x: np.ndarray, fs_hz: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        raise ValueError("Not enough finite samples for Welch PSD")
    if nperseg is None:
        nperseg = min(1024, len(x))
    return signal.welch(x, fs=fs_hz, window="hann", nperseg=nperseg, detrend="constant")


def moving_average(x: np.ndarray, samples: int) -> np.ndarray:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    kernel = np.ones(samples, dtype=float) / samples
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


def standardize(x: np.ndarray, mean: float, std: float, eps: float = 1e-12) -> np.ndarray:
    return (np.asarray(x, dtype=float) - mean) / max(float(std), eps)
