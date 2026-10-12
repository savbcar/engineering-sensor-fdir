from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


class ResidualDetector:
    """Physics/residual-based detector with fault-specific diagnostic flags."""

    def __init__(self, fs_hz: float, sigma_threshold: float = 6.0) -> None:
        self.fs_hz = fs_hz
        self.sigma_threshold = sigma_threshold
        self.stats: dict[str, tuple[float, float]] = {}
        self.thresholds: dict[str, float] = {}

    @staticmethod
    def _robust_stats(x: np.ndarray) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        sigma = max(1.4826 * mad, float(np.std(x)), 1e-8)
        return med, sigma

    def _residuals(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "gyro": df["gyro_z_rad_s"].to_numpy(dtype=float) - df["rpm_model"].to_numpy(dtype=float) * 2 * np.pi / 60.0,
            "rpm": df["rpm"].to_numpy(dtype=float) - df["rpm_model"].to_numpy(dtype=float),
            "temp": df["temperature_C"].to_numpy(dtype=float) - df["temperature_model_C"].to_numpy(dtype=float),
            "pressure": df["pressure_bar"].to_numpy(dtype=float) - df["pressure_model_bar"].to_numpy(dtype=float),
            "voltage": df["voltage_V"].to_numpy(dtype=float) - df["voltage_model_V"].to_numpy(dtype=float),
        }

    def fit(self, healthy: pd.DataFrame) -> ResidualDetector:
        res = self._residuals(healthy)
        self.stats = {name: self._robust_stats(x) for name, x in res.items()}
        w = max(round(self.fs_hz), 4)
        pstd = healthy["pressure_bar"].rolling(w, min_periods=w).std().dropna().to_numpy()

        ax = healthy["accel_x_g"].to_numpy(dtype=float)
        med_ax, sig_ax = self._robust_stats(ax)
        self.thresholds["accel_x_median"] = med_ax
        self.thresholds["accel_x_sigma"] = sig_ax
        self.thresholds["pressure_std_low"] = max(float(np.quantile(pstd, 0.02)) * 0.35, 1e-6)

        ay = healthy["accel_y_g"].to_numpy(dtype=float)
        sos_noise = signal.butter(4, (0.80 * (self.fs_hz / 2.0), 0.96 * (self.fs_hz / 2.0)), btype="bandpass", fs=self.fs_hz, output="sos")
        noise_band = signal.sosfiltfilt(sos_noise, ay)
        noise_rms = np.sqrt(pd.Series(noise_band**2).rolling(w, min_periods=w).mean()).dropna().to_numpy()
        self.thresholds["noise_band_rms_high"] = float(np.quantile(noise_rms, 0.995)) * 1.8

        sos_rot = signal.butter(4, (35.0, 70.0), btype="bandpass", fs=self.fs_hz, output="sos")
        rot_band = signal.sosfiltfilt(sos_rot, ay)
        rot_rms = np.sqrt(pd.Series(rot_band**2).rolling(2 * w, min_periods=2 * w).mean()).dropna().to_numpy()
        self.thresholds["rot_band_rms_high"] = float(np.quantile(rot_rms, 0.995)) * 1.55

        # Narrow-band healthy threshold for 90 Hz interference.
        sos = signal.butter(4, (85.0, 95.0), btype="bandpass", fs=self.fs_hz, output="sos")
        band = signal.sosfiltfilt(sos, ax)
        brms = np.sqrt(pd.Series(band**2).rolling(w, min_periods=w).mean()).dropna().to_numpy()
        self.thresholds["band90_rms_high"] = float(np.quantile(brms, 0.995)) * 2.5
        return self

    def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.stats:
            raise RuntimeError("Detector must be fitted before predict")
        len(df)
        w = max(round(self.fs_hz), 4)
        w2 = max(round(2.0 * self.fs_hz), 4)
        res = self._residuals(df)
        flags = pd.DataFrame({"time_s": df["time_s"].to_numpy()})
        scores = pd.DataFrame({"time_s": df["time_s"].to_numpy()})

        def zscore(name: str) -> np.ndarray:
            mu, sigma = self.stats[name]
            z = np.abs((res[name] - mu) / sigma)
            z[~np.isfinite(z)] = self.sigma_threshold + 2.0
            scores[f"{name}_residual_z"] = z
            return z

        z_gyro = zscore("gyro")
        z_rpm = zscore("rpm")
        z_temp_raw = (res["temp"] - self.stats["temp"][0]) / self.stats["temp"][1]
        z_voltage = zscore("voltage")
        _ = zscore("pressure")

        temp_roll = pd.Series(z_temp_raw).rolling(w2, min_periods=w2).mean().abs().to_numpy()
        flags["gyro_bias"] = (z_gyro > self.sigma_threshold).astype(int)
        flags["rpm_gain"] = (z_rpm > self.sigma_threshold).astype(int)
        flags["temperature_drift"] = (temp_roll > self.sigma_threshold).astype(int)
        flags["voltage_dropout"] = df["voltage_V"].isna().astype(int)

        pstd = df["pressure_bar"].rolling(w, min_periods=w).std()
        flags["pressure_stuck"] = (pstd < self.thresholds["pressure_std_low"]).fillna(False).astype(int)

        ax = df["accel_x_g"].to_numpy(dtype=float)
        spike_score = np.abs((ax - self.thresholds["accel_x_median"]) / self.thresholds["accel_x_sigma"])
        flags["vibration_spike"] = (spike_score > 7.5).astype(int)
        scores["spike_score"] = spike_score

        ay = df["accel_y_g"].to_numpy(dtype=float)
        sos_noise = signal.butter(4, (0.80 * (self.fs_hz / 2.0), 0.96 * (self.fs_hz / 2.0)), btype="bandpass", fs=self.fs_hz, output="sos")
        noise_band = signal.sosfiltfilt(sos_noise, ay)
        noise_rms = np.sqrt(pd.Series(noise_band**2).rolling(w, min_periods=w).mean()).to_numpy()
        flags["vibration_noise"] = (noise_rms > self.thresholds["noise_band_rms_high"]).astype(int)
        scores["noise_band_rms"] = np.nan_to_num(noise_rms, nan=0.0)

        sos90 = signal.butter(4, (85.0, 95.0), btype="bandpass", fs=self.fs_hz, output="sos")
        band90 = signal.sosfiltfilt(sos90, ax)
        band90_rms = np.sqrt(pd.Series(band90**2).rolling(w, min_periods=w).mean()).to_numpy()
        flags["vibration_interference"] = (band90_rms > self.thresholds["band90_rms_high"]).astype(int)
        scores["band90_rms"] = np.nan_to_num(band90_rms, nan=0.0)

        # Low-frequency rotating-band RMS; broadband noise is suppressed by the band-pass.
        sosrot = signal.butter(4, (35.0, 70.0), btype="bandpass", fs=self.fs_hz, output="sos")
        rotband = signal.sosfiltfilt(sosrot, ay)
        rot_rms = np.sqrt(pd.Series(rotband**2).rolling(w2, min_periods=w2).mean()).to_numpy()
        flags["bearing_degradation"] = ((rot_rms > self.thresholds["rot_band_rms_high"]) & (noise_rms <= self.thresholds["noise_band_rms_high"])).astype(int)
        scores["rotating_band_rms"] = np.nan_to_num(rot_rms, nan=0.0)

        fault_cols = [c for c in flags.columns if c != "time_s"]
        flags["det_any"] = (flags[fault_cols].sum(axis=1) > 0).astype(int)
        scores["residual_max_score"] = np.nanmax(
            np.column_stack([z_gyro, z_rpm, np.nan_to_num(temp_roll), z_voltage]), axis=1
        )
        return flags, scores
