from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from .config import Config
from .detectors import (
    AutoencoderDetector,
    IsolationForestDetector,
    ResidualDetector,
    StatisticalDetector,
    ThresholdRuleDetector,
)
from .evaluation import (
    aggregate_sample_predictions,
    binary_metrics,
    contiguous_events,
    fault_specific_metrics,
    window_labels,
)
from .faults import fault_manifest, inject_faults
from .features import diagnostic_feature_matrix, extract_window_features
from .plotting import (
    plot_aliasing_demo,
    plot_f1_comparison,
    plot_fault_timeline,
    plot_fft,
    plot_filter_response,
    plot_score,
    plot_signal,
    plot_two_signals,
)
from .simulation import generate_baseline


def run_pipeline(cfg: Config, output_dir: str | Path) -> dict[str, object]:
    out = Path(output_dir)
    data_dir = out / "data"
    fig_dir = out / "figures"
    metrics_dir = out / "metrics"
    model_dir = out / "models"
    for d in (data_dir, fig_dir, metrics_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    fs = cfg.simulation.fs_hz
    baseline = generate_baseline(cfg.simulation)
    faulty, labels = inject_faults(baseline, fs_hz=fs, seed=cfg.simulation.seed + 100)
    manifest = fault_manifest()

    baseline.to_csv(data_dir / "healthy_signals.csv", index=False)
    faulty.to_csv(data_dir / "faulty_signals.csv", index=False)
    labels.to_csv(data_dir / "fault_labels.csv", index=False)
    manifest.to_csv(data_dir / "fault_manifest.csv", index=False)

    cal_mask = faulty["time_s"] < cfg.pipeline.calibration_end_s
    calibration = faulty.loc[cal_mask].copy()

    rules = ThresholdRuleDetector(fs)
    rule_flags = rules.predict(faulty)
    rule_flags.to_csv(data_dir / "threshold_rule_flags.csv", index=False)

    stat = StatisticalDetector(threshold=6.0).fit(calibration)
    stat_flags, stat_score = stat.predict(faulty)
    stat_flags.to_csv(data_dir / "statistical_flags.csv", index=False)

    residual = ResidualDetector(fs_hz=fs, sigma_threshold=6.0).fit(calibration)
    residual_flags, residual_scores = residual.predict(faulty)
    residual_flags.to_csv(data_dir / "residual_flags.csv", index=False)
    residual_scores.to_csv(data_dir / "residual_scores.csv", index=False)

    fault_metrics = fault_specific_metrics(labels, residual_flags)
    fault_metrics.to_csv(metrics_dir / "fault_specific_metrics.csv", index=False)
    events = contiguous_events(labels["time_s"].to_numpy(), residual_flags)
    events.to_csv(metrics_dir / "diagnosed_events.csv", index=False)

    features = extract_window_features(
        faulty,
        fs_hz=fs,
        window_s=cfg.pipeline.feature_window_s,
        stride_s=cfg.pipeline.feature_stride_s,
    )
    x, feature_names = diagnostic_feature_matrix(features)
    healthy_features = extract_window_features(
        baseline,
        fs_hz=fs,
        window_s=cfg.pipeline.feature_window_s,
        stride_s=cfg.pipeline.feature_stride_s,
    )
    x_train, healthy_feature_names = diagnostic_feature_matrix(healthy_features)
    if healthy_feature_names != feature_names:
        raise RuntimeError("Healthy and faulty feature schemas differ")
    y_window = window_labels(labels, features, "fault_any")
    features_out = features.copy()
    features_out["fault_any"] = y_window
    features_out.to_csv(data_dir / "window_features.csv", index=False)
    healthy_features.to_csv(data_dir / "healthy_window_features.csv", index=False)

    iso = IsolationForestDetector(random_state=cfg.simulation.seed).fit(x_train)
    iso_pred, iso_score = iso.predict(x)
    with (model_dir / "isolation_forest.pkl").open("wb") as f:
        pickle.dump(iso, f)

    ae = AutoencoderDetector(
        hidden_dim=cfg.pipeline.autoencoder_hidden_dim,
        latent_dim=cfg.pipeline.autoencoder_latent_dim,
        epochs=cfg.pipeline.autoencoder_epochs,
        learning_rate=cfg.pipeline.autoencoder_learning_rate,
        seed=cfg.simulation.seed,
    ).fit(x_train)
    ae_pred, ae_score = ae.predict(x)
    torch.save(
        {
            "state_dict": ae.model.state_dict() if ae.model is not None else {},
            "feature_names": feature_names,
            "threshold": ae.threshold,
            "config": {
                "hidden_dim": ae.hidden_dim,
                "latent_dim": ae.latent_dim,
            },
        },
        model_dir / "autoencoder_state.pt",
    )

    ml = pd.DataFrame(
        {
            "window_mid_s": features["window_mid_s"],
            "fault_any": y_window,
            "isolation_forest_pred": iso_pred,
            "isolation_forest_score": iso_score,
            "autoencoder_pred": ae_pred,
            "autoencoder_score": ae_score,
        }
    )
    ml.to_csv(data_dir / "ml_window_predictions.csv", index=False)

    t = faulty["time_s"].to_numpy(dtype=float)
    detector_windows = {
        "threshold_rules": aggregate_sample_predictions(t, rule_flags["det_any"].to_numpy(), features),
        "statistical": aggregate_sample_predictions(t, stat_flags["det_any"].to_numpy(), features),
        "residual_based": aggregate_sample_predictions(t, residual_flags["det_any"].to_numpy(), features),
        "isolation_forest": iso_pred,
        "autoencoder": ae_pred,
    }
    rows: list[dict[str, object]] = []
    for name, pred in detector_windows.items():
        score = None
        if name == "isolation_forest":
            score = iso_score
        elif name == "autoencoder":
            score = ae_score
        row: dict[str, object] = {"detector": name, "evaluation_unit": f"{cfg.pipeline.feature_window_s:.1f} s windows / {cfg.pipeline.feature_stride_s:.1f} s stride"}
        row.update(binary_metrics(y_window, pred, score=score))
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(metrics_dir / "detector_comparison_window_level.csv", index=False)

    # Figures: one engineering question per figure.
    plot_signal(
        baseline["time_s"].to_numpy(), baseline["rpm"].to_numpy(),
        "Time [s]", "Speed [rpm]", "Healthy rotational-speed measurement", fig_dir / "01_healthy_rpm.png"
    )
    plot_two_signals(
        faulty["time_s"].to_numpy(), faulty["gyro_z_rad_s"].to_numpy(),
        faulty["rpm_model"].to_numpy() * 2 * np.pi / 60.0,
        ("Gyro measurement", "Model-expected angular rate"), "Angular velocity [rad/s]",
        "Gyro bias fault and physics residual", fig_dir / "02_gyro_bias_residual.png"
    )
    plot_fft(baseline, fs, fig_dir / "03_vibration_fft.png")
    plot_filter_response(fs, fig_dir / "04_butterworth_response.png")
    plot_aliasing_demo(fig_dir / "05_aliasing_demo.png")
    plot_fault_timeline(labels, fig_dir / "06_fault_timeline.png")
    plot_score(features["window_mid_s"].to_numpy(), iso_score, iso.threshold, "Isolation Forest anomaly score", fig_dir / "07_isolation_forest_score.png")
    plot_score(features["window_mid_s"].to_numpy(), ae_score, ae.threshold, "Autoencoder reconstruction-error score", fig_dir / "08_autoencoder_score.png")
    plot_f1_comparison(comparison, fig_dir / "09_detector_f1_comparison.png")

    summary = {
        "status": "pipeline_completed",
        "samples": len(faulty),
        "feature_windows": len(features),
        "training_windows": len(x_train),
        "sampling_frequency_hz": fs,
        "nyquist_hz": fs / 2.0,
        "fault_count": len(manifest),
        "feature_count": len(feature_names),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "ml_thresholds": {
            "isolation_forest": iso.threshold,
            "autoencoder": ae.threshold,
        },
        "autoencoder_final_training_loss": ae.loss_history[-1] if ae.loss_history else None,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "comparison": comparison,
        "fault_metrics": fault_metrics,
        "events": events,
    }
