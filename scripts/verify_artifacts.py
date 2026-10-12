from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


REQUIRED = [
    "run_summary.json",
    "metrics/detector_comparison_window_level.csv",
    "metrics/fault_specific_metrics.csv",
    "metrics/diagnosed_events.csv",
    "data/fault_manifest.csv",
    "data/window_features.csv",
    "data/ml_window_predictions.csv",
    "models/isolation_forest.pkl",
    "models/autoencoder_state.pt",
    "figures/01_healthy_rpm.png",
    "figures/03_vibration_fft.png",
    "figures/04_butterworth_response.png",
    "figures/06_fault_timeline.png",
    "figures/09_detector_f1_comparison.png",
]


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/demo")
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        print("Missing required artifacts:")
        for name in missing:
            print(f" - {name}")
        return 1

    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "pipeline_completed":
        print("Run summary does not report pipeline_completed")
        return 1

    comparison = pd.read_csv(root / "metrics/detector_comparison_window_level.csv")
    expected = {"threshold_rules", "statistical", "residual_based", "isolation_forest", "autoencoder"}
    if set(comparison["detector"]) != expected:
        print("Detector comparison is incomplete")
        return 1

    faults = pd.read_csv(root / "data/fault_manifest.csv")
    if len(faults) != 9:
        print(f"Expected 9 fault families, found {len(faults)}")
        return 1

    print("Artifact verification PASSED")
    print(f"Samples: {summary['samples']}")
    print(f"Feature windows: {summary['feature_windows']}")
    print(f"Fault families: {summary['fault_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
