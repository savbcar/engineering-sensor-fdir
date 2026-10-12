from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Engineering Sensor FDIR platform demo.")
    parser.add_argument("--config", default="config/default.json", help="Path to JSON configuration")
    parser.add_argument("--output", default="artifacts/demo", help="Output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    result = run_pipeline(cfg, Path(args.output))
    comparison = result["comparison"]
    print("\nWindow-level detector comparison")
    print(comparison[["detector", "precision", "recall", "f1", "false_positive_rate", "roc_auc"]].to_string(index=False))
    print(f"\nArtifacts written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
