from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationConfig:
    fs_hz: float = 250.0
    duration_s: float = 240.0
    seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    calibration_end_s: float = 40.0
    feature_window_s: float = 2.0
    feature_stride_s: float = 1.0
    autoencoder_epochs: int = 40
    autoencoder_hidden_dim: int = 24
    autoencoder_latent_dim: int = 6
    autoencoder_learning_rate: float = 1e-3


@dataclass(frozen=True)
class Config:
    simulation: SimulationConfig = SimulationConfig()
    pipeline: PipelineConfig = PipelineConfig()


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config(
        simulation=SimulationConfig(**raw.get("simulation", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
    )
