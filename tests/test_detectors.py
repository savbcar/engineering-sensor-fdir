import numpy as np

from sensor_fdir.config import SimulationConfig
from sensor_fdir.detectors import ResidualDetector, StatisticalDetector, ThresholdRuleDetector
from sensor_fdir.faults import inject_faults
from sensor_fdir.simulation import generate_baseline


def _data():
    cfg = SimulationConfig(fs_hz=200.0, duration_s=240.0, seed=5)
    healthy = generate_baseline(cfg)
    faulty, labels = inject_faults(healthy, cfg.fs_hz, seed=10)
    cal = faulty[faulty.time_s < 40]
    return cfg, faulty, labels, cal


def test_rule_detector_finds_dropout():
    cfg, faulty, labels, _ = _data()
    pred = ThresholdRuleDetector(cfg.fs_hz).predict(faulty)
    truth = labels["fault_voltage_dropout"].to_numpy(dtype=bool)
    assert pred.loc[truth, "voltage_dropout"].mean() == 1.0


def test_statistical_detector_runs_and_returns_scores():
    _, faulty, _, cal = _data()
    det = StatisticalDetector().fit(cal)
    flags, score = det.predict(faulty)
    assert len(flags) == len(faulty)
    assert np.isfinite(score).all()


def test_residual_detector_detects_gyro_bias_and_rpm_gain():
    cfg, faulty, labels, cal = _data()
    det = ResidualDetector(cfg.fs_hz).fit(cal)
    flags, scores = det.predict(faulty)
    gyro = labels["fault_gyro_bias"].to_numpy(dtype=bool)
    gain = labels["fault_rpm_gain"].to_numpy(dtype=bool)
    assert flags.loc[gyro, "gyro_bias"].mean() > 0.95
    assert flags.loc[gain, "rpm_gain"].mean() > 0.95
    assert "residual_max_score" in scores.columns
