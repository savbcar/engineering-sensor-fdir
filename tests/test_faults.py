import numpy as np

from sensor_fdir.config import SimulationConfig
from sensor_fdir.faults import inject_faults
from sensor_fdir.simulation import generate_baseline


def test_all_fault_labels_exist():
    cfg = SimulationConfig(fs_hz=100.0, duration_s=240.0, seed=3)
    healthy = generate_baseline(cfg)
    _, labels = inject_faults(healthy, cfg.fs_hz)
    expected = {
        "fault_gyro_bias", "fault_temperature_drift", "fault_pressure_stuck",
        "fault_voltage_dropout", "fault_vibration_spike", "fault_vibration_noise",
        "fault_rpm_gain", "fault_vibration_interference", "fault_bearing_degradation", "fault_any",
    }
    assert expected.issubset(labels.columns)


def test_dropout_is_nan_and_gain_changes_rpm():
    cfg = SimulationConfig(fs_hz=100.0, duration_s=240.0, seed=3)
    healthy = generate_baseline(cfg)
    faulty, _ = inject_faults(healthy, cfg.fs_hz)
    drop = (faulty.time_s >= 125) & (faulty.time_s < 128)
    gain = (faulty.time_s >= 175) & (faulty.time_s < 190)
    assert faulty.loc[drop, "voltage_V"].isna().all()
    ratio = faulty.loc[gain, "rpm"].to_numpy() / healthy.loc[gain, "rpm"].to_numpy()
    assert np.allclose(np.nanmedian(ratio), 1.06, atol=1e-3)
