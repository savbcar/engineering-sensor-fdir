import numpy as np

from sensor_fdir.config import SimulationConfig
from sensor_fdir.simulation import generate_baseline


def test_baseline_shape_and_sampling():
    cfg = SimulationConfig(fs_hz=100.0, duration_s=5.0, seed=1)
    df = generate_baseline(cfg)
    assert len(df) == 500
    assert np.isclose(np.diff(df["time_s"]).mean(), 0.01)
    assert {"rpm", "gyro_z_rad_s", "temperature_C", "accel_x_g"}.issubset(df.columns)


def test_baseline_is_deterministic():
    cfg = SimulationConfig(fs_hz=100.0, duration_s=5.0, seed=7)
    a = generate_baseline(cfg)
    b = generate_baseline(cfg)
    assert np.allclose(a["rpm"], b["rpm"])
