from sensor_fdir.config import SimulationConfig
from sensor_fdir.features import extract_window_features, feature_matrix
from sensor_fdir.simulation import generate_baseline


def test_feature_extraction_has_time_and_frequency_features():
    cfg = SimulationConfig(fs_hz=200.0, duration_s=6.0, seed=2)
    df = generate_baseline(cfg)
    feat = extract_window_features(df, cfg.fs_hz, window_s=2.0, stride_s=1.0)
    assert len(feat) == 5
    assert "rpm__rms" in feat.columns
    assert "accel_x_g__dominant_frequency_hz" in feat.columns
    assert "accel_y_g__harmonic_power_ratio" in feat.columns
    x, names = feature_matrix(feat)
    assert x.shape[0] == 5
    assert x.shape[1] == len(names)
