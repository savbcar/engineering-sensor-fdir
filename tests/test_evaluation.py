import numpy as np

from sensor_fdir.evaluation import binary_metrics, first_detection_delay


def test_binary_metrics_known_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    m = binary_metrics(y, p)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert np.isclose(m["recall"], 1.0)


def test_detection_delay():
    t = np.arange(6, dtype=float)
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0, 0, 0, 1, 1, 0])
    assert first_detection_delay(t, y, p) == 1.0
