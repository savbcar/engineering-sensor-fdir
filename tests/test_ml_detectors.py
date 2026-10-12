import numpy as np

from sensor_fdir.detectors import AutoencoderDetector, IsolationForestDetector


def test_isolation_forest_and_autoencoder_execute():
    rng = np.random.default_rng(4)
    healthy = rng.normal(0, 1, size=(60, 8))
    mixed = np.vstack([healthy[:10], rng.normal(8, 0.2, size=(5, 8))])

    iso = IsolationForestDetector(random_state=4).fit(healthy)
    pred_iso, score_iso = iso.predict(mixed)
    assert pred_iso.shape == (15,)
    assert np.isfinite(score_iso).all()

    ae = AutoencoderDetector(hidden_dim=8, latent_dim=3, epochs=3, seed=4).fit(healthy)
    pred_ae, score_ae = ae.predict(mixed)
    assert pred_ae.shape == (15,)
    assert np.isfinite(score_ae).all()
