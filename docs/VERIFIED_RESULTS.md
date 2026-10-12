# Verified Local Results

These results were produced by the repository's default configuration in the local verification environment. They are **synthetic benchmark results**, not claims of field performance.

## Reproducible run

- 60,000 samples
- 250 Hz sampling rate / 125 Hz Nyquist frequency
- 240 s scenario
- 9 injected fault families
- 239 overlapping feature windows (2.0 s window, 1.0 s stride)
- 55 operating-point-compensated diagnostic features used by the unsupervised models
- separate fault-free reference run used to fit Isolation Forest and the autoencoder

## Common window-level detector comparison

| Detector | Precision | Recall | F1 | False-positive rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Fixed threshold rules | 0.745 | 0.580 | 0.652 | 0.241 | n/a |
| Robust statistical | 1.000 | 0.137 | 0.242 | 0.000 | n/a |
| Physics/residual-based | 0.969 | 0.947 | 0.958 | 0.037 | n/a |
| Isolation Forest | 0.857 | 0.504 | 0.635 | 0.102 | 0.860 |
| PyTorch autoencoder | 0.977 | 0.969 | 0.973 | 0.028 | 0.996 |

The autoencoder has the best aggregate anomaly-detection F1 in this controlled benchmark, while the residual detector remains the strongest interpretable method and provides fault-specific isolation/identification flags. The Isolation Forest ranks anomalies reasonably (ROC-AUC ~0.86) but has materially lower recall at the healthy-calibrated operating threshold.

## Fault-specific residual detector

| Fault | Precision | Recall | F1 | FPR | Detection delay |
|---|---:|---:|---:|---:|---:|
| gyro bias | 1.000 | 1.000 | 1.000 | 0.0000 | 0.000 s |
| temperature drift | 0.905 | 0.851 | 0.877 | 0.0082 | 2.988 s |
| pressure stuck | 1.000 | 0.941 | 0.969 | 0.0000 | 0.892 s |
| voltage dropout | 1.000 | 1.000 | 1.000 | 0.0000 | 0.000 s |
| vibration spike | 1.000 | 0.923 | 0.960 | 0.0000 | 0.000 s |
| vibration noise | 0.962 | 0.891 | 0.925 | 0.0024 | 0.516 s |
| RPM gain | 1.000 | 1.000 | 1.000 | 0.0000 | 0.000 s |
| 90 Hz interference | 0.952 | 0.986 | 0.968 | 0.0033 | 0.216 s |
| bearing degradation | 0.930 | 0.703 | 0.801 | 0.0048 | 5.932 s |

Slow degradation is intentionally the hardest residual fault: the amplitude ramps gradually, so lower detection delay would require a more sensitive threshold and usually a higher false-alarm rate.

## Locally verified

- all 14 pytest tests passed
- Python byte-code compilation passed for `src`, `tests` and `scripts`
- editable package build/install succeeded with the local setuptools backend
- installed console entry point was created
- end-to-end default pipeline completed
- artifact verification script passed
- both ML models serialized successfully
- nine PNG engineering figures generated

## Not executed locally

- GitHub Actions itself (workflow is supplied, but only GitHub's hosted runner can execute that workflow)
- Ruff and mypy, because those executables are not installed in this offline runtime; CI is configured to install the dev extras and run Ruff, while the repository supplies a separate `make typecheck` target for mypy
- Docker image build/run, because the local runtime has no Docker executable

See `docs/LIMITATIONS.md` for the non-claims that apply to these results.
