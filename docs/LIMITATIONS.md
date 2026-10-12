# Limitations and Non-Claims

1. The main validation dataset is synthetic and generated from the same broad modelling assumptions used by the residual detector. Results therefore demonstrate software correctness and controlled diagnostic behavior, not field accuracy.
2. Faults are injected one family at a time in mostly separated intervals. Real systems may experience interacting or simultaneous faults.
3. Model residuals benefit from access to nominal model channels. Production observers would have their own uncertainty and modelling error.
4. Offline zero-phase filtering is non-causal and cannot be used unchanged for real-time recovery logic.
5. The Isolation Forest and autoencoder are unsupervised anomaly detectors. Their alarms do not by themselves prove fault type or root cause.
6. The autoencoder operates on engineered windows rather than raw temporal sequences; it is not a sequence model.
7. Thresholds are selected from a healthy calibration interval and are not formally optimized for safety risk, maintenance cost or certification objectives.
8. Recovery/reconfiguration is outside the implemented scope. The project stops at detection plus partial isolation/identification.
9. No claim is made about compliance with ISO 26262, DO-178C, DO-254, ECSS, IEC 61508 or any certification standard.
10. Docker configuration is supplied for reproducibility, but Docker execution depends on the host environment.
