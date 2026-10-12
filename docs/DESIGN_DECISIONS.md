# Design Decisions

## Why synthetic physics-informed data first?

Ground truth for fault onset, type and magnitude is exact, allowing controlled validation. Real engineering data should be added later to test domain shift, sensor quirks and unknown fault mechanisms. Synthetic-only performance is not evidence of field reliability.

## Why preserve simple detectors?

A strong portfolio project should show that the author can choose the simplest method that satisfies a requirement. Dropout, stuck-sensor and hard-consistency faults are often better handled with deterministic logic than with opaque ML. ML is added only for multivariate anomalies that are difficult to define with one threshold.

## Why an operating model?

Raw sensor distributions change during acceleration, load changes and temperature transients. Physics residuals reduce dependence on operating point by comparing a measurement with an expected value conditioned on known system behavior.

## Why separate sample and window evaluation?

Classical detectors operate at sample rate while feature/ML detectors operate on overlapping windows. Directly comparing those outputs would be misleading. The repository therefore retains sample-level fault-specific metrics for residual diagnostics and aggregates all detector families to the same window grid for the common comparison table.

## Why no LSTM/TCN in v1.0?

The project requirement explicitly calls for progressive complexity. A small autoencoder already tests whether nonlinear multivariate representation learning adds value. A temporal network would be justified only after demonstrating a sequence-specific failure mode, collecting more realistic data, and defining a fair temporal validation split.

## Why offline zero-phase filtering?

The plots and feature extraction are offline analysis. `sosfiltfilt` removes phase distortion and makes diagnostic interpretation cleaner. It is not deployable in a causal real-time loop. This limitation is explicitly documented rather than hidden.

## Why train unsupervised ML on a separate healthy reference run?

A short calibration segment may not cover later healthy operating points. That can make operating-point changes look more anomalous than faults. The benchmark therefore trains Isolation Forest and the autoencoder on a separate fault-free run spanning the same commanded operating envelope. Fault labels are not used for fitting. This is representative of commissioning with known-good reference data, while also making the synthetic benchmark limitation explicit.
