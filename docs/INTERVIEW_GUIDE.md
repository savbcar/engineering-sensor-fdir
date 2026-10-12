# Interview Defence Guide

## Explain FFT in one minute

The DFT decomposes a finite discrete signal into complex sinusoidal basis components. Each bin measures correlation with a particular discrete frequency. The FFT is not a different transform; it is an efficient family of algorithms for computing the DFT in roughly `O(N log N)` instead of `O(N^2)`. I window the signal before the FFT because finite acquisition causes leakage; a Hann window reduces sidelobes at the cost of wider peaks.

## Explain Nyquist and aliasing

Sampling maps continuous frequencies onto discrete-time frequency modulo the sampling rate. Once a component exceeds half the sampling rate, another lower frequency can produce indistinguishable samples. That is aliasing. A real acquisition chain therefore needs both an adequate sampling rate and an analogue anti-alias filter before the ADC.

## Explain a Butterworth filter

A Butterworth filter is maximally flat in the passband. Its magnitude reaches -3.01 dB at the cutoff for the standard low-pass prototype. I used SOS form for numerical robustness. For offline plots I used forward/backward filtering, but a deployed controller would require causal filtering and delay analysis.

## Explain residual-based FDI

Instead of asking whether a raw value is unusual, I compare measurements against physically consistent expectations. For example, shaft RPM predicts angular velocity through `omega = rpm*2*pi/60`. A persistent discrepancy is a residual. Healthy residual statistics provide a noise-aware threshold. With independent references, residual patterns also help isolate which sensor is inconsistent.

## Why might simple methods beat ML?

If a failure has a direct physical signature—NaN dropout, impossible range, frozen variance, or a redundant-sensor mismatch—a deterministic rule is data-efficient, explainable, cheap and easy to validate. ML becomes useful when the healthy envelope is multivariate and nonlinear or the anomaly signature is diffuse. Complexity should follow evidence, not fashion.

## What creates false positives?

Operating-point changes, model mismatch, unmodelled disturbances, sensor noise, transients, threshold selection and domain shift. False-positive cost matters: too many alarms produce nuisance maintenance or operators ignoring the system.

## How does this transfer to automotive, Airbus or DLR work?

The specific sensors change, but the workflow is the same: define the physical system and failure hypotheses; understand acquisition/sampling; condition signals; derive interpretable features and residuals; detect anomalies; isolate sources using redundancy or models; validate against representative operating envelopes; quantify missed detections, false alarms and detection delay; then define safe reconfiguration or maintenance action.

## What would you improve for deployment?

Use real sensor datasets and hardware logs, causal filters, timestamp/jitter handling, sensor calibration/units metadata, train/validation splits by mission or drive cycle, uncertainty-aware thresholds, fault severity estimation, multi-fault scenarios, online state observers, concept-drift monitoring, reproducible model cards, and hardware-in-the-loop testing.
