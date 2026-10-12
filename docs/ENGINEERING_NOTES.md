# Engineering Notes

## System boundary

The platform models an instrumented electromechanical rotor with a DC bus, rotating shaft, bearings/structure and a cooling loop. The sensor suite contains rotational speed, angular rate, temperature, pressure, voltage and two vibration channels. The synthetic generator is intentionally physics-informed rather than a collection of unrelated waveforms: angular rate is linked to shaft speed, thermal response is first order, pressure depends on speed/load, bus voltage sags with load, and vibration contains shaft-order harmonics plus structural modes.

## Sampling, Nyquist and aliasing

For sampling period `T_s`, the sampling frequency is `f_s = 1/T_s`. A sinusoid above `f_s/2` cannot be uniquely represented. Frequencies separated by integer multiples of `f_s` produce the same discrete samples, so the observable baseband alias is

`f_alias = |((f + f_s/2) mod f_s) - f_s/2|`.

The default run uses 250 Hz, giving a Nyquist frequency of 125 Hz. This is sufficient for the injected 90 Hz interference and the default structural content below 125 Hz. A production system would choose sample rates from sensor bandwidth, anti-alias-filter design, transient requirements and timing constraints rather than from this demo.

## DFT, FFT and windowing

The DFT is

`X[k] = sum_{n=0}^{N-1} x[n] exp(-j 2 pi k n / N)`.

The FFT is an efficient algorithm for computing that same transform, reducing computational complexity from approximately `O(N^2)` to `O(N log N)`. A Hann window is used in spectral analysis to reduce sidelobe leakage caused by finite observation windows. The trade-off is a wider main lobe and therefore reduced ability to separate very closely spaced tones.

Frequency resolution is `Delta f = f_s/N = 1/T_obs`. Raising the sample rate alone does not improve frequency resolution if the observation duration is unchanged.

## Filters

The platform provides zero-phase offline Butterworth filtering through forward/backward SOS filtering. For an nth-order analogue Butterworth prototype,

`|H(j Omega)|^2 = 1 / (1 + (Omega/Omega_c)^(2n))`.

At cutoff the magnitude is `1/sqrt(2)`, or approximately -3.01 dB. Butterworth is used because of its maximally flat passband, not because it is universally optimal. Chebyshev, elliptic, Bessel, FIR and causal IIR alternatives have different amplitude, phase, delay and transition-band trade-offs.

The demo uses `sosfiltfilt`, which is **non-causal** and therefore suitable for offline analysis only. A deployed real-time implementation must use a causal filter (`sosfilt`) and explicitly budget its phase/group delay.

## Feature engineering

Each overlapping feature window contains, per sensor:

- mean
- RMS
- variance
- skewness
- excess kurtosis
- peak-to-peak range
- RMS first difference
- linear slope
- dropout fraction

For both vibration axes it additionally computes:

- dominant frequency
- spectral centroid
- band powers (0-20, 20-80, 80-180 Hz)
- shaft-order harmonic power ratio around 1x, 2x and 3x rotational frequency

The default window is 2 s with 1.0 s stride. This improves statistical stability but creates detection latency and overlapping observations; these are engineering trade-offs, not free accuracy.

## Detection ladder

### 1. Fixed threshold rules

The simplest detector uses transparent engineering limits such as dropout, excessively high vibration amplitude, low pressure variance and elevated vibration variance. It is easy to audit and certify but weak when the healthy operating envelope changes.

### 2. Statistical detection

A robust channel-wise detector learns median and scale from a known healthy calibration interval. MAD-based scaling is used to reduce sensitivity to outliers. This layer is intentionally naive to show why purely distributional methods struggle with strongly non-stationary operating conditions.

### 3. Residual-based detection

Residuals compare measurements against physically related expectations:

- `r_gyro = omega_gyro - rpm_model * 2*pi/60`
- `r_rpm = rpm_sensor - rpm_model`
- `r_temp = T_sensor - T_model`
- `r_pressure = p_sensor - p_model`
- `r_voltage = V_sensor - V_model`

Healthy residual statistics establish thresholds. This is closer to classical FDI because it uses redundancy and model consistency rather than only raw magnitude.

### 4. Isolation Forest

Isolation Forest operates on the engineered window feature vectors. It is trained only on healthy calibration windows and assigns higher anomaly scores to observations that are isolated with short random partition paths. It provides anomaly detection, not inherently fault type or root-cause isolation.

### 5. Autoencoder

A small feed-forward PyTorch autoencoder is trained only on healthy standardized feature windows. Reconstruction MSE is the anomaly score. This is intentionally not a large temporal neural network: with synthetic data and a small calibration set, a more complex LSTM/TCN would add representational capacity without proving better engineering validity.

## FDIR terminology

- **Detection:** determine whether something is wrong.
- **Isolation:** locate the affected sensor/subsystem.
- **Identification:** infer fault type and, where possible, magnitude.
- **Recovery:** take corrective action such as reconfiguration, substitution, graceful degradation or shutdown.

This repository implements detection, model-assisted isolation and rule-based identification for the injected sensor faults. It does **not** implement closed-loop recovery or safety certification.

## Metrics

Precision measures alarm trustworthiness: `TP/(TP+FP)`.

Recall measures fault coverage: `TP/(TP+FN)`.

F1 balances precision and recall.

False-positive rate is `FP/(FP+TN)` and matters because nuisance alarms can cause unnecessary maintenance, trips or alarm fatigue.

ROC-AUC is reported for continuous ML anomaly scores where both classes are present. It measures ranking quality across thresholds but does not pick an operational threshold or encode asymmetric safety costs.

Detection delay is tracked per fault-specific residual detector. A classifier can have excellent aggregate metrics and still be unacceptable if the first alarm arrives too late.
