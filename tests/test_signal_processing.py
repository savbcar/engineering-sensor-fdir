import numpy as np

from sensor_fdir.signal_processing import (
    alias_frequency,
    butter_filter,
    fft_spectrum,
    nyquist_frequency,
)


def test_nyquist_and_alias():
    assert nyquist_frequency(500.0) == 250.0
    assert alias_frequency(70.0, 100.0) == 30.0


def test_fft_finds_tone():
    fs = 500.0
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 50.0 * t)
    f, a = fft_spectrum(x, fs)
    assert abs(f[np.argmax(a)] - 50.0) <= 0.5


def test_lowpass_rejects_high_frequency():
    fs = 500.0
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 120 * t)
    y = butter_filter(x, fs, 40.0, order=4, btype="lowpass")
    f, a = fft_spectrum(y, fs)
    a10 = a[np.argmin(np.abs(f - 10))]
    a120 = a[np.argmin(np.abs(f - 120))]
    assert a10 > 10 * a120
