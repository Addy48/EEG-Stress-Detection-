import numpy as np
import pytest
from src.preprocessing import remove_baseline_drift, wavelet_denoise, preprocess_channel, preprocess_trial

def test_remove_baseline_drift():
    t = np.linspace(0, 4 * np.pi, 256)
    clean = np.sin(t)
    drift = 0.5 * t
    noisy = clean + drift
    
    filtered = remove_baseline_drift(noisy, window_length=51, polyorder=3)
    assert filtered.shape == clean.shape
    # Baseline drift slope should be substantially reduced
    assert np.abs(np.polyfit(t, filtered, 1)[0]) < np.abs(np.polyfit(t, noisy, 1)[0])

def test_wavelet_denoise():
    np.random.seed(42)
    t = np.linspace(0, 1, 256)
    signal = np.sin(2 * np.pi * 10 * t)
    noise = 0.1 * np.random.randn(len(t))
    noisy = signal + noise
    
    denoised = wavelet_denoise(noisy, wavelet='db2', level=3)
    assert len(denoised) == len(signal)
    # Signal reconstruction fidelity
    corr = np.corrcoef(denoised, signal)[0, 1]
    assert corr > 0.85

def test_preprocess_trial():
    n_channels = 14
    n_samples = 256
    trial = np.random.randn(n_channels, n_samples)
    preprocessed = preprocess_trial(trial, fs=128)
    assert preprocessed.shape == (n_channels, n_samples)
