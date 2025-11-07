"""
EEG Signal Preprocessing Utilities
-----------------------------------
Preprocessing pipeline for raw EEG signals from SAM-40 dataset.
Includes baseline drift removal and wavelet-based denoising.

Authors: Prakriti Sharma, Aaditya Upadhyay, Dr. Akshay Jadhav
Manipal University Jaipur
"""

import numpy as np
from scipy.signal import savgol_filter


def remove_baseline_drift(signal, window_length=51, polyorder=3):
    """
    Remove low-frequency baseline wander using a Savitzky-Golay filter.

    Parameters
    ----------
    signal : np.ndarray
        Raw 1D EEG channel signal.
    window_length : int
        Length of the filter window (must be odd positive integer).
    polyorder : int
        Order of polynomial to fit within window.

    Returns
    -------
    np.ndarray
        Signal with baseline drift removed.
    """
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= len(signal):
        window_length = len(signal) if len(signal) % 2 != 0 else len(signal) - 1
    baseline = savgol_filter(signal, window_length, polyorder)
    return signal - baseline
