"""
=========================================================
Signal Filtering Module
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import numpy as np

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import medfilt
from scipy.stats import zscore

# ==========================================================
# Moving Average Filter
# ==========================================================

def moving_average(signal, window_size=5):
    """
    Smooths signal using Moving Average Filter.

    Parameters
    ----------
    signal : ndarray
    window_size : int

    Returns
    -------
    ndarray
    """

    signal = np.asarray(signal)

    if len(signal) < window_size:
        return signal

    kernel = np.ones(window_size) / window_size

    return np.convolve(signal, kernel, mode="same")


# ==========================================================
# Median Filter
# ==========================================================

def median_filter(signal, kernel_size=3):
    """
    Removes impulsive noise (spikes).
    """

    signal = np.asarray(signal)

    return medfilt(signal, kernel_size=kernel_size)


# ==========================================================
# Butterworth Low-Pass Filter
# ==========================================================

def butter_lowpass_filter(
        signal,
        cutoff=10,
        fs=100,
        order=4
):

    """
    Removes high-frequency electrical noise.

    cutoff : Cutoff frequency (Hz)
    fs     : Sampling frequency (Hz)
    """

    signal = np.asarray(signal)

    nyquist = 0.5 * fs

    normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype="low"
    )

    return filtfilt(b, a, signal)


# ==========================================================
# Outlier Removal
# ==========================================================

def remove_outliers(signal, threshold=3):
    """
    Replaces extreme values using Z-score.
    """

    signal = np.asarray(signal)

    filtered = signal.copy()

    z = np.abs(zscore(filtered))

    mean = np.mean(filtered)

    filtered[z > threshold] = mean

    return filtered


# ==========================================================
# Min-Max Normalization
# ==========================================================

def normalize(signal):
    """
    Normalizes signal between 0 and 1.
    """

    signal = np.asarray(signal)

    minimum = np.min(signal)

    maximum = np.max(signal)

    if maximum == minimum:
        return signal

    return (signal - minimum) / (maximum - minimum)


# ==========================================================
# Standardization
# ==========================================================

def standardize(signal):
    """
    Standardizes signal.

    Mean = 0
    Std = 1
    """

    signal = np.asarray(signal)

    mean = np.mean(signal)

    std = np.std(signal)

    if std == 0:
        return signal

    return (signal - mean) / std


# ==========================================================
# Complete Filtering Pipeline
# ==========================================================

def preprocess_signal(
        signal,
        moving_avg=True,
        median=True,
        butterworth=True,
        outlier=True
):

    """
    Complete preprocessing pipeline.
    """

    signal = np.asarray(signal)

    if outlier:
        signal = remove_outliers(signal)

    if median:
        signal = median_filter(signal)

    if butterworth:
        signal = butter_lowpass_filter(signal)

    if moving_avg:
        signal = moving_average(signal)

    return signal


# ==========================================================
# Filter Three-Phase Signals
# ==========================================================

def preprocess_three_phase(
        va,
        vb,
        vc,
        ia,
        ib,
        ic
):

    """
    Preprocesses all six signals.
    """

    va = preprocess_signal(va)
    vb = preprocess_signal(vb)
    vc = preprocess_signal(vc)

    ia = preprocess_signal(ia)
    ib = preprocess_signal(ib)
    ic = preprocess_signal(ic)

    return (
        va,
        vb,
        vc,
        ia,
        ib,
        ic
    )