"""
=========================================================
Preprocessing Package
AI-Based Real-Time Fault Detection System

This package contains all signal preprocessing,
feature engineering and sequence component modules.
=========================================================
"""

# ==========================================================
# Signal Filters
# ==========================================================

from .filters import (
    moving_average,
    median_filter,
    butter_lowpass_filter,
    remove_outliers,
    normalize,
    standardize,
    preprocess_signal,
    preprocess_three_phase
)

# ==========================================================
# Feature Engineering
# ==========================================================

from .feature_engineering import (
    average_voltage,
    average_current,
    rms,
    phase_impedance,
    average_impedance,
    voltage_imbalance,
    current_imbalance,
    voltage_difference,
    current_difference,
    apparent_power,
    active_power,
    reactive_power,
    power_factor,
    rate_of_change,
    extract_features,
    build_feature_dataframe
)

# ==========================================================
# Sequence Components
# ==========================================================

from .sequence_components import (
    voltage_sequence_components,
    current_sequence_components,
    sequence_magnitudes,
    sequence_ratios,
    fault_indices,
    extract_sequence_features
)

# ==========================================================
# Version
# ==========================================================

__version__ = "1.0.0"