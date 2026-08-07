"""
=========================================================
Protection Package
AI-Based Real-Time Fault Detection System

This package contains all modules responsible for
power system protection, relay decision making,
fault location estimation, and protection management.

Author : Aditya Patel
=========================================================
"""

# ==========================================================
# Fault Locator
# ==========================================================

from .fault_locator import (
    calculate_fault_impedance,
    calculate_fault_distance,
    classify_protection_zone
)

# ==========================================================
# Relay Logic
# ==========================================================

from .relay_logic import (
    ProtectionRelay,
    RelayState
)

# ==========================================================
# Fault Classifier
# ==========================================================

from .fault_classifier import (
    FaultClassifier
)

# ==========================================================
# Protection Manager
# ==========================================================

from .protection_manager import (
    ProtectionManager
)

# ==========================================================
# Version
# ==========================================================

__version__ = "1.0.0"

# ==========================================================
# Public API
# ==========================================================

__all__ = [

    # Relay
    "ProtectionRelay",
    "RelayState",

    # Fault Location
    "calculate_fault_impedance",
    "calculate_fault_distance",
    "classify_protection_zone",

    # Classifier
    "FaultClassifier",

    # Manager
    "ProtectionManager"
]