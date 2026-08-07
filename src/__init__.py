"""
=========================================================
AI-Based Real-Time Fault Detection System
=========================================================

A modular AI-based protection system for transmission
line fault detection, classification, fault location,
and relay operation.

Project:
    Design and Implementation of an AI-Based Real-Time
    Fault Detection, Classification, and Protection
    System for Power Transmission Lines

Author:
    Aditya Patel

Technology Stack:
    • Python
    • Scikit-Learn
    • NumPy
    • Pandas
    • Streamlit
    • Joblib

Modules:
    • Acquisition
    • Preprocessing
    • Machine Learning
    • Protection
    • Dashboard
    • Utilities

=========================================================
"""

# ==========================================================
# Version Information
# ==========================================================

__title__ = "AI Fault Detection System"

__version__ = "1.0.0"

__author__ = "Aditya Patel"

__license__ = "MIT"

# ==========================================================
# Configuration
# ==========================================================

from config.config import *

# ==========================================================
# Preprocessing
# ==========================================================

from .preprocessing import *

# ==========================================================
# Utilities
# ==========================================================

from .utils import *

# ==========================================================
# Models
# ==========================================================

from .models import *

# ==========================================================
# Protection
# ==========================================================

from .protection import *

# ==========================================================
# Dashboard
# ==========================================================

from .dashboard import *

# ==========================================================
# Acquisition
# ==========================================================

from .acquisition import *

# ==========================================================
# Public API
# ==========================================================

__all__ = [

    # Metadata
    "__title__",
    "__version__",
    "__author__",

]