"""
=========================================================
Utilities Package
AI-Based Real-Time Fault Detection System

This package contains all utility modules used
throughout the project such as logging, metrics,
helper functions, etc.

Author : Aditya Patel
=========================================================
"""

# ==========================================================
# Logger
# ==========================================================

from .logger import (
    logger,
    initialize_csv,
    log_fault,
    info,
    warning,
    error,
    critical,
    startup,
    shutdown
)

# ==========================================================
# Metrics
# ==========================================================

from .metrics import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    classification_metrics,
    save_metrics
)

# ==========================================================
# Version
# ==========================================================

__version__ = "1.0.0"

__all__ = [
    "logger",
    "initialize_csv",
    "log_fault",
    "info",
    "warning",
    "error",
    "critical",
    "startup",
    "shutdown",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "classification_metrics",
    "save_metrics"
]