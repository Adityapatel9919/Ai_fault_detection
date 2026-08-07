"""
=========================================================
Configuration File
AI-Based Real-Time Fault Detection System
=========================================================

Author : Aditya Patel
Project : AI-Based Transmission Line Protection System
Algorithm : Support Vector Machine (SVM)
=========================================================
"""

from pathlib import Path

# =========================================================
# PROJECT PATHS
# =========================================================

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# Log directory
LOG_DIR = ROOT_DIR / "logs"

# Report directory
REPORT_DIR = ROOT_DIR / "reports"

# Notebook directory
NOTEBOOK_DIR = ROOT_DIR / "notebooks"

# =========================================================
# DATASET
# =========================================================

DATASET_NAME = "fault_dataset.csv"

DATASET_PATH = RAW_DATA_DIR / DATASET_NAME

# =========================================================
# MODEL FILES
# =========================================================

MODEL_FILE = MODEL_DIR / "svm_fault_detection_model.pkl"

SCALER_FILE = MODEL_DIR / "scaler.pkl"

# =========================================================
# TRAINING PARAMETERS
# =========================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

CROSS_VALIDATION = 10

# =========================================================
# SVM PARAMETERS
# =========================================================

SVM_KERNEL = "rbf"

SVM_C = 10

SVM_GAMMA = "scale"

ENABLE_GRID_SEARCH = True

# =========================================================
# RELAY PARAMETERS
# =========================================================

CONFIDENCE_THRESHOLD = 0.90

MAJORITY_WINDOW = 5

MIN_FAULT_COUNT = 4

# =========================================================
# TRANSMISSION LINE PARAMETERS
# =========================================================

LINE_VOLTAGE = 400          # Volts

PHASE_VOLTAGE = 230         # Volts

LINE_FREQUENCY = 50         # Hz

# Positive sequence impedance (Example)
LINE_IMPEDANCE_PER_KM = complex(0.12, 0.38)

# =========================================================
# ADC PARAMETERS
# =========================================================

ADC_RESOLUTION = 1023

ADC_REFERENCE = 5.0

# =========================================================
# SERIAL COMMUNICATION
# =========================================================

SERIAL_PORT = "COM3"

BAUD_RATE = 115200

SERIAL_TIMEOUT = 1

# =========================================================
# DASHBOARD
# =========================================================

DASHBOARD_REFRESH = 1

# seconds

# =========================================================
# LOGGING
# =========================================================

LOG_FILE = LOG_DIR / "fault_log.csv"

ENABLE_LOGGING = True

# =========================================================
# FEATURE FLAGS
# =========================================================

USE_IMPEDANCE = True

USE_SEQUENCE_COMPONENTS = True

USE_POWER_FEATURES = True

USE_VOLTAGE_IMBALANCE = True

USE_CURRENT_IMBALANCE = True

USE_MOVING_AVERAGE_FILTER = True

USE_CONFIDENCE_CHECK = True

USE_MAJORITY_VOTING = True

# =========================================================
# LABEL MAPPING
# =========================================================

FAULT_LABELS = {
    0: "Normal",
    1: "Line-to-Ground (LG)",
    2: "Line-to-Line (LL)",
    3: "Double Line-to-Ground (LLG)",
    4: "Three-Phase Fault (LLL)"
}