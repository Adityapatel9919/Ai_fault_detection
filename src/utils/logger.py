"""
=========================================================
Logger Module
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import csv
import logging
from pathlib import Path
from datetime import datetime

from config.config import LOG_DIR, LOG_FILE

# ==========================================================
# Create Log Directory
# ==========================================================

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Python Logger Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ==========================================================
# Create CSV File (if not exists)
# ==========================================================

CSV_HEADER = [
    "Timestamp",
    "Va (V)",
    "Vb (V)",
    "Vc (V)",
    "Ia (A)",
    "Ib (A)",
    "Ic (A)",
    "Fault Type",
    "Confidence",
    "Distance (km)",
    "Relay Status"
]


def initialize_csv():

    """
    Creates fault_log.csv if it doesn't exist.
    """

    if not LOG_FILE.exists():

        with open(LOG_FILE, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow(CSV_HEADER)

        logger.info("Fault log created.")


# ==========================================================
# Log Fault Event
# ==========================================================

def log_fault(
    va,
    vb,
    vc,
    ia,
    ib,
    ic,
    fault_type,
    confidence,
    distance,
    relay_status
):

    """
    Logs every prediction into fault_log.csv
    """

    initialize_csv()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            round(va, 3),
            round(vb, 3),
            round(vc, 3),
            round(ia, 3),
            round(ib, 3),
            round(ic, 3),
            fault_type,
            round(confidence, 4),
            distance,
            relay_status
        ])

    logger.info(
        f"{fault_type} | "
        f"Confidence={confidence:.2f} | "
        f"Distance={distance} km | "
        f"Relay={relay_status}"
    )


# ==========================================================
# System Messages
# ==========================================================

def info(message):

    logger.info(message)


def warning(message):

    logger.warning(message)


def error(message):

    logger.error(message)


def critical(message):

    logger.critical(message)


# ==========================================================
# Startup Banner
# ==========================================================

def startup():

    logger.info("=" * 60)
    logger.info("AI-Based Transmission Line Protection System")
    logger.info("Support Vector Machine + Impedance Method")
    logger.info("System Initialized Successfully")
    logger.info("=" * 60)


# ==========================================================
# Shutdown Banner
# ==========================================================

def shutdown():

    logger.info("=" * 60)
    logger.info("System Shutdown")
    logger.info("=" * 60)