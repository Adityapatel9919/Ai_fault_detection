"""
Dataset Loader
AI-Based Real-Time Fault Detection System

Loads the final machine learning dataset generated
from the PROTECT-90 Feature Extraction Engine.

Author : Aditya Patel
"""

from pathlib import Path
import pandas as pd

from config.config import DATASET_PATH, PROCESSED_DATA_DIR
from src.utils import info

# ==========================================================
# Target Columns
# ==========================================================

TARGET_COLUMNS = [
    "sc_type",
    "fault_target",
    "phase_select",
]

# ==========================================================
# Columns not used for training
# ==========================================================

DROP_COLUMNS = [
    "episode_id",
]

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load ML-ready dataset.
    Supports Parquet and CSV.
    """

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)

    elif str(path).endswith(".csv"):
        df = pd.read_csv(path)

    else:
        raise ValueError(
            "Unsupported dataset format. Use .parquet or .csv"
        )

    info(f"Dataset Loaded : {path}")

    return df

# ==========================================================
# Validate Dataset
# ==========================================================

def validate_dataset(df: pd.DataFrame):

    missing = [
        col
        for col in TARGET_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing target columns: {missing}"
        )

    info("Dataset Validation Successful")

# ==========================================================
# Remove Missing Values
# ==========================================================

def remove_missing(df: pd.DataFrame):

    before = len(df)

    df = df.dropna()

    info(
        f"Removed {before-len(df)} rows containing missing values."
    )

    return df

# ==========================================================
# Remove Duplicate Rows
# ==========================================================

def remove_duplicates(df: pd.DataFrame):

    before = len(df)

    df = df.drop_duplicates()

    info(
        f"Removed {before-len(df)} duplicate rows."
    )

    return df

# ==========================================================
# Prepare Features
# ==========================================================

def prepare_features(df: pd.DataFrame):

    X = df.drop(
        columns=DROP_COLUMNS + TARGET_COLUMNS,
        errors="ignore",
    )

    return X

# ==========================================================
# Prepare Targets
# ==========================================================

def prepare_targets(df: pd.DataFrame):

    y_sc_type = df["sc_type"]

    y_fault_target = df["fault_target"]

    y_phase = df["phase_select"]

    return y_sc_type, y_fault_target, y_phase

# ==========================================================
# Dataset Summary
# ==========================================================

def dataset_summary(df):

    print("=" * 60)

    print("DATASET SUMMARY")

    print("=" * 60)

    print(f"Rows     : {len(df)}")

    print(f"Columns  : {len(df.columns)}")

    print()

    print(df.dtypes.value_counts())

# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(
    df: pd.DataFrame,
    filename="protect90_features_clean.parquet",
):

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = PROCESSED_DATA_DIR / filename

    df.to_parquet(
        path,
        index=False,
    )

    info(f"Dataset Saved : {path}")

# ==========================================================
# Complete Pipeline
# ==========================================================

def prepare_dataset():

    df = load_dataset()

    validate_dataset(df)

    df = remove_missing(df)

    df = remove_duplicates(df)

    dataset_summary(df)

    X = prepare_features(df)

    y_sc_type, y_fault_target, y_phase = prepare_targets(df)

    return (
        X,
        y_sc_type,
        y_fault_target,
        y_phase,
    )

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    X, y1, y2, y3 = prepare_dataset()

    print()

    print("Feature Matrix :", X.shape)

    print("Fault Type Labels :", y1.shape)

    print("Fault Location Labels :", y2.shape)

    print("Phase Labels :", y3.shape)