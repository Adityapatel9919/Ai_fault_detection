"""
=========================================================
Dataset Loader
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import pandas as pd

from pathlib import Path

from config.config import (
    DATASET_PATH,
    PROCESSED_DATA_DIR
)

from src.utils import info

# ==========================================================
# Required Dataset Columns
# ==========================================================

REQUIRED_COLUMNS = [

    "Va",
    "Vb",
    "Vc",

    "Ia",
    "Ib",
    "Ic",

    "G",
    "C",
    "B",
    "A"

]

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(path=DATASET_PATH):

    """
    Loads CSV dataset.
    """

    if not Path(path).exists():

        raise FileNotFoundError(

            f"Dataset not found : {path}"

        )

    df = pd.read_csv(path)

    info(f"Dataset Loaded : {path}")

    return df


# ==========================================================
# Validate Dataset
# ==========================================================

def validate_dataset(df):

    """
    Checks whether all required
    columns are available.
    """

    missing = [

        col

        for col in REQUIRED_COLUMNS

        if col not in df.columns

    ]

    if missing:

        raise ValueError(

            f"Missing Columns : {missing}"

        )

    info("Dataset Validation Successful")


# ==========================================================
# Remove Missing Values
# ==========================================================

def remove_missing(df):

    before = len(df)

    df = df.dropna()

    after = len(df)

    info(

        f"Removed {before-after} rows containing missing values."

    )

    return df


# ==========================================================
# Remove Duplicate Rows
# ==========================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    info(

        f"Removed {before-after} duplicate rows."

    )

    return df


# ==========================================================
# Dataset Information
# ==========================================================

def dataset_summary(df):

    print()

    print("="*60)

    print("DATASET SUMMARY")

    print("="*60)

    print()

    print(df.info())

    print()

    print(df.describe())

    print()

    print("Shape :", df.shape)

    print()

    print("Columns")

    for col in df.columns:

        print(f" • {col}")


# ==========================================================
# Save Clean Dataset
# ==========================================================

def save_dataset(

    df,

    filename="clean_dataset.csv"

):

    PROCESSED_DATA_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    path = PROCESSED_DATA_DIR / filename

    df.to_csv(

        path,

        index=False

    )

    info(f"Dataset Saved : {path}")


# ==========================================================
# Complete Pipeline
# ==========================================================

def prepare_dataset():

    """
    Complete preprocessing pipeline.
    """

    df = load_dataset()

    validate_dataset(df)

    df = remove_missing(df)

    df = remove_duplicates(df)

    dataset_summary(df)

    save_dataset(df)

    return df


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    prepare_dataset()