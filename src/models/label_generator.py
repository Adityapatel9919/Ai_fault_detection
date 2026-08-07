"""
=========================================================
Label Generator
AI-Based Real-Time Fault Detection System

Author : Aditya Patel

Converts fault indicator columns (G, C, B, A)
into machine learning class labels.
=========================================================
"""

import pandas as pd

from src.utils.logger import info

# ==========================================================
# Fault Mapping
# ==========================================================

FAULT_MAPPING = {

    # Normal
    (0, 0, 0, 0): "Normal",

    # Line to Ground Faults
    (1, 0, 0, 1): "AG",
    (1, 0, 1, 0): "BG",
    (1, 1, 0, 0): "CG",

    # Line to Line Faults
    (0, 0, 1, 1): "AB",
    (0, 1, 0, 1): "AC",
    (0, 1, 1, 0): "BC",

    # Double Line to Ground Faults
    (1, 0, 1, 1): "ABG",
    (1, 1, 0, 1): "ACG",
    (1, 1, 1, 0): "BCG",

    # Three Phase Fault
    (0, 1, 1, 1): "ABC",

    # Three Phase to Ground
    (1, 1, 1, 1): "ABCG"

}

# ==========================================================
# Generate Label
# ==========================================================

def generate_label(row):
    """
    Convert one dataset row into a fault label.
    """

    key = (
        int(row["G"]),
        int(row["C"]),
        int(row["B"]),
        int(row["A"])
    )

    return FAULT_MAPPING.get(key, "Unknown")


# ==========================================================
# Apply Labels
# ==========================================================

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'Fault' column to the dataframe.
    """

    df = df.copy()

    df["Fault"] = df.apply(generate_label, axis=1)

    info("Fault labels generated successfully.")

    return df


# ==========================================================
# Show Distribution
# ==========================================================

def label_distribution(df: pd.DataFrame):

    print()

    print("=" * 60)
    print("FAULT DISTRIBUTION")
    print("=" * 60)

    print(df["Fault"].value_counts())

    print()


# ==========================================================
# Remove Unknown Labels
# ==========================================================

def remove_unknown(df: pd.DataFrame):

    before = len(df)

    df = df[df["Fault"] != "Unknown"]

    after = len(df)

    info(f"Removed {before-after} unknown samples.")

    return df


# ==========================================================
# Complete Pipeline
# ==========================================================

def process_labels(df: pd.DataFrame):

    df = generate_labels(df)

    df = remove_unknown(df)

    label_distribution(df)

    return df