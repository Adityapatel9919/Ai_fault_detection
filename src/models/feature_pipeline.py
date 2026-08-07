"""
=========================================================
Feature Pipeline
AI-Based Real-Time Fault Detection System

Author : Aditya Patel

Combines preprocessing, feature engineering and
sequence component extraction into a single pipeline.
=========================================================
"""

import pandas as pd

from src.preprocessing.feature_engineering import (
    build_feature_dataframe
)

from src.preprocessing.sequence_components import (
    extract_sequence_features
)

from src.utils.logger import info

from config.config import (
    PROCESSED_DATA_DIR
)

# ==========================================================
# Sequence Feature Extraction
# ==========================================================

def build_sequence_dataframe(df: pd.DataFrame):

    """
    Generate sequence component features
    for every sample.
    """

    sequence_features = []

    for _, row in df.iterrows():

        features = extract_sequence_features(

            row["Va"],
            row["Vb"],
            row["Vc"],

            row["Ia"],
            row["Ib"],
            row["Ic"]

        )

        sequence_features.append(features)

    return pd.DataFrame(sequence_features)


# ==========================================================
# Complete Feature Dataset
# ==========================================================

def build_training_dataset(df: pd.DataFrame):

    """
    Returns complete ML-ready dataset.
    """

    info("Generating engineered features...")

    engineered = build_feature_dataframe(df)

    info("Generating sequence component features...")

    sequence = build_sequence_dataframe(df)

    dataset = pd.concat(

        [

            engineered,

            sequence

        ],

        axis=1

    )

    # Preserve target label if available

    if "Fault" in df.columns:

        dataset["Fault"] = df["Fault"]

    info("Feature engineering completed.")

    return dataset


# ==========================================================
# Save Dataset
# ==========================================================

def save_training_dataset(

    dataset,

    filename="training_dataset.csv"

):

    PROCESSED_DATA_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    path = PROCESSED_DATA_DIR / filename

    dataset.to_csv(

        path,

        index=False

    )

    info(f"Training dataset saved : {path}")


# ==========================================================
# Pipeline
# ==========================================================

def run_feature_pipeline(df):

    dataset = build_training_dataset(df)

    save_training_dataset(dataset)

    return dataset


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    from src.models.data_loader import prepare_dataset
    from src.models.label_generator import process_labels

    data = prepare_dataset()

    data = process_labels(data)

    final_dataset = run_feature_pipeline(data)

    print()

    print("="*60)

    print("FINAL DATASET")

    print("="*60)

    print(final_dataset.head())

    print()

    print(final_dataset.shape)