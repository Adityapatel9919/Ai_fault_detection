"""
=========================================================
Model Training
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config.config import (
    MODEL_FILE,
    SCALER_FILE,
    TEST_SIZE,
    RANDOM_STATE,
    SVM_KERNEL,
    SVM_C,
    SVM_GAMMA
)

from src.models.data_loader import prepare_dataset
from src.models.label_generator import process_labels
from src.models.feature_pipeline import run_feature_pipeline

from src.utils.metrics import evaluate_model
from src.utils.logger import info

# ==========================================================
# Training Pipeline
# ==========================================================

def train_model():

    info("Loading dataset...")

    df = prepare_dataset()

    info("Generating labels...")

    df = process_labels(df)

    info("Building feature dataset...")

    dataset = run_feature_pipeline(df)

    # ---------------------------------------
    # Features & Labels
    # ---------------------------------------

    X = dataset.drop(columns=["Fault"])

    y = dataset["Fault"]

    # ---------------------------------------
    # Train/Test Split
    # ---------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    # ---------------------------------------
    # Scaling
    # ---------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # ---------------------------------------
    # Model
    # ---------------------------------------

    model = SVC(

        kernel=SVM_KERNEL,

        C=SVM_C,

        gamma=SVM_GAMMA,

        probability=True,

        random_state=RANDOM_STATE

    )

    info("Training model...")

    model.fit(

        X_train,

        y_train

    )

    # ---------------------------------------
    # Prediction
    # ---------------------------------------

    y_pred = model.predict(

        X_test

    )

    # ---------------------------------------
    # Evaluation
    # ---------------------------------------

    evaluate_model(

        model,

        X_test,

        y_test,

        y_pred

    )

    # ---------------------------------------
    # Save Model
    # ---------------------------------------

    MODEL_FILE.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    joblib.dump(

        model,

        MODEL_FILE

    )

    joblib.dump(

        scaler,

        SCALER_FILE

    )

    info("Model Saved.")

    info("Scaler Saved.")

    return model, scaler


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train_model()