"""
=========================================================
Prediction Module
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import joblib
import pandas as pd

from config.config import (
    MODEL_FILE,
    SCALER_FILE
)

from src.preprocessing.feature_engineering import extract_features
from src.preprocessing.sequence_components import extract_sequence_features

# ==========================================================
# Load Model
# ==========================================================

def load_model():
    global model, scaler

    if model is None:
        model = joblib.load(MODEL_FILE)

    if scaler is None:
        scaler = joblib.load(SCALER_FILE)


# ==========================================================
# Prepare Input Features
# ==========================================================

def prepare_input(
    va,
    vb,
    vc,
    ia,
    ib,
    ic
):
    """
    Converts one measurement into the
    feature vector expected by the model.
    """

    row = {

        "Va": va,
        "Vb": vb,
        "Vc": vc,

        "Ia": ia,
        "Ib": ib,
        "Ic": ic

    }

    row = pd.Series(row)

    engineered = extract_features(row)

    sequence = extract_sequence_features(

        va,
        vb,
        vc,

        ia,
        ib,
        ic

    )

    features = {

        **engineered,

        **sequence

    }

    X = pd.DataFrame([features])

    X = scaler.transform(X)

    return X

# ==========================================================
# Prediction
# ==========================================================

def predict_fault(

    va,
    vb,
    vc,

    ia,
    ib,
    ic

):

    X = prepare_input(

        va,
        vb,
        vc,

        ia,
        ib,
        ic

    )

    prediction = model.predict(X)[0]

    confidence = model.predict_proba(X)[0].max()

    return {

        "fault": prediction,

        "confidence": round(confidence,4)

    }

# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    result = predict_fault(

        230,

        229,

        231,

        5.1,

        5.0,

        5.2

    )

    print()

    print("="*60)

    print(result)

    print("="*60)