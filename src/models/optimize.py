"""
=========================================================
Model Optimization
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import joblib

from sklearn.model_selection import (
    GridSearchCV,
    train_test_split
)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config.config import (
    MODEL_FILE,
    SCALER_FILE,
    TEST_SIZE,
    RANDOM_STATE
)

from src.models.data_loader import prepare_dataset
from src.models.label_generator import process_labels
from src.models.feature_pipeline import run_feature_pipeline

from src.utils.logger import info
from src.utils.metrics import evaluate_model


# ==========================================================
# Parameter Grid
# ==========================================================

PARAM_GRID = {

    "C": [0.1, 1, 10, 50, 100],

    "kernel": [

        "linear",

        "rbf"

    ],

    "gamma": [

        "scale",

        "auto",

        0.1,

        0.01,

        0.001

    ]

}


# ==========================================================
# Optimize SVM
# ==========================================================

def optimize_model():

    info("Loading dataset...")

    df = prepare_dataset()

    info("Generating labels...")

    df = process_labels(df)

    info("Generating features...")

    dataset = run_feature_pipeline(df)

    X = dataset.drop(columns=["Fault"])

    y = dataset["Fault"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    svm = SVC(

        probability=True,

        random_state=RANDOM_STATE

    )

    grid = GridSearchCV(

        estimator=svm,

        param_grid=PARAM_GRID,

        scoring="accuracy",

        cv=5,

        n_jobs=-1,

        verbose=2

    )

    info("Optimizing SVM...")

    grid.fit(

        X_train,

        y_train

    )

    best_model = grid.best_estimator_

    print()

    print("=" * 60)

    print("BEST PARAMETERS")

    print("=" * 60)

    print(grid.best_params_)

    print()

    print("Best CV Accuracy :", round(grid.best_score_, 4))

    print()

    y_pred = best_model.predict(

        X_test

    )

    evaluate_model(

        best_model,

        X_test,

        y_test,

        y_pred

    )

    MODEL_FILE.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    joblib.dump(

        best_model,

        MODEL_FILE

    )

    joblib.dump(

        scaler,

        SCALER_FILE

    )

    info("Optimized model saved.")

    info("Scaler saved.")

    return best_model


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    optimize_model()