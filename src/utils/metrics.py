"""
=========================================================
Model Evaluation Metrics
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.model_selection import cross_val_score

from config.config import REPORT_DIR

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Accuracy
# ==========================================================

def accuracy(y_true, y_pred):

    return accuracy_score(y_true, y_pred)


# ==========================================================
# Precision
# ==========================================================

def precision(y_true, y_pred):

    return precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


# ==========================================================
# Recall
# ==========================================================

def recall(y_true, y_pred):

    return recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


# ==========================================================
# F1 Score
# ==========================================================

def f1(y_true, y_pred):

    return f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


# ==========================================================
# Classification Metrics
# ==========================================================

def classification_metrics(y_true, y_pred):

    return {

        "Accuracy":

            accuracy(y_true, y_pred),

        "Precision":

            precision(y_true, y_pred),

        "Recall":

            recall(y_true, y_pred),

        "F1 Score":

            f1(y_true, y_pred)

    }


# ==========================================================
# Classification Report
# ==========================================================

def print_classification_report(
    y_true,
    y_pred
):

    print(

        classification_report(

            y_true,

            y_pred

        )

    )


# ==========================================================
# Confusion Matrix
# ==========================================================

def plot_confusion_matrix(

    y_true,

    y_pred,

    labels=None,

    save=True

):

    cm = confusion_matrix(

        y_true,

        y_pred

    )

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=labels

    )

    fig, ax = plt.subplots(figsize=(8,6))

    disp.plot(ax=ax)

    plt.title("Confusion Matrix")

    if save:

        plt.savefig(

            REPORT_DIR /

            "confusion_matrix.png",

            dpi=300,

            bbox_inches="tight"

        )

    plt.show()


# ==========================================================
# ROC Curve
# ==========================================================

def plot_roc_curve(

    y_true,

    y_score

):

    fpr, tpr, _ = roc_curve(

        y_true,

        y_score

    )

    roc_auc = auc(

        fpr,

        tpr

    )

    plt.figure(figsize=(8,6))

    plt.plot(

        fpr,

        tpr,

        label=f"AUC = {roc_auc:.3f}"

    )

    plt.plot(

        [0,1],

        [0,1],

        "--"

    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid(True)

    plt.savefig(

        REPORT_DIR /

        "roc_curve.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.show()


# ==========================================================
# Cross Validation
# ==========================================================

def cross_validation(

    model,

    x,

    y,

    cv=10

):

    scores = cross_val_score(

        model,

        x,

        y,

        cv=cv

    )

    return scores


# ==========================================================
# Save Metrics
# ==========================================================

def save_metrics(

    metrics,

    filename="metrics.csv"

):

    df = pd.DataFrame(

        [metrics]

    )

    df.to_csv(

        REPORT_DIR /

        filename,

        index=False

    )


# ==========================================================
# Complete Evaluation
# ==========================================================

def evaluate_model(

    model,

    x_test,

    y_test,

    y_pred

):

    metrics = classification_metrics(

        y_test,

        y_pred

    )

    print()

    print("="*60)

    print("MODEL PERFORMANCE")

    print("="*60)

    for k,v in metrics.items():

        print(

            f"{k:15}: {v:.4f}"

        )

    print()

    print_classification_report(

        y_test,

        y_pred

    )

    plot_confusion_matrix(

        y_test,

        y_pred

    )

    save_metrics(

        metrics

    )

    return metrics