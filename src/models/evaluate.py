"""
evaluate.py

Executes rigorous performance evaluation for trained machine learning models 
in the AI-Based Real-Time Fault Detection system.

Given the multi-target nature of the system (sc_type, fault_target, phase_select),
this module calculates independent classification metrics for each target 
(Precision, Recall, F1-Score, Confusion Matrices) as well as the overall 
Exact Match Ratio (System-level accuracy).

Updated to support both pandas DataFrames and NumPy ndarrays agnostically.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Initialize module logger
logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Exception raised for errors occurring during model evaluation."""
    pass


class ModelEvaluator:
    """
    Evaluates multi-target machine learning models for power system protection.
    Calculates detailed metrics per protection target and overall system accuracy.
    Supports both pandas DataFrame and NumPy ndarray inputs seamlessly.
    """

    TARGET_COLUMNS = ['sc_type', 'fault_target', 'phase_select']

    def __init__(self) -> None:
        """Initializes the ModelEvaluator."""
        pass

    def _validate_inputs(
        self, 
        model: BaseEstimator, 
        X_array: np.ndarray, 
        y_array: np.ndarray
    ) -> None:
        """
        Validates the shapes and types of internal NumPy arrays before evaluation.

        Args:
            model (BaseEstimator): The trained model to evaluate.
            X_array (np.ndarray): The feature dataset as a NumPy array.
            y_array (np.ndarray): The ground truth target dataset as a NumPy array.

        Raises:
            EvaluationError: If inputs are invalid or misaligned.
        """
        if X_array.size == 0 or y_array.size == 0:
            logger.error("Evaluation datasets (X_test or y_test) cannot be empty.")
            raise EvaluationError("Empty dataset provided for evaluation.")

        if X_array.shape[0] != y_array.shape[0]:
            logger.error(
                f"Row mismatch: X_test has {X_array.shape[0]} rows, "
                f"but y_test has {y_array.shape[0]} rows."
            )
            raise EvaluationError("Mismatch in number of samples between X_test and y_test.")

        # Ensure y_array has exactly the number of columns as TARGET_COLUMNS
        if len(y_array.shape) < 2 or y_array.shape[1] != len(self.TARGET_COLUMNS):
            logger.error(
                f"y_test has invalid shape. Expected {len(self.TARGET_COLUMNS)} "
                f"columns, got {y_array.shape[1] if len(y_array.shape) > 1 else 1}."
            )
            raise EvaluationError(f"Missing or invalid target columns in y_test.")

        if not hasattr(model, "predict"):
            logger.error("Provided model does not implement a 'predict' method.")
            raise EvaluationError("Model must implement the 'predict' method.")

    def _calculate_target_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        target_name: str
    ) -> Dict[str, Any]:
        """
        Calculates classification metrics for a single specific target.

        Args:
            y_true (np.ndarray): 1D array of ground truth labels.
            y_pred (np.ndarray): 1D array of predicted labels.
            target_name (str): The name of the target being evaluated.

        Returns:
            Dict[str, Any]: A dictionary of metrics for the specific target.
        """
        logger.debug(f"Calculating metrics for target: {target_name}")

        accuracy = float(accuracy_score(y_true, y_pred))
        
        # 'weighted' accounts for class imbalance by weighting metrics by support
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Output dict=True makes it easily serializable for reporting downstream
        class_report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        conf_matrix = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_score_weighted": float(f1_score),
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist()  # Converted to list for JSON serialization
        }

    def evaluate(
        self, 
        model: BaseEstimator, 
        X_test: Union[pd.DataFrame, np.ndarray], 
        y_test: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Executes a comprehensive multi-target evaluation of the model.

        Args:
            model (BaseEstimator): The trained multi-output model.
            X_test (Union[pd.DataFrame, np.ndarray]): The test feature dataset.
            y_test (Union[pd.DataFrame, np.ndarray]): The test ground truth target dataset.

        Returns:
            Dict[str, Any]: A nested dictionary containing metrics for each target
                as well as system-wide exact match ratio metrics.

        Raises:
            EvaluationError: If prediction or metric calculation fails.
        """
        logger.info("Starting model evaluation process...")
        
        # Internally convert any pandas inputs to NumPy arrays immediately
        X_array = np.asarray(X_test)
        y_array = np.asarray(y_test)
        
        self._validate_inputs(model, X_array, y_array)

        try:
            logger.info("Generating predictions on the test set...")
            predictions = model.predict(X_array)
        except Exception as e:
            logger.error(f"Model prediction failed during evaluation: {e}")
            raise EvaluationError(f"Prediction failed: {e}") from e

        # Ensure predictions are in the correct shape (N_samples, N_targets)
        if len(predictions.shape) == 1:
            logger.error("Expected multi-output predictions, got 1D array.")
            raise EvaluationError("Model did not return multi-output predictions.")
            
        results: Dict[str, Any] = {
            "targets": {},
            "overall": {}
        }

        # Calculate exact match ratio (subset accuracy)
        # In power protection, a relay must get ALL targets correct for a proper decision
        exact_match = np.all(predictions == y_array, axis=1)
        exact_match_ratio = float(np.mean(exact_match))
        
        results["overall"]["exact_match_ratio"] = exact_match_ratio
        results["overall"]["total_samples"] = len(X_array)

        # Calculate metrics for each individual target
        for idx, target in enumerate(self.TARGET_COLUMNS):
            y_true_target = y_array[:, idx]
            y_pred_target = predictions[:, idx]
            
            try:
                target_metrics = self._calculate_target_metrics(
                    y_true_target, y_pred_target, target
                )
                results["targets"][target] = target_metrics
            except Exception as e:
                logger.error(f"Failed to calculate metrics for target '{target}': {e}")
                raise EvaluationError(f"Metric calculation failed for '{target}': {e}") from e

        logger.info(
            f"Evaluation complete. Overall Exact Match Ratio: {exact_match_ratio:.4f}"
        )
        return results

    def save_evaluation_report(
        self, 
        results: Dict[str, Any], 
        output_dir: Union[str, Path], 
        filename: str = "evaluation_report.json"
    ) -> Path:
        """
        Saves the generated evaluation metrics to a JSON file for CI/CD tracking.

        Args:
            results (Dict[str, Any]): The evaluation results dictionary.
            output_dir (Union[str, Path]): The directory to save the report in.
            filename (str): The name of the output JSON file.

        Returns:
            Path: The full path to the saved evaluation report.

        Raises:
            EvaluationError: If the report cannot be saved to disk.
        """
        output_path = Path(output_dir)
        
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            file_path = output_path / filename
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"Evaluation report successfully saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report to {output_path}: {e}")
            raise EvaluationError(f"Report saving failed: {e}") from e