"""
train.py

Executes the comprehensive automated training pipeline for the AI-Based 
Real-Time Fault Detection system.

This module orchestrates the complete machine learning lifecycle: loading data, 
leveraging the pre-existing data preprocessor, training all specified models 
(SVM, Random Forest, XGBoost, ANN) via the ModelFactory, evaluating their 
performance, saving artifacts, and generating a detailed comparison report to 
identify the optimal model for edge deployment.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputClassifier

# Import existing enterprise modules
from .data_preprocessor import DataPreprocessor, PreprocessedData
from .model_factory import ModelFactory, FactoryConfig
from .evaluate import ModelEvaluator
from .save_load import ModelArtifactManager

# Initialize module logger
logger = logging.getLogger(__name__)


class PipelineExecutionError(Exception):
    """Exception raised for fatal errors during the automated training pipeline."""
    pass


class AutomatedTrainingPipeline:
    """
    Orchestrates the end-to-end multi-model training, evaluation, and comparison 
    process for power transmission line fault classification.
    """

    TARGET_COLUMNS = ['sc_type', 'fault_target', 'phase_select']
    MODELS_TO_TRAIN = ['svm', 'random_forest', 'xgboost', 'mlp']

    def __init__(self, output_dir: str = "results"):
        """
        Initializes the AutomatedTrainingPipeline.

        Args:
            output_dir (str): Directory where the comparison and best model 
                artifacts will be explicitly saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.factory = ModelFactory()
        self.evaluator = ModelEvaluator()
        self.artifact_manager = ModelArtifactManager(base_artifact_dir=self.output_dir / "model_store")

        # Populated by _load_and_preprocess(); kept around so downstream
        # code (e.g. decoding predictions back to string labels) has
        # access to the exact fitted preprocessor without re-fitting it.
        self.preprocessor: Optional[DataPreprocessor] = None

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _load_and_preprocess(
        self, dataset_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Loads the Parquet dataset and delegates all preprocessing and splitting
        to the external `DataPreprocessor` module.

        `DataPreprocessor.fit_transform()` returns a `PreprocessedData` object,
        NOT a DataFrame -- it exposes `X_train`/`X_test` as numpy arrays,
        `y_train`/`y_test` as `dict[target_name -> numpy array]`, and
        `feature_names` as a plain list of column names. There is nothing to
        `.drop()` or index with `[...]` on that object; every value needed
        here is read directly off its attributes.

        Args:
            dataset_path (Path): Path to the input dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
                X_train, X_test, y_train, y_test, feature_names. `y_train`/
                `y_test` are 2D numpy arrays of shape
                `(n_samples, len(TARGET_COLUMNS))`, one column per target in
                `TARGET_COLUMNS` order, ready for `MultiOutputClassifier`.

        Raises:
            PipelineExecutionError: If data loading or preprocessing fails.
        """
        logger.info(f"Loading dataset from {dataset_path}...")
        try:
            df = pd.read_parquet(dataset_path)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise PipelineExecutionError(f"Dataset load failed: {e}") from e

        logger.info("Delegating to DataPreprocessor...")
        try:
            self.preprocessor = DataPreprocessor()
            data: PreprocessedData = self.preprocessor.fit_transform(df)

            missing_targets = [t for t in self.TARGET_COLUMNS if t not in data.y_train]
            if missing_targets:
                raise PipelineExecutionError(
                    f"PreprocessedData is missing expected target(s) {missing_targets}. "
                    f"Available targets: {data.target_names()}"
                )

            X_train, X_test = data.X_train, data.X_test
            y_train = np.column_stack([data.y_train[t] for t in self.TARGET_COLUMNS])
            y_test = np.column_stack([data.y_test[t] for t in self.TARGET_COLUMNS])
            feature_names = data.feature_names

            logger.info(
                f"Preprocessing complete. Train size: {X_train.shape[0]}, "
                f"Test size: {X_test.shape[0]}, Features: {len(feature_names)}"
            )
            return X_train, X_test, y_train, y_test, feature_names
        except PipelineExecutionError:
            raise
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise PipelineExecutionError(f"Preprocessing failed: {e}") from e

    def _aggregate_metrics(self, report: Dict[str, Any]) -> Dict[str, float]:
        """
        Aggregates multi-target metrics into single averages for the comparison table.

        Args:
            report (Dict[str, Any]): The evaluation report generated by ModelEvaluator.

        Returns:
            Dict[str, float]: Aggregated scores (Accuracy, Precision, Recall, F1).
        """
        acc_list, prec_list, rec_list, f1_list = [], [], [], []
        
        for target in self.TARGET_COLUMNS:
            t_metrics = report['targets'][target]
            acc_list.append(t_metrics.get('accuracy', 0.0))
            prec_list.append(t_metrics.get('precision_weighted', 0.0))
            rec_list.append(t_metrics.get('recall_weighted', 0.0))
            f1_list.append(t_metrics.get('f1_score_weighted', 0.0))
            
        return {
            "Accuracy": float(np.mean(acc_list)),
            "Precision": float(np.mean(prec_list)),
            "Recall": float(np.mean(rec_list)),
            "F1": float(np.mean(f1_list)),
            "Exact Match Ratio": report['overall'].get('exact_match_ratio', 0.0)
        }

    def run(self, dataset_filepath: str) -> None:
        """
        Executes the fully automated pipeline.

        Args:
            dataset_filepath (str): Path to 'protect90_features_ml_ready.parquet'.
        """
        logger.info("=== Starting Automated Training Pipeline ===")
        
        # 1-3. Load, Preprocess, and Split Dataset
        dataset_path = Path(dataset_filepath)
        X_train, X_test, y_train, y_test, features_list = self._load_and_preprocess(dataset_path)
        
        comparison_results = []
        trained_models: Dict[str, BaseEstimator] = {}
        evaluation_reports: Dict[str, Dict[str, Any]] = {}

        # 4-7. Train, Evaluate, and Save every model
        for model_name in self.MODELS_TO_TRAIN:
            logger.info(f"--- Processing Model: {model_name} ---")
            
            try:
                # 4a. Instantiate model
                config = FactoryConfig(model_name=model_name)
                base_model = self.factory.create(config)
                multi_model = MultiOutputClassifier(base_model, n_jobs=-1)
                
                # 4b. Train and record time
                logger.info(f"Training {model_name}...")
                train_start = time.time()
                multi_model.fit(X_train, y_train)
                train_time = time.time() - train_start
                logger.info(f"Training completed in {train_time:.2f}s.")
                
                # 5a. Predict and record time (latency)
                logger.info(f"Evaluating {model_name}...")
                pred_start = time.time()
                # Dummy prediction to measure bulk prediction time
                _ = multi_model.predict(X_test)
                predict_time = time.time() - pred_start
                
                # 5b. Evaluate
                report = self.evaluator.evaluate(multi_model, X_test, y_test)
                evaluation_reports[model_name] = report
                
                # Extract and aggregate metrics
                agg_metrics = self._aggregate_metrics(report)
                
                # Append to comparison DataFrame tracking
                comparison_results.append({
                    "Model": model_name,
                    "Accuracy": agg_metrics["Accuracy"],
                    "Precision": agg_metrics["Precision"],
                    "Recall": agg_metrics["Recall"],
                    "F1": agg_metrics["F1"],
                    "Exact Match Ratio": agg_metrics["Exact Match Ratio"],
                    "Training Time": round(train_time, 4),
                    "Prediction Time": round(predict_time, 4)
                })
                
                # 6. Save Model using Artifact Manager
                self.artifact_manager.save_model(
                    model=multi_model,
                    model_name=model_name,
                    features=features_list,
                    metrics=agg_metrics
                )
                
                # 7. Save Evaluation Report
                self.evaluator.save_evaluation_report(
                    results=report,
                    output_dir=self.output_dir / "reports",
                    filename=f"{model_name}_report.json"
                )
                
                trained_models[model_name] = multi_model
                
            except Exception as e:
                logger.error(f"Pipeline failed for model {model_name}: {e}")
                continue

        if not comparison_results:
            logger.error("No models were successfully trained. Aborting.")
            return

        # 8. Create Comparison DataFrame
        logger.info("Generating Comparison DataFrame...")
        comparison_df = pd.DataFrame(comparison_results)
        
        # 9. Sort by Exact Match Ratio (Descending)
        comparison_df = comparison_df.sort_values(by="Exact Match Ratio", ascending=False)
        
        # 10. Print best model
        best_model_name = comparison_df.iloc[0]["Model"]
        best_score = comparison_df.iloc[0]["Exact Match Ratio"]
        
        logger.info("=== PIPELINE COMPLETION SUMMARY ===")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        logger.info(f"\n🏆 BEST MODEL IDENTIFIED: {best_model_name}")
        logger.info(f"🏆 EXACT MATCH RATIO: {best_score:.4f}")
        
        # 11. Save comparison.csv
        csv_path = self.output_dir / "comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison report to {csv_path}")
        
        # 12. Save best_model.pkl (Explicitly as requested)
        best_model_instance = trained_models[best_model_name]
        best_model_path = self.output_dir / "best_model.pkl"
        joblib.dump(best_model_instance, best_model_path)
        logger.info(f"Saved best model artifact to {best_model_path}")
        
        # 13. Save best_model_report.json
        best_report_path = self.output_dir / "best_model_report.json"
        with open(best_report_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_reports[best_model_name], f, indent=4)
        logger.info(f"Saved best model evaluation report to {best_report_path}")
        
        logger.info("Automated Training Pipeline executed successfully.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    dataset_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "protect90_features_ml_ready.parquet"
    )

    pipeline = AutomatedTrainingPipeline(
        output_dir=PROJECT_ROOT / "results"
    )

    try:
        pipeline.run(str(dataset_file))
    except Exception as e:
        logging.critical(f"Fatal pipeline error: {e}")