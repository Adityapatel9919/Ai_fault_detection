"""
compare_models.py

Executes a robust comparison of multiple machine learning models based on 
their multi-target evaluation reports for the AI-Based Real-Time Fault Detection system.

This module parses evaluation metrics (e.g., Exact Match Ratio, F1-Score) across 
different models, ranks them according to user-defined criteria, and identifies 
the optimal model for deployment on the ATM90E32AS hardware ecosystem.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Initialize module logger
logger = logging.getLogger(__name__)


class ComparisonError(Exception):
    """Exception raised for errors occurring during model comparison."""
    pass


class ModelComparator:
    """
    Compares evaluation reports of multiple trained models to determine 
    the best performing architecture based on specified criteria.
    """

    # Allowed levels for metric extraction based on evaluate.py structure
    ALLOWED_LEVELS = ['overall', 'targets']

    def __init__(self) -> None:
        """Initializes the ModelComparator."""
        pass

    def load_reports_from_directory(
        self, 
        directory: Union[str, Path], 
        file_pattern: str = "*_report.json"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Scans a directory for evaluation report JSON files and loads them.

        Args:
            directory (Union[str, Path]): Directory containing evaluation reports.
            file_pattern (str): Glob pattern to match report files.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping model names (derived 
                from filenames) to their parsed evaluation report dictionaries.

        Raises:
            ComparisonError: If the directory does not exist or files are malformed.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Invalid evaluation report directory: {dir_path}")
            raise ComparisonError(f"Directory not found: {dir_path}")

        reports = {}
        for file_path in dir_path.glob(file_pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                # Derive model name from filename (e.g., "svm_eval_report.json" -> "svm_eval")
                model_name = file_path.stem.replace('_report', '')
                reports[model_name] = report_data
                logger.debug(f"Loaded evaluation report for model: {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to parse report {file_path.name}: {e}")
                continue

        if not reports:
            logger.error(f"No valid reports found in {dir_path} matching {file_pattern}")
            raise ComparisonError("No evaluation reports could be loaded.")

        return reports

    def _extract_metric(
        self, 
        report: Dict[str, Any], 
        metric: str, 
        level: str, 
        target_name: Optional[str] = None
    ) -> float:
        """
        Safely extracts a specific numerical metric from a multi-target evaluation report.

        Args:
            report (Dict[str, Any]): The evaluation report dictionary.
            metric (str): The metric key to extract (e.g., 'exact_match_ratio', 'f1_score_weighted').
            level (str): The logical level ('overall' or 'targets').
            target_name (Optional[str]): The specific target name if level is 'targets'.

        Returns:
            float: The extracted metric value.

        Raises:
            ComparisonError: If the metric path is invalid or missing.
        """
        if level not in self.ALLOWED_LEVELS:
            raise ComparisonError(f"Invalid level '{level}'. Allowed: {self.ALLOWED_LEVELS}")

        try:
            if level == 'overall':
                return float(report['overall'][metric])
            
            if level == 'targets':
                if not target_name:
                    raise ComparisonError("target_name must be provided for 'targets' level.")
                return float(report['targets'][target_name][metric])
                
        except KeyError as e:
            logger.error(f"Metric path missing in report. Level: {level}, Target: {target_name}, Metric: {metric}")
            raise ComparisonError(f"Failed to extract metric '{metric}': {e}") from e
        except ValueError as e:
            logger.error(f"Metric '{metric}' is not a valid float.")
            raise ComparisonError(f"Invalid metric value format: {e}") from e
            
        return 0.0  # Fallback, theoretically unreachable

    def compare(
        self, 
        reports: Dict[str, Dict[str, Any]], 
        primary_metric: str = "exact_match_ratio",
        level: str = "overall",
        target_name: Optional[str] = None,
        higher_is_better: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Compares and ranks multiple model reports based on a primary metric.

        Args:
            reports (Dict[str, Dict[str, Any]]): Dictionary of model names to evaluation reports.
            primary_metric (str): The key of the metric to rank by.
            level (str): 'overall' or 'targets'.
            target_name (Optional[str]): Specific target if evaluating on a single target.
            higher_is_better (bool): True if ascending metric value means better performance.

        Returns:
            List[Tuple[str, float]]: A list of (model_name, metric_value) tuples, 
                sorted from best to worst performance.

        Raises:
            ComparisonError: If reports are empty or metric extraction fails for any model.
        """
        if not reports:
            logger.error("Empty reports dictionary provided for comparison.")
            raise ComparisonError("Cannot compare empty reports.")

        logger.info(
            f"Comparing {len(reports)} models on {primary_metric} "
            f"(Level: {level}, Target: {target_name})"
        )

        rankings = []
        for model_name, report in reports.items():
            metric_val = self._extract_metric(report, primary_metric, level, target_name)
            rankings.append((model_name, metric_val))

        # Sort based on performance direction
        rankings.sort(key=lambda x: x[1], reverse=higher_is_better)
        
        for rank, (model, score) in enumerate(rankings, 1):
            logger.debug(f"Rank {rank}: {model} ({primary_metric}: {score:.4f})")

        return rankings

    def get_best_model(
        self, 
        reports: Dict[str, Dict[str, Any]], 
        primary_metric: str = "exact_match_ratio",
        level: str = "overall",
        target_name: Optional[str] = None,
        higher_is_better: bool = True
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Identifies and returns the best model and its full report based on the criteria.

        Args:
            reports (Dict[str, Dict[str, Any]]): Dictionary of model reports.
            primary_metric (str): The key of the metric to rank by.
            level (str): 'overall' or 'targets'.
            target_name (Optional[str]): Specific target if evaluating on a single target.
            higher_is_better (bool): True if ascending metric value means better performance.

        Returns:
            Tuple[str, float, Dict[str, Any]]: A tuple containing the best model's name, 
                its score, and its full evaluation report dictionary.
        """
        rankings = self.compare(
            reports=reports,
            primary_metric=primary_metric,
            level=level,
            target_name=target_name,
            higher_is_better=higher_is_better
        )
        
        best_model_name, best_score = rankings[0]
        best_model_report = reports[best_model_name]
        
        logger.info(
            f"Best model identified: {best_model_name} "
            f"with {primary_metric} = {best_score:.4f}"
        )
        
        return best_model_name, best_score, best_model_report

    def save_comparison_summary(
        self, 
        rankings: List[Tuple[str, float]], 
        metric_used: str,
        output_dir: Union[str, Path], 
        filename: str = "model_comparison_summary.json"
    ) -> Path:
        """
        Saves the resulting rankings into a summary JSON file for CI/CD pipelines.

        Args:
            rankings (List[Tuple[str, float]]): The sorted list of model rankings.
            metric_used (str): The metric that was used to generate the rankings.
            output_dir (Union[str, Path]): Directory to save the summary file.
            filename (str): Name of the summary JSON file.

        Returns:
            Path: The full path to the saved summary file.

        Raises:
            ComparisonError: If the file cannot be written to disk.
        """
        output_path = Path(output_dir)
        
        summary_data = {
            "ranking_metric": metric_used,
            "total_models_compared": len(rankings),
            "best_model": rankings[0][0] if rankings else None,
            "best_score": rankings[0][1] if rankings else None,
            "rankings": [{"model": model, "score": score} for model, score in rankings]
        }
        
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            file_path = output_path / filename
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4)
                
            logger.info(f"Comparison summary successfully saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save comparison summary to {output_path}: {e}")
            raise ComparisonError(f"Summary report saving failed: {e}") from e