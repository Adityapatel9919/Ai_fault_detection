"""
save_load.py

Handles secure serialization, deserialization, and lifecycle management of 
trained machine learning models for the AI-Based Real-Time Fault Detection system.

This module guarantees enterprise-grade model persistence by enforcing cryptographic 
checksums (SHA-256) to prevent corruption during edge deployment. It seamlessly 
binds models with their metadata (features, metrics, creation timestamps, and 
environment details) to ensure full traceability and reproducibility in production.
"""

import hashlib
import json
import logging
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import sklearn
from sklearn.base import BaseEstimator

# Initialize module logger
logger = logging.getLogger(__name__)


class ModelStorageError(Exception):
    """Base exception for model serialization and deserialization failures."""
    pass


class ChecksumMismatchError(ModelStorageError):
    """Exception raised when a loaded model's hash does not match its metadata."""
    pass


class ModelNotFoundError(ModelStorageError):
    """Exception raised when attempting to load a model that does not exist."""
    pass


@dataclass
class ModelMetadata:
    """
    Strict schema for model metadata to ensure traceability and reproducibility.
    """
    model_name: str
    version: str
    creation_timestamp_utc: str
    python_version: str
    sklearn_version: str
    os_platform: str
    features: List[str]
    targets: List[str]
    metrics: Dict[str, Any]
    model_checksum_sha256: str = ""
    description: str = "AI-Based Real-Time Fault Detection Model"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


class ModelArtifactManager:
    """
    Manages the secure persistence and retrieval of machine learning artifacts,
    specifically designed for CI/CD pipelines and embedded edge deployment.
    """

    TARGET_COLUMNS = ['sc_type', 'fault_target', 'phase_select']

    def __init__(self, base_artifact_dir: Union[str, Path] = "artifacts/models") -> None:
        """
        Initializes the ModelArtifactManager.

        Args:
            base_artifact_dir (Union[str, Path]): Root directory where all models 
                and their corresponding metadata are securely stored.
        """
        self.base_dir = Path(base_artifact_dir).resolve()
        
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"ModelArtifactManager initialized at: {self.base_dir}")
        except Exception as e:
            logger.error(f"Failed to create base artifact directory at {self.base_dir}: {e}")
            raise ModelStorageError(f"Directory initialization failed: {e}") from e

    def _generate_sha256_checksum(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Computes a cryptographic SHA-256 hash of a file efficiently using chunking.
        Vital for ensuring model integrity before edge deployment or inference.

        Args:
            file_path (Path): Path to the target file.
            chunk_size (int): Size of byte chunks for memory-efficient reading.

        Returns:
            str: The hexadecimal SHA-256 checksum.

        Raises:
            ModelStorageError: If the file cannot be read.
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(chunk_size), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute SHA-256 for {file_path}: {e}")
            raise ModelStorageError(f"Checksum computation failed: {e}") from e

    def save_model(
        self,
        model: BaseEstimator,
        model_name: str,
        features: List[str],
        metrics: Dict[str, Any],
        version: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> Path:
        """
        Securely serializes a trained machine learning model and its metadata.

        Args:
            model (BaseEstimator): The trained model to save.
            model_name (str): Identifier for the model architecture (e.g., 'svm_multi').
            features (List[str]): Ordered list of features the model requires.
            metrics (Dict[str, Any]): Evaluation metrics (Exact Match Ratio, F1, etc.).
            version (Optional[str]): Semantic or timestamp version. If None, auto-generates.
            hyperparameters (Optional[Dict[str, Any]]): Model configuration parameters.
            description (Optional[str]): Optional deployment notes.

        Returns:
            Path: The directory containing the serialized `.joblib` and `metadata.json`.

        Raises:
            ModelStorageError: If serialization fails at any step.
        """
        if not version:
            version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        model_dir = self.base_dir / model_name / version
        
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving model '{model_name}' (Version: {version}) to {model_dir}")
        except Exception as e:
            logger.error(f"Failed to create model directory {model_dir}: {e}")
            raise ModelStorageError(f"Directory creation failed: {e}") from e

        model_filepath = model_dir / "model.joblib"
        metadata_filepath = model_dir / "metadata.json"

        # 1. Serialize the BaseEstimator using Joblib (with compression)
        try:
            # compress=3 offers a good balance between disk space and serialization speed
            joblib.dump(model, model_filepath, compress=3)
            logger.debug(f"Model serialized successfully to {model_filepath}")
        except Exception as e:
            logger.error(f"Joblib dump failed for {model_filepath}: {e}")
            # Cleanup partially saved files
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
            raise ModelStorageError(f"Model serialization failed: {e}") from e

        # 2. Compute cryptographically secure checksum
        checksum = self._generate_sha256_checksum(model_filepath)

        # 3. Construct Metadata
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            creation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version.split(" ")[0],
            sklearn_version=sklearn.__version__,
            os_platform=platform.platform(),
            features=features,
            targets=self.TARGET_COLUMNS,
            metrics=metrics,
            model_checksum_sha256=checksum,
            hyperparameters=hyperparameters or {},
            description=description or "AI-Based Real-Time Fault Detection Model"
        )

        # 4. Save Metadata
        try:
            with open(metadata_filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f, indent=4)
            logger.debug(f"Metadata saved successfully to {metadata_filepath}")
        except Exception as e:
            logger.error(f"Metadata JSON dump failed for {metadata_filepath}: {e}")
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
            raise ModelStorageError(f"Metadata serialization failed: {e}") from e

        logger.info(f"Successfully saved and verified '{model_name}' version '{version}'.")
        return model_dir

    def load_model(
        self, 
        model_name: str, 
        version: str, 
        validate_integrity: bool = True
    ) -> Tuple[BaseEstimator, ModelMetadata]:
        """
        Retrieves a trained model and validates its cryptographic integrity before loading.

        Args:
            model_name (str): Identifier for the model architecture.
            version (str): The specific version string to load.
            validate_integrity (bool): If True, computes and validates the SHA-256 hash.

        Returns:
            Tuple[BaseEstimator, ModelMetadata]: The fully instantiated scikit-learn 
                estimator and its validated metadata object.

        Raises:
            ModelNotFoundError: If the requested model/version does not exist.
            ChecksumMismatchError: If the model file has been modified or corrupted.
            ModelStorageError: If parsing or deserialization fails.
        """
        model_dir = self.base_dir / model_name / version
        model_filepath = model_dir / "model.joblib"
        metadata_filepath = model_dir / "metadata.json"

        if not model_dir.exists() or not model_filepath.exists() or not metadata_filepath.exists():
            logger.error(f"Artifacts missing for model '{model_name}' version '{version}'")
            raise ModelNotFoundError(f"Model artifacts not found in {model_dir}")

        logger.info(f"Initiating load sequence for '{model_name}' (Version: {version})")

        # 1. Load Metadata
        try:
            with open(metadata_filepath, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata(**metadata_dict)
        except Exception as e:
            logger.error(f"Failed to parse metadata at {metadata_filepath}: {e}")
            raise ModelStorageError(f"Metadata parsing failed: {e}") from e

        # 2. Validate Integrity (Crucial for Edge Security)
        if validate_integrity:
            logger.debug(f"Validating SHA-256 checksum for {model_filepath.name}...")
            current_checksum = self._generate_sha256_checksum(model_filepath)
            
            if current_checksum != metadata.model_checksum_sha256:
                logger.critical(
                    f"SECURITY ALERT: Checksum mismatch for '{model_name}'! "
                    f"Expected {metadata.model_checksum_sha256}, got {current_checksum}. "
                    f"The model file may be corrupted or tampered with."
                )
                raise ChecksumMismatchError(
                    f"Integrity check failed for model {model_name} version {version}."
                )
            logger.debug("Integrity verification passed.")

        # 3. Deserialize Model
        try:
            start_time = time.time()
            model = joblib.load(model_filepath)
            elapsed = time.time() - start_time
            logger.info(f"Model successfully loaded into memory in {elapsed:.3f}s.")
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to deserialize model at {model_filepath}: {e}")
            raise ModelStorageError(f"Model deserialization failed: {e}") from e

    def list_available_models(self) -> Dict[str, List[str]]:
        """
        Scans the artifact directory and discovers all available models and their versions.

        Returns:
            Dict[str, List[str]]: A mapping of model names to a list of available versions.
        """
        if not self.base_dir.exists():
            return {}

        available_models = {}
        for model_path in self.base_dir.iterdir():
            if model_path.is_dir():
                versions = []
                for version_path in model_path.iterdir():
                    if version_path.is_dir() and (version_path / "model.joblib").exists():
                        versions.append(version_path.name)
                
                if versions:
                    # Sort versions assuming chronological timestamp strings
                    available_models[model_path.name] = sorted(versions, reverse=True)
                    
        return available_models

    def get_latest_version(self, model_name: str) -> str:
        """
        Determines the most recently saved version of a specific model.

        Args:
            model_name (str): Identifier for the model architecture.

        Returns:
            str: The version string of the latest model artifact.

        Raises:
            ModelNotFoundError: If the model has no saved versions.
        """
        models = self.list_available_models()
        if model_name not in models or not models[model_name]:
            logger.error(f"No versions found for model '{model_name}'.")
            raise ModelNotFoundError(f"Model '{model_name}' has no saved artifacts.")
        
        # list_available_models sorts in reverse order, so index 0 is the newest
        latest = models[model_name][0]
        logger.debug(f"Latest version for '{model_name}' resolved to '{latest}'")
        return latest

    def delete_model_version(self, model_name: str, version: str) -> None:
        """
        Permanently deletes a specific model version from disk to manage storage limits.

        Args:
            model_name (str): Identifier for the model architecture.
            version (str): The specific version string to delete.

        Raises:
            ModelStorageError: If deletion fails.
        """
        model_dir = self.base_dir / model_name / version
        if not model_dir.exists():
            logger.warning(f"Attempted to delete non-existent model directory: {model_dir}")
            return
            
        try:
            shutil.rmtree(model_dir)
            logger.info(f"Successfully deleted model artifacts at {model_dir}")
        except Exception as e:
            logger.error(f"Failed to delete model directory {model_dir}: {e}")
            raise ModelStorageError(f"Artifact deletion failed: {e}") from e