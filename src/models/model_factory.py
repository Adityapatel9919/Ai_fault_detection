"""
model_factory.py

This module implements a robust Factory Pattern for constructing and configuring
machine learning models used in the AI-Based Real-Time Fault Detection system.
It supports dynamic registration of model builders, parameter validation, and
graceful fallbacks for optional dependencies.

It operates exclusively as a builder and does not execute training or evaluation.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Tuple

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# Initialize module logger
logger = logging.getLogger(__name__)


class ModelCreationError(Exception):
    """Exception raised for errors occurring during model instantiation."""
    pass


@dataclass
class FactoryConfig:
    """
    Configuration dataclass for model instantiation.
    
    Attributes:
        model_name (str): The registered name of the model to build.
        params (Dict[str, Any]): Hyperparameters for the model.
        ensemble_configs (Optional[List['FactoryConfig']]): Configurations for 
            base estimators when building ensemble models (Voting, Stacking).
        voting_type (str): 'hard' or 'soft', specific to VotingClassifier.
    """
    model_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    ensemble_configs: Optional[List['FactoryConfig']] = None
    voting_type: str = 'hard'


class ModelRegistry:
    """
    Registry to manage available model builders dynamically.
    """
    
    def __init__(self) -> None:
        """Initializes the ModelRegistry with an empty builder dictionary."""
        self._builders: Dict[str, Callable[[FactoryConfig], BaseEstimator]] = {}

    def register(self, name: str, builder: Callable[[FactoryConfig], BaseEstimator]) -> None:
        """
        Registers a model builder function under a specific name.
        
        Args:
            name (str): The string identifier for the model.
            builder (Callable): The function responsible for instantiating the model.
        """
        if name in self._builders:
            logger.warning(f"Overwriting existing builder for model '{name}'.")
        self._builders[name] = builder
        logger.debug(f"Model '{name}' registered successfully.")

    def unregister(self, name: str) -> None:
        """
        Unregisters a previously registered model builder.
        
        Args:
            name (str): The string identifier for the model.
            
        Raises:
            KeyError: If the model name is not found in the registry.
        """
        if name not in self._builders:
            logger.error(f"Cannot unregister '{name}': Model not found.")
            raise KeyError(f"Model '{name}' is not registered.")
        del self._builders[name]
        logger.debug(f"Model '{name}' unregistered successfully.")

    def get_builder(self, name: str) -> Callable[[FactoryConfig], BaseEstimator]:
        """
        Retrieves a registered builder function.
        
        Args:
            name (str): The string identifier for the model.
            
        Returns:
            Callable: The model builder function.
            
        Raises:
            ModelCreationError: If the model name is not registered.
        """
        builder = self._builders.get(name)
        if not builder:
            logger.error(f"Requested model '{name}' is not registered.")
            raise ModelCreationError(f"Model '{name}' is not registered.")
        return builder

    def available_models(self) -> List[str]:
        """
        Returns a list of all currently registered model names.
        
        Returns:
            List[str]: List of model identifiers.
        """
        return list(self._builders.keys())


class BaseModelFactory(ABC):
    """
    Abstract base class for the model factory, establishing the contract
    for factory implementations.
    """
    
    @abstractmethod
    def create(self, config: FactoryConfig) -> BaseEstimator:
        """
        Creates and configures a machine learning model based on the config.
        
        Args:
            config (FactoryConfig): The configuration object for the model.
            
        Returns:
            BaseEstimator: An instantiated scikit-learn compatible estimator.
        """
        pass

    @abstractmethod
    def validate_parameters(
        self, model_class: Type[BaseEstimator], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates provided parameters against the target class signature.
        
        Args:
            model_class (Type[BaseEstimator]): The class of the model being built.
            params (Dict[str, Any]): The provided hyperparameters.
            
        Returns:
            Dict[str, Any]: The filtered and validated dictionary of parameters.
        """
        pass


class ModelFactory(BaseModelFactory):
    """
    Concrete implementation of the model factory.
    Handles configuration, parameter validation, and instantiation of models.
    """
    
    def __init__(self) -> None:
        """Initializes the ModelFactory and registers default models."""
        self.registry = ModelRegistry()
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Registers the built-in model types into the registry."""
        self.register_model('svm', self.create_svm)
        self.register_model('random_forest', self.create_random_forest)
        self.register_model('xgboost', self.create_xgboost)
        self.register_model('mlp', self.create_ann)
        self.register_model('voting', self.create_voting_classifier)
        self.register_model('stacking', self.create_stacking_classifier)

    def register_model(self, name: str, builder: Callable[[FactoryConfig], BaseEstimator]) -> None:
        """
        Registers a new model with the internal registry.
        
        Args:
            name (str): Identifier for the model.
            builder (Callable): Instantiation function for the model.
        """
        self.registry.register(name, builder)

    def unregister_model(self, name: str) -> None:
        """
        Removes a model from the internal registry.
        
        Args:
            name (str): Identifier for the model.
        """
        self.registry.unregister(name)

    def available_models(self) -> List[str]:
        """
        Lists all currently supported model names.
        
        Returns:
            List[str]: List of registered model names.
        """
        return self.registry.available_models()

    def validate_parameters(
        self, model_class: Type[Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filters and validates hyperparameters by inspecting the model class signature.
        Logs warnings for any invalid parameters that are ignored.
        
        Args:
            model_class (Type[Any]): The class type being instantiated.
            params (Dict[str, Any]): The raw parameter dictionary.
            
        Returns:
            Dict[str, Any]: Parameters that are strictly accepted by the model's __init__.
        """
        validated = {}
        try:
            signature = inspect.signature(model_class.__init__)
            valid_keys = set(signature.parameters.keys())
            
            # Remove 'self' from valid keys if present
            valid_keys.discard('self')
            
            # If kwargs is present, the model accepts anything
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD 
                for param in signature.parameters.values()
            )

            for key, value in params.items():
                if key in valid_keys or accepts_kwargs:
                    validated[key] = value
                else:
                    logger.warning(
                        f"Parameter '{key}' is invalid for {model_class.__name__} "
                        f"and will be ignored."
                    )
        except ValueError as e:
            logger.error(f"Failed to inspect signature for {model_class.__name__}: {e}")
            # Fallback: return raw params if inspection fails
            return params
            
        return validated

    def create(self, config: FactoryConfig) -> BaseEstimator:
        """
        Creates a model dynamically based on the provided configuration.
        
        Args:
            config (FactoryConfig): Configuration detailing the model and params.
            
        Returns:
            BaseEstimator: The fully constructed model.
            
        Raises:
            ModelCreationError: If creation fails or model is not registered.
        """
        logger.info(f"Creating model: {config.model_name}")
        builder = self.registry.get_builder(config.model_name)
        try:
            model = builder(config)
            logger.info(f"Successfully created {config.model_name}.")
            return model
        except Exception as e:
            logger.error(f"Failed to create model '{config.model_name}': {e}")
            raise ModelCreationError(f"Error instantiating {config.model_name}: {e}") from e

    def create_svm(self, config: FactoryConfig) -> BaseEstimator:
        """
        Builds a Support Vector Machine (SVC) classifier.
        
        Args:
            config (FactoryConfig): Configuration containing SVM hyperparameters.
            
        Returns:
            BaseEstimator: Configured SVC instance.
        """
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        }
        merged_params = {**default_params, **config.params}
        valid_params = self.validate_parameters(SVC, merged_params)
        return SVC(**valid_params)

    def create_random_forest(self, config: FactoryConfig) -> BaseEstimator:
        """
        Builds a Random Forest classifier.
        
        Args:
            config (FactoryConfig): Configuration containing RF hyperparameters.
            
        Returns:
            BaseEstimator: Configured RandomForestClassifier instance.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'n_jobs': -1,
            'random_state': 42
        }
        merged_params = {**default_params, **config.params}
        valid_params = self.validate_parameters(RandomForestClassifier, merged_params)
        return RandomForestClassifier(**valid_params)

    def create_xgboost(self, config: FactoryConfig) -> BaseEstimator:
        """
        Builds an XGBoost classifier. Handles missing xgboost library gracefully.
        
        Args:
            config (FactoryConfig): Configuration containing XGBoost hyperparameters.
            
        Returns:
            BaseEstimator: Configured XGBClassifier instance.
            
        Raises:
            ModelCreationError: If xgboost is not installed in the environment.
        """
        if not XGB_AVAILABLE:
            logger.error("XGBoost is requested but not installed.")
            raise ModelCreationError(
                "XGBoost library is not installed. Please install it using 'pip install xgboost'."
            )
            
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1
        }
        merged_params = {**default_params, **config.params}
        valid_params = self.validate_parameters(XGBClassifier, merged_params)
        return XGBClassifier(**valid_params)

    def create_ann(self, config: FactoryConfig) -> BaseEstimator:
        """
        Builds a Multi-Layer Perceptron (MLP) Artificial Neural Network.
        
        Args:
            config (FactoryConfig): Configuration containing MLP hyperparameters.
            
        Returns:
            BaseEstimator: Configured MLPClassifier instance.
        """
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True
        }
        merged_params = {**default_params, **config.params}
        valid_params = self.validate_parameters(MLPClassifier, merged_params)
        return MLPClassifier(**valid_params)

    def _build_ensemble_estimators(
        self, configs: Optional[List[FactoryConfig]]
    ) -> List[Tuple[str, BaseEstimator]]:
        """
        Helper method to construct the base estimators for ensemble models.
        
        Args:
            configs (Optional[List[FactoryConfig]]): Configurations for base models.
            
        Returns:
            List[Tuple[str, BaseEstimator]]: A list of tuples containing a string 
                identifier and the instantiated estimator.
                
        Raises:
            ModelCreationError: If configurations are empty or invalid.
        """
        if not configs:
            logger.error("Ensemble models require 'ensemble_configs'.")
            raise ModelCreationError("Cannot build ensemble without base estimator configs.")
            
        estimators = []
        for i, est_config in enumerate(configs):
            est_name = f"{est_config.model_name}_{i}"
            est_model = self.create(est_config)
            estimators.append((est_name, est_model))
            
        return estimators

    def create_voting_classifier(self, config: FactoryConfig) -> BaseEstimator:
        """
        Builds a Voting Classifier from a list of base configurations.
        
        Args:
            config (FactoryConfig): Configuration containing ensemble configs and voting type.
            
        Returns:
            BaseEstimator: Configured VotingClassifier instance.
        """
        estimators = self._build_ensemble_estimators(config.ensemble_configs)
        
        voting_type = config.voting_type
        if voting_type not in ['hard', 'soft']:
            logger.warning(f"Invalid voting type '{voting_type}'. Defaulting to 'hard'.")
            voting_type = 'hard'
            
        valid_params = self.validate_parameters(VotingClassifier, config.params)
        
        # Override estimators and voting type explicitly
        valid_params['estimators'] = estimators
        valid_params['voting'] = voting_type
        
        return VotingClassifier(**valid_params)

    def create_stacking_classifier(self, config: FactoryConfig) -> BaseEstimator:
        """
        Builds a Stacking Classifier with LogisticRegression as the default meta-learner.
        
        Args:
            config (FactoryConfig): Configuration containing base estimator configs.
            
        Returns:
            BaseEstimator: Configured StackingClassifier instance.
        """
        estimators = self._build_ensemble_estimators(config.ensemble_configs)
        
        # Determine meta-learner configuration if provided, otherwise default
        final_estimator = None
        if 'final_estimator_config' in config.params:
            meta_config = config.params.pop('final_estimator_config')
            if isinstance(meta_config, FactoryConfig):
                final_estimator = self.create(meta_config)
        
        if final_estimator is None:
            logger.info("Using default LogisticRegression as meta-learner for Stacking.")
            final_estimator = LogisticRegression(max_iter=1000, random_state=42)
            
        valid_params = self.validate_parameters(StackingClassifier, config.params)
        valid_params['estimators'] = estimators
        valid_params['final_estimator'] = final_estimator
        
        return StackingClassifier(**valid_params)