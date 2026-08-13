"""
data_preprocessor.py
=====================
Production data-preprocessing module for the AI-Based Real-Time Fault
Detection, Classification, and Protection System.

Consumes the flat, ML-ready feature dataset produced by the upstream
Feature Extraction Engine (`protect90_features_ml_ready.parquet`:
9022 rows x 202 columns, targets `sc_type`, `fault_target`,
`phase_select`) and produces train/test-ready, scaled, encoded numpy
arrays plus every fitted artifact (imputers, encoders, scaler) needed
to reproduce the exact same transformation at inference time on the
ATM90E32AS deployment path.

Design rules
------------
- Every stateful transform (imputers, one-hot encoder, label encoders,
  scaler, dropped-constant-column list) is FIT ONLY on training data
  inside `fit_transform()`, then reused verbatim by `transform()` at
  inference time. This is what guarantees "the inference pipeline
  uses the EXACT SAME feature schema produced by the feature
  extraction engine" end to end, not just at training time.
- Nothing is a module-level mutable global. All behaviour is driven by
  the frozen-by-convention `PreprocessingConfig` dataclass, constructed
  once and passed explicitly.
- Target columns, id columns, and diagnostic columns are excluded from
  every feature-side transform (constant-column dropping, missing
  value imputation, outlier handling, scaling) so they are never
  accidentally leaked into the feature matrix.
- All persisted artifacts are versionable, plain `joblib` files plus
  one human-readable JSON manifest, so `save_load.py` can wrap this in
  a versioning/rollback scheme without touching this module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.model_selection import train_test_split as sk_train_test_split

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ======================================================================
# Exceptions
# ======================================================================

class DataPreprocessingError(Exception):
    """Base exception for all preprocessing failures."""


class DataValidationError(DataPreprocessingError):
    """Raised when the input DataFrame fails structural/content checks."""


class PreprocessorStateError(DataPreprocessingError):
    """Raised when `transform()` is called before `fit_transform()`/`load()`,
    or when a saved artifact directory is missing required files."""


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class PreprocessingConfig:
    """Single source of truth for the preprocessing pipeline.

    Attributes
    ----------
    target_columns:
        Ground-truth label columns. Never scaled, imputed as features,
        or included in outlier/constant-column logic. Each gets its
        own `LabelEncoder`.
    id_columns:
        Identifier columns (e.g. `episode_id`) carried through for
        traceability but excluded from the feature matrix entirely.
    columns_to_drop:
        Diagnostic / non-feature columns from the extraction engine
        (e.g. `_processing_error`, `_episode_valid`) that must never
        reach the model.
    categorical_feature_columns:
        Feature columns that are categorical (e.g. `protection_zone`)
        and must be one-hot encoded rather than scaled as numeric.
        Auto-detected (object/category dtype) if left empty, minus any
        column explicitly listed in `numeric_override_columns`.
    numeric_override_columns:
        Columns that would otherwise be auto-detected as categorical
        (e.g. a numeric column with dtype accidentally read as object)
        but must be forced to numeric. Rare; provided as an escape
        hatch so auto-detection never silently misclassifies a feature.
    missing_value_strategy:
        Imputation strategy for numeric feature columns:
        "median" | "mean" | "most_frequent" | "constant".
    missing_value_fill:
        Fill value used only when `missing_value_strategy == "constant"`.
    missing_row_drop_threshold:
        Rows with a fraction of missing feature values strictly greater
        than this threshold are dropped entirely (both features and
        targets) BEFORE imputation. Set to 1.0 to disable row dropping.
    outlier_method:
        "zscore" | "iqr" | "none".
    outlier_zscore_threshold:
        Absolute z-score above which a numeric value is considered an
        outlier (only used when `outlier_method == "zscore"`).
    outlier_iqr_multiplier:
        Multiplier on the IQR for the Tukey fence (only used when
        `outlier_method == "iqr"`).
    outlier_strategy:
        "clip" (winsorize to the fence, keeps every row -- the safe
        default for a labelled, class-balanced dataset) or "remove"
        (drop offending rows entirely).
    scaling_method:
        "standard" | "minmax" | "robust" | "none".
    drop_constant_columns:
        If True, feature columns with <= `constant_column_unique_threshold`
        unique values (computed on the TRAIN split only) are dropped.
    constant_column_unique_threshold:
        Number of unique values at/below which a column counts as
        constant. 1 means "exactly one distinct value".
    test_size, random_state:
        Passed straight to `sklearn.model_selection.train_test_split`.
    stratify_target:
        Which target column to stratify the split on. If None, no
        stratification. If the requested stratification is infeasible
        (a class with a single member), the split automatically falls
        back to unstratified and logs a warning rather than raising.
    artifacts_dir:
        Directory every `save()` call writes to and every `load()`
        call reads from.
    """

    target_columns: tuple[str, ...] = ("sc_type", "fault_target", "phase_select")
    id_columns: tuple[str, ...] = ("episode_id",)
    columns_to_drop: tuple[str, ...] = ("_processing_error", "_episode_valid")

    categorical_feature_columns: tuple[str, ...] = field(default_factory=tuple)
    numeric_override_columns: tuple[str, ...] = field(default_factory=tuple)

    missing_value_strategy: str = "median"
    missing_value_fill: float = 0.0
    missing_row_drop_threshold: float = 0.5

    outlier_method: str = "zscore"
    outlier_zscore_threshold: float = 6.0
    outlier_iqr_multiplier: float = 3.0
    outlier_strategy: str = "clip"

    scaling_method: str = "standard"

    drop_constant_columns: bool = True
    constant_column_unique_threshold: int = 1

    test_size: float = 0.2
    random_state: int = 42
    stratify_target: Optional[str] = "fault_target"

    artifacts_dir: Path = Path("models/artifacts/preprocessing")
    scaler_filename: str = "scaler.joblib"
    numeric_imputer_filename: str = "numeric_imputer.joblib"
    categorical_imputer_filename: str = "categorical_imputer.joblib"
    onehot_encoder_filename: str = "onehot_encoder.joblib"
    label_encoder_filename_template: str = "label_encoder_{target}.joblib"
    manifest_filename: str = "feature_manifest.json"

    def __post_init__(self) -> None:
        valid_missing = {"median", "mean", "most_frequent", "constant"}
        if self.missing_value_strategy not in valid_missing:
            raise ValueError(
                f"missing_value_strategy must be one of {valid_missing}, "
                f"got {self.missing_value_strategy!r}."
            )
        valid_outlier_methods = {"zscore", "iqr", "none"}
        if self.outlier_method not in valid_outlier_methods:
            raise ValueError(
                f"outlier_method must be one of {valid_outlier_methods}, "
                f"got {self.outlier_method!r}."
            )
        valid_outlier_strategies = {"clip", "remove"}
        if self.outlier_strategy not in valid_outlier_strategies:
            raise ValueError(
                f"outlier_strategy must be one of {valid_outlier_strategies}, "
                f"got {self.outlier_strategy!r}."
            )
        valid_scaling = {"standard", "minmax", "robust", "none"}
        if self.scaling_method not in valid_scaling:
            raise ValueError(
                f"scaling_method must be one of {valid_scaling}, "
                f"got {self.scaling_method!r}."
            )
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}.")
        if not (0.0 <= self.missing_row_drop_threshold <= 1.0):
            raise ValueError(
                "missing_row_drop_threshold must be in [0, 1], got "
                f"{self.missing_row_drop_threshold}."
            )
        if self.stratify_target is not None and self.stratify_target not in self.target_columns:
            raise ValueError(
                f"stratify_target={self.stratify_target!r} must be one of "
                f"target_columns={self.target_columns!r} or None."
            )
        overlap = set(self.categorical_feature_columns) & set(self.numeric_override_columns)
        if overlap:
            raise ValueError(
                "categorical_feature_columns and numeric_override_columns "
                f"must be disjoint; overlap: {sorted(overlap)}"
            )
        self.artifacts_dir = Path(self.artifacts_dir)


# ======================================================================
# Result container
# ======================================================================

@dataclass
class PreprocessedData:
    """Everything downstream training code needs, in one object."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: dict[str, np.ndarray]
    y_test: dict[str, np.ndarray]
    feature_names: list[str]
    train_ids: pd.Series
    test_ids: pd.Series

    def target_names(self) -> list[str]:
        return list(self.y_train.keys())

    def summary(self) -> dict[str, Any]:
        return {
            "n_train": int(self.X_train.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "n_features": int(self.X_train.shape[1]),
            "targets": self.target_names(),
        }


# ======================================================================
# Preprocessor
# ======================================================================

class DataPreprocessor:
    """Fits every preprocessing artifact on training data and applies it
    consistently to both the held-out test split and, later, to brand
    new single-row inference requests coming from the ATM90E32AS.

    Typical usage
    -------------
    Training time::

        config = PreprocessingConfig()
        pre = DataPreprocessor(config)
        data = pre.fit_transform(df)
        pre.save(config.artifacts_dir)

    Inference time::

        pre = DataPreprocessor.load(config.artifacts_dir, config)
        X = pre.transform(new_row_df)
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config: PreprocessingConfig = config or PreprocessingConfig()

        # Fitted state (populated by fit_transform / load)
        self._is_fitted: bool = False
        self.feature_columns_: list[str] = []
        self.numeric_feature_columns_: list[str] = []
        self.categorical_feature_columns_: list[str] = []
        self.dropped_constant_columns_: list[str] = []
        self.numeric_imputer_: Optional[SimpleImputer] = None
        self.categorical_imputer_: Optional[SimpleImputer] = None
        self.onehot_encoder_: Optional[OneHotEncoder] = None
        self.onehot_output_columns_: list[str] = []
        self.label_encoders_: dict[str, LabelEncoder] = {}
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self.final_feature_names_: list[str] = []
        self._outlier_bounds_: dict[str, tuple[float, float]] = {}

        logger.info(
            "DataPreprocessor initialized (missing=%s, outlier=%s/%s, "
            "scaling=%s, test_size=%.2f).",
            self.config.missing_value_strategy,
            self.config.outlier_method,
            self.config.outlier_strategy,
            self.config.scaling_method,
            self.config.test_size,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame, *, require_targets: bool = True) -> None:
        """Structural validation of the incoming dataset.

        Raises
        ------
        DataValidationError
            If the DataFrame is empty, missing required id/target
            columns, entirely duplicated, or contains non-finite
            values in a way that cannot be safely handled downstream.
        """
        if df is None or df.empty:
            raise DataValidationError("Input DataFrame is empty or None.")

        if require_targets:
            missing_targets = [c for c in self.config.target_columns if c not in df.columns]
            if missing_targets:
                raise DataValidationError(
                    f"Missing required target column(s): {missing_targets}. "
                    f"Available columns: {df.columns.tolist()}"
                )

        n_duplicates = int(df.duplicated().sum())
        if n_duplicates > 0:
            logger.warning(
                "Input DataFrame contains %d fully duplicated row(s); "
                "they will pass through unless dropped upstream.",
                n_duplicates,
            )

        n_inf = int(
            np.isinf(df.select_dtypes(include=[np.number]).to_numpy(dtype=float)).sum()
        ) if not df.select_dtypes(include=[np.number]).empty else 0
        if n_inf > 0:
            logger.warning(
                "Input DataFrame contains %d infinite numeric value(s); "
                "these will be treated as missing during imputation.",
                n_inf,
            )

        logger.info(
            "Validation passed: shape=%s, duplicates=%d, infinite_values=%d.",
            df.shape, n_duplicates, n_inf,
        )

    # ------------------------------------------------------------------
    # Column bookkeeping
    # ------------------------------------------------------------------

    def _non_feature_columns(self) -> set[str]:
        return (
            set(self.config.target_columns)
            | set(self.config.id_columns)
            | set(self.config.columns_to_drop)
        )

    def _raw_feature_columns(self, df: pd.DataFrame) -> list[str]:
        excluded = self._non_feature_columns()
        return [c for c in df.columns if c not in excluded]

    def _split_feature_types(self, df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
        """Partition feature columns into (numeric, categorical).

        Auto-detection is dtype-based: object/category/pandas-string ->
        categorical; everything else, INCLUDING bool, -> numeric. bool
        is deliberately treated as numeric (a 0/1 flag such as a fault
        indicator or protection-zone hit/miss flag), not one-hot
        expanded, then overridden by the explicit
        `categorical_feature_columns` / `numeric_override_columns` lists
        in the config.

        bool columns routed here as "numeric" are NOT yet safe to hand
        to `SimpleImputer` -- scikit-learn's imputer explicitly rejects
        raw bool-dtype arrays regardless of strategy. `_coerce_bool_columns()`
        (called at the top of `_handle_missing_values`) is what actually
        converts them to an imputer-safe dtype; this method only decides
        *which* imputer (numeric vs. categorical) a column is routed to.
        """
        explicit_categorical = set(self.config.categorical_feature_columns)
        explicit_numeric = set(self.config.numeric_override_columns)

        categorical, numeric = [], []
        for col in feature_cols:
            if col in explicit_categorical:
                categorical.append(col)
                continue
            if col in explicit_numeric:
                numeric.append(col)
                continue
            dtype = df[col].dtype
            is_bool = pd.api.types.is_bool_dtype(dtype)
            is_numeric = is_bool or pd.api.types.is_numeric_dtype(dtype)
            # Covers legacy object dtype, pandas categorical dtype, AND
            # pandas >= 2.x/3.x's dedicated string dtype -- any of which
            # can hold e.g. "ZONE1"/"ZONE2"/"UNKNOWN" from protection_zone.
            if is_numeric:
                numeric.append(col)
            else:
                categorical.append(col)
        return numeric, categorical

    @staticmethod
    def _coerce_bool_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Convert every bool-dtype column in `cols` to an imputer-safe
        numeric dtype before it reaches `SimpleImputer`.

        `sklearn.impute.SimpleImputer` raises `ValueError: SimpleImputer
        does not support data with dtype bool` for ANY bool-dtype input,
        regardless of `strategy` ("median", "mean", "most_frequent", ...).
        This is true for plain numpy `bool` columns and for pandas'
        nullable `"boolean"` extension dtype alike.

        Two cases, handled distinctly:
          - Plain numpy bool (or a fully-populated nullable "boolean"
            column with no missing entries): safe to cast straight to
            `int8` -- compact, and both `SimpleImputer` and the scaler
            accept it without complaint.
          - Nullable "boolean" dtype WITH missing entries (`pd.NA`):
            `int8` cannot represent a missing value, so these are cast
            to `float64` instead, which preserves the missingness as
            `NaN` for the imputer to actually impute. Casting straight
            to `int8` here would silently corrupt or crash on the NA
            entries rather than impute them.
        """
        bool_cols = [c for c in cols if c in df.columns and pd.api.types.is_bool_dtype(df[c])]
        if not bool_cols:
            return df
        df = df.copy()
        for col in bool_cols:
            if df[col].isna().any():
                df[col] = df[col].astype("float64")
            else:
                df[col] = df[col].astype("int8")
        return df

    # ------------------------------------------------------------------
    # Step 1: drop constant columns
    # ------------------------------------------------------------------

    def _drop_constant_columns(self, df: pd.DataFrame, feature_cols: list[str], *, fit: bool) -> list[str]:
        if not self.config.drop_constant_columns:
            return feature_cols

        if fit:
            dropped = [
                col for col in feature_cols
                if df[col].nunique(dropna=True) <= self.config.constant_column_unique_threshold
            ]
            self.dropped_constant_columns_ = dropped
            if dropped:
                logger.info("Dropping %d constant feature column(s): %s", len(dropped), dropped)
            else:
                logger.info("No constant feature columns found.")

        remaining = [c for c in feature_cols if c not in self.dropped_constant_columns_]
        return remaining

    # ------------------------------------------------------------------
    # Step 2: missing values
    # ------------------------------------------------------------------

    def _drop_high_missing_rows(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        threshold = self.config.missing_row_drop_threshold
        if threshold >= 1.0 or not feature_cols:
            return df

        missing_fraction = df[feature_cols].isna().mean(axis=1)
        keep_mask = missing_fraction <= threshold
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            logger.warning(
                "Dropping %d row(s) with >%.0f%% missing feature values.",
                n_dropped, threshold * 100,
            )
        return df.loc[keep_mask].copy()

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        *,
        fit: bool,
    ) -> pd.DataFrame:
        df = df.copy()

        # Bool-dtype numeric feature columns (e.g. fault-indicator flags)
        # must be coerced to int8/float64 BEFORE they reach SimpleImputer,
        # which rejects raw bool arrays outright. Applied identically here
        # on both the fit (training) and transform (inference) paths, so
        # a boolean feature is imputed the same way regardless of source.
        df = self._coerce_bool_columns(df, numeric_cols)

        # Non-finite (inf/-inf) numeric values are treated as missing.
        if numeric_cols:
            numeric_block = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

            strategy = self.config.missing_value_strategy
            imputer_kwargs: dict[str, Any] = {"strategy": strategy}
            if strategy == "constant":
                imputer_kwargs["fill_value"] = self.config.missing_value_fill

            if fit:
                n_missing = int(numeric_block.isna().sum().sum())
                self.numeric_imputer_ = SimpleImputer(**imputer_kwargs)
                imputed = self.numeric_imputer_.fit_transform(numeric_block)
                logger.info(
                    "Numeric imputation fitted (strategy=%s); %d missing "
                    "value(s) in training data.", strategy, n_missing,
                )
            else:
                if self.numeric_imputer_ is None:
                    raise PreprocessorStateError(
                        "Numeric imputer not fitted; call fit_transform() or load() first."
                    )
                imputed = self.numeric_imputer_.transform(numeric_block)

            df[numeric_cols] = imputed

        if categorical_cols:
            # A bool column can only land in `categorical_cols` via an
            # explicit `categorical_feature_columns` override (auto-
            # detection now always routes bool to `numeric_cols`; see
            # `_split_feature_types`). SimpleImputer rejects raw bool
            # dtype here too, regardless of strategy, so the same
            # dtype hazard applies -- cast to `object` (preserving True/
            # False as two distinct categories) rather than `int8`,
            # since the caller explicitly asked for one-hot categorical
            # treatment, not numeric treatment, for this column.
            bool_cat_cols = [c for c in categorical_cols if pd.api.types.is_bool_dtype(df[c])]
            if bool_cat_cols:
                df[bool_cat_cols] = df[bool_cat_cols].astype(object)
                df[bool_cat_cols] = df[bool_cat_cols].where(df[bool_cat_cols].notna(), np.nan)

            categorical_block = df[categorical_cols]
            if fit:
                n_missing = int(categorical_block.isna().sum().sum())
                self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
                imputed_cat = self.categorical_imputer_.fit_transform(categorical_block)
                logger.info(
                    "Categorical imputation fitted (most_frequent); %d "
                    "missing value(s) in training data.", n_missing,
                )
            else:
                if self.categorical_imputer_ is None:
                    raise PreprocessorStateError(
                        "Categorical imputer not fitted; call fit_transform() or load() first."
                    )
                imputed_cat = self.categorical_imputer_.transform(categorical_block)

            df[categorical_cols] = imputed_cat

        return df

    # ------------------------------------------------------------------
    # Step 3: outlier handling (numeric features only, fit on train)
    # ------------------------------------------------------------------

    def _compute_outlier_bounds(self, df: pd.DataFrame, numeric_cols: list[str]) -> None:
        bounds: dict[str, tuple[float, float]] = {}
        if self.config.outlier_method == "zscore":
            means = df[numeric_cols].mean()
            stds = df[numeric_cols].std(ddof=0).replace(0.0, np.nan)
            t = self.config.outlier_zscore_threshold
            for col in numeric_cols:
                mean, std = means[col], stds[col]
                if pd.isna(std):
                    bounds[col] = (float(df[col].min()), float(df[col].max()))
                else:
                    bounds[col] = (float(mean - t * std), float(mean + t * std))
        elif self.config.outlier_method == "iqr":
            q1 = df[numeric_cols].quantile(0.25)
            q3 = df[numeric_cols].quantile(0.75)
            iqr = (q3 - q1).replace(0.0, np.nan)
            m = self.config.outlier_iqr_multiplier
            for col in numeric_cols:
                if pd.isna(iqr[col]):
                    bounds[col] = (float(df[col].min()), float(df[col].max()))
                else:
                    bounds[col] = (float(q1[col] - m * iqr[col]), float(q3[col] + m * iqr[col]))
        self._outlier_bounds_ = bounds

    def _apply_outlier_handling(self, df: pd.DataFrame, numeric_cols: list[str], *, fit: bool) -> pd.DataFrame:
        if self.config.outlier_method == "none" or not numeric_cols:
            return df

        df = df.copy()
        if fit:
            self._compute_outlier_bounds(df, numeric_cols)

        if not self._outlier_bounds_:
            raise PreprocessorStateError(
                "Outlier bounds not fitted; call fit_transform() or load() first."
            )

        if self.config.outlier_strategy == "clip":
            n_clipped = 0
            for col in numeric_cols:
                if col not in self._outlier_bounds_:
                    continue
                lower, upper = self._outlier_bounds_[col]
                out_of_range = ((df[col] < lower) | (df[col] > upper)).sum()
                n_clipped += int(out_of_range)
                df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info("Outlier clipping applied: %d value(s) clipped.", n_clipped)
        else:  # "remove"
            mask = pd.Series(True, index=df.index)
            for col in numeric_cols:
                if col not in self._outlier_bounds_:
                    continue
                lower, upper = self._outlier_bounds_[col]
                mask &= df[col].between(lower, upper)
            n_removed = int((~mask).sum())
            logger.info("Outlier removal applied: %d row(s) dropped.", n_removed)
            df = df.loc[mask].copy()

        return df

    # ------------------------------------------------------------------
    # Step 4: categorical (one-hot) encoding of feature columns
    # ------------------------------------------------------------------

    def _encode_categorical_features(
        self, df: pd.DataFrame, categorical_cols: list[str], *, fit: bool
    ) -> pd.DataFrame:
        if not categorical_cols:
            # IMPORTANT: must return an EMPTY frame here, not `df` itself.
            # `df` at both call sites is `working_df`, which still carries
            # the target columns (sc_type/fault_target/phase_select), the
            # id column, and diagnostic columns alongside the features --
            # they're kept there only so `_encode_labels()` and `id_series`
            # can read them earlier in the pipeline. Returning `df`
            # unchanged (e.g. via `df.drop(columns=[], errors="ignore")`,
            # which drops nothing) would concatenate ALL of those
            # non-feature columns into `feature_df`, and the target
            # columns' string values would then blow up the
            # `.to_numpy(dtype=np.float64)` cast in `fit_transform()` /
            # `transform()` -- exactly the "could not convert string to
            # float" failure this method must never reintroduce.
            return pd.DataFrame(index=df.index)

        block = df[categorical_cols].astype(str)

        if fit:
            self.onehot_encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float64
            )
            encoded = self.onehot_encoder_.fit_transform(block)
            self.onehot_output_columns_ = list(
                self.onehot_encoder_.get_feature_names_out(categorical_cols)
            )
            logger.info(
                "One-hot encoding fitted on %d categorical column(s) -> "
                "%d output column(s).", len(categorical_cols), len(self.onehot_output_columns_),
            )
        else:
            if self.onehot_encoder_ is None:
                raise PreprocessorStateError(
                    "One-hot encoder not fitted; call fit_transform() or load() first."
                )
            encoded = self.onehot_encoder_.transform(block)

        encoded_df = pd.DataFrame(encoded, columns=self.onehot_output_columns_, index=df.index)
        return encoded_df

    # ------------------------------------------------------------------
    # Step 5: target label encoding
    # ------------------------------------------------------------------

    def _encode_labels(self, df: pd.DataFrame, *, fit: bool) -> dict[str, np.ndarray]:
        encoded_targets: dict[str, np.ndarray] = {}
        for target in self.config.target_columns:
            if target not in df.columns:
                raise DataValidationError(f"Target column {target!r} not found in DataFrame.")

            raw_values = df[target].astype(str)
            if fit:
                encoder = LabelEncoder()
                encoded_targets[target] = encoder.fit_transform(raw_values)
                self.label_encoders_[target] = encoder
                logger.info(
                    "LabelEncoder fitted for target %r: %d class(es) -> %s",
                    target, len(encoder.classes_), list(encoder.classes_),
                )
            else:
                encoder = self.label_encoders_.get(target)
                if encoder is None:
                    raise PreprocessorStateError(
                        f"Label encoder for target {target!r} not fitted; "
                        "call fit_transform() or load() first."
                    )
                unseen = set(raw_values.unique()) - set(encoder.classes_)
                if unseen:
                    raise DataValidationError(
                        f"Target {target!r} contains unseen label(s) at transform "
                        f"time: {sorted(unseen)}. Re-fit the preprocessor if these "
                        "are legitimate new classes."
                    )
                encoded_targets[target] = encoder.transform(raw_values)
        return encoded_targets

    def inverse_transform_labels(self, target: str, encoded_values: np.ndarray) -> np.ndarray:
        """Decode integer-encoded predictions back to their original string
        labels for a given target column. Used by `predict.py`."""
        encoder = self.label_encoders_.get(target)
        if encoder is None:
            raise PreprocessorStateError(
                f"Label encoder for target {target!r} not fitted; "
                "call fit_transform() or load() first."
            )
        return encoder.inverse_transform(np.asarray(encoded_values))

    # ------------------------------------------------------------------
    # Step 6: scaling
    # ------------------------------------------------------------------

    def _build_scaler(self) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        if self.config.scaling_method == "standard":
            return StandardScaler()
        if self.config.scaling_method == "minmax":
            return MinMaxScaler()
        if self.config.scaling_method == "robust":
            return RobustScaler()
        raise ValueError(f"Unsupported scaling_method: {self.config.scaling_method!r}")

    def _scale(self, X: pd.DataFrame, *, fit: bool) -> np.ndarray:
        if self.config.scaling_method == "none":
            return X.to_numpy(dtype=np.float64)

        if fit:
            self.scaler_ = self._build_scaler()
            values = self.scaler_.fit_transform(X)
            logger.info("Scaler fitted: %s on %d feature(s).", self.config.scaling_method, X.shape[1])
        else:
            if self.scaler_ is None:
                raise PreprocessorStateError(
                    "Scaler not fitted; call fit_transform() or load() first."
                )
            values = self.scaler_.transform(X)
        return np.asarray(values, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 7: train/test split
    # ------------------------------------------------------------------

    def _build_stratify_key(self, targets_encoded: dict[str, np.ndarray]) -> Optional[np.ndarray]:
        if self.config.stratify_target is None:
            return None
        key = targets_encoded.get(self.config.stratify_target)
        if key is None:
            logger.warning(
                "stratify_target=%r not found among encoded targets; "
                "falling back to unstratified split.", self.config.stratify_target,
            )
            return None
        _, counts = np.unique(key, return_counts=True)
        if counts.min() < 2:
            logger.warning(
                "Stratification target %r has a class with fewer than 2 "
                "members; falling back to unstratified split.",
                self.config.stratify_target,
            )
            return None
        return key

    def train_test_split(
        self,
        X: np.ndarray,
        targets_encoded: dict[str, np.ndarray],
        ids: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], pd.Series, pd.Series]:
        """Single stratified split shared consistently across every target
        column (all targets originate from the same row, so one split
        index set is reused for all of them)."""
        stratify_key = self._build_stratify_key(targets_encoded)
        target_names = list(targets_encoded.keys())
        stacked_targets = np.column_stack([targets_encoded[t] for t in target_names])

        indices = np.arange(X.shape[0])
        try:
            (
                X_train, X_test,
                y_train_stack, y_test_stack,
                idx_train, idx_test,
            ) = sk_train_test_split(
                X, stacked_targets, indices,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_key,
            )
        except ValueError as exc:
            logger.warning(
                "Stratified split failed (%s); retrying without stratification.", exc
            )
            (
                X_train, X_test,
                y_train_stack, y_test_stack,
                idx_train, idx_test,
            ) = sk_train_test_split(
                X, stacked_targets, indices,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=None,
            )

        y_train = {name: y_train_stack[:, i] for i, name in enumerate(target_names)}
        y_test = {name: y_test_stack[:, i] for i, name in enumerate(target_names)}
        train_ids = ids.iloc[idx_train].reset_index(drop=True)
        test_ids = ids.iloc[idx_test].reset_index(drop=True)

        logger.info(
            "Train/test split complete: %d train / %d test rows (test_size=%.2f).",
            X_train.shape[0], X_test.shape[0], self.config.test_size,
        )
        return X_train, X_test, y_train, y_test, train_ids, test_ids

    # ------------------------------------------------------------------
    # Orchestration: fit_transform (training time)
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> PreprocessedData:
        """Run the full preprocessing pipeline end to end and fit every
        stateful artifact on the resulting training split.

        Order of operations
        --------------------
        1. Validate.
        2. Determine feature columns (exclude targets/ids/diagnostics).
        3. Drop constant feature columns.
        4. Drop rows with excessive missing feature values.
        5. Impute remaining missing values (numeric + categorical).
        6. Remove/clip outliers on numeric features.
        7. Encode target labels (LabelEncoder per target).
        8. One-hot encode categorical feature columns.
        9. Assemble final feature matrix (numeric + one-hot columns).
        10. Train/test split (single split shared across all targets).
        11. Fit and apply the scaler on the training split only, then
            transform the test split with the same fitted scaler.
        """
        logger.info("Starting fit_transform on DataFrame of shape %s.", df.shape)
        self.validate(df, require_targets=True)

        id_series = (
            df[self.config.id_columns[0]]
            if self.config.id_columns and self.config.id_columns[0] in df.columns
            else pd.Series(np.arange(len(df)), name="row_index")
        )

        raw_feature_cols = self._raw_feature_columns(df)
        numeric_cols, categorical_cols = self._split_feature_types(df, raw_feature_cols)

        surviving_numeric = self._drop_constant_columns(df, numeric_cols, fit=True)
        surviving_categorical = [
            c for c in categorical_cols if c not in self.dropped_constant_columns_
        ]

        working_cols = surviving_numeric + surviving_categorical
        working_df = df[list(self._non_feature_columns() & set(df.columns)) + working_cols].copy()
        working_df = self._drop_high_missing_rows(working_df, working_cols)

        working_df = self._handle_missing_values(
            working_df, surviving_numeric, surviving_categorical, fit=True
        )
        working_df = self._apply_outlier_handling(working_df, surviving_numeric, fit=True)

        targets_encoded = self._encode_labels(working_df, fit=True)
        id_series = id_series.loc[working_df.index].reset_index(drop=True)

        numeric_block = working_df[surviving_numeric].reset_index(drop=True)
        categorical_encoded = self._encode_categorical_features(
            working_df, surviving_categorical, fit=True
        ).reset_index(drop=True)

        feature_df = pd.concat([numeric_block, categorical_encoded], axis=1)
        self.numeric_feature_columns_ = surviving_numeric
        self.categorical_feature_columns_ = surviving_categorical
        self.final_feature_names_ = list(feature_df.columns)
        self.feature_columns_ = self.final_feature_names_

        X_train_raw, X_test_raw, y_train, y_test, train_ids, test_ids = self.train_test_split(
            feature_df.to_numpy(dtype=np.float64), targets_encoded, id_series
        )

        X_train = self._scale(pd.DataFrame(X_train_raw, columns=self.final_feature_names_), fit=True)
        X_test = self._scale(pd.DataFrame(X_test_raw, columns=self.final_feature_names_), fit=False)

        self._is_fitted = True
        logger.info(
            "fit_transform complete: %d final feature(s), %d train row(s), %d test row(s).",
            len(self.final_feature_names_), X_train.shape[0], X_test.shape[0],
        )

        return PreprocessedData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self.final_feature_names_,
            train_ids=train_ids,
            test_ids=test_ids,
        )

    # ------------------------------------------------------------------
    # Orchestration: transform (inference time, no split/targets required)
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply every previously-fitted transform to new data (e.g. one
        row produced live by the ATM90E32AS via the feature extraction
        engine's `process_single_atm90e32as_reading`). Targets are not
        required and, if present, are ignored.

        Returns
        -------
        np.ndarray of shape (n_rows, n_final_features), column order
        identical to `self.final_feature_names_` / training time.
        """
        if not self._is_fitted:
            raise PreprocessorStateError(
                "DataPreprocessor is not fitted. Call fit_transform() first, "
                "or load() a previously saved preprocessor."
            )

        self.validate(df, require_targets=False)

        missing_cols = [
            c for c in self.numeric_feature_columns_ + self.categorical_feature_columns_
            if c not in df.columns
        ]
        if missing_cols:
            raise DataValidationError(
                f"Input is missing {len(missing_cols)} feature column(s) required "
                f"by the fitted preprocessor: {missing_cols}"
            )

        working_df = df[self.numeric_feature_columns_ + self.categorical_feature_columns_].copy()
        working_df = self._handle_missing_values(
            working_df, self.numeric_feature_columns_, self.categorical_feature_columns_, fit=False
        )
        working_df = self._apply_outlier_handling(working_df, self.numeric_feature_columns_, fit=False)

        numeric_block = working_df[self.numeric_feature_columns_].reset_index(drop=True)
        categorical_encoded = self._encode_categorical_features(
            working_df, self.categorical_feature_columns_, fit=False
        ).reset_index(drop=True)

        feature_df = pd.concat([numeric_block, categorical_encoded], axis=1)
        feature_df = feature_df.reindex(columns=self.final_feature_names_, fill_value=0.0)

        X = self._scale(feature_df, fit=False)
        return X

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Optional[Union[str, Path]] = None) -> Path:
        """Persist every fitted artifact plus a JSON manifest describing
        the exact feature schema, so `save_load.py` can version this
        directory and `predict.py` can reload it deterministically."""
        if not self._is_fitted:
            raise PreprocessorStateError("Cannot save an unfitted DataPreprocessor.")

        out_dir = Path(directory) if directory is not None else self.config.artifacts_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.scaler_ is not None:
            joblib.dump(self.scaler_, out_dir / self.config.scaler_filename)
        if self.numeric_imputer_ is not None:
            joblib.dump(self.numeric_imputer_, out_dir / self.config.numeric_imputer_filename)
        if self.categorical_imputer_ is not None:
            joblib.dump(self.categorical_imputer_, out_dir / self.config.categorical_imputer_filename)
        if self.onehot_encoder_ is not None:
            joblib.dump(self.onehot_encoder_, out_dir / self.config.onehot_encoder_filename)

        for target, encoder in self.label_encoders_.items():
            filename = self.config.label_encoder_filename_template.format(target=target)
            joblib.dump(encoder, out_dir / filename)

        manifest = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self.config).items()
            },
            "numeric_feature_columns": self.numeric_feature_columns_,
            "categorical_feature_columns": self.categorical_feature_columns_,
            "dropped_constant_columns": self.dropped_constant_columns_,
            "onehot_output_columns": self.onehot_output_columns_,
            "final_feature_names": self.final_feature_names_,
            "target_classes": {
                target: [str(c) for c in encoder.classes_]
                for target, encoder in self.label_encoders_.items()
            },
            "outlier_bounds": self._outlier_bounds_,
        }
        manifest_path = out_dir / self.config.manifest_filename
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)

        logger.info("DataPreprocessor artifacts saved to %s", out_dir)
        return out_dir

    @classmethod
    def load(
        cls, directory: Union[str, Path], config: Optional[PreprocessingConfig] = None
    ) -> "DataPreprocessor":
        """Reconstruct a fully-fitted `DataPreprocessor` from a directory
        previously written by `save()`. `config` may be supplied to
        override paths/filenames; if omitted, the manifest's persisted
        config is used to reconstruct one."""
        in_dir = Path(directory)
        manifest_filename = (config.manifest_filename if config else PreprocessingConfig().manifest_filename)
        manifest_path = in_dir / manifest_filename
        if not manifest_path.exists():
            raise PreprocessorStateError(
                f"No manifest found at {manifest_path}; cannot load DataPreprocessor."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if config is None:
            saved_cfg = dict(manifest["config"])
            saved_cfg["artifacts_dir"] = Path(saved_cfg["artifacts_dir"])
            saved_cfg["target_columns"] = tuple(saved_cfg["target_columns"])
            saved_cfg["id_columns"] = tuple(saved_cfg["id_columns"])
            saved_cfg["columns_to_drop"] = tuple(saved_cfg["columns_to_drop"])
            saved_cfg["categorical_feature_columns"] = tuple(saved_cfg["categorical_feature_columns"])
            saved_cfg["numeric_override_columns"] = tuple(saved_cfg["numeric_override_columns"])
            config = PreprocessingConfig(**saved_cfg)

        instance = cls(config)

        scaler_path = in_dir / config.scaler_filename
        if scaler_path.exists():
            instance.scaler_ = joblib.load(scaler_path)

        numeric_imputer_path = in_dir / config.numeric_imputer_filename
        if numeric_imputer_path.exists():
            instance.numeric_imputer_ = joblib.load(numeric_imputer_path)

        categorical_imputer_path = in_dir / config.categorical_imputer_filename
        if categorical_imputer_path.exists():
            instance.categorical_imputer_ = joblib.load(categorical_imputer_path)

        onehot_path = in_dir / config.onehot_encoder_filename
        if onehot_path.exists():
            instance.onehot_encoder_ = joblib.load(onehot_path)

        for target in config.target_columns:
            filename = config.label_encoder_filename_template.format(target=target)
            encoder_path = in_dir / filename
            if encoder_path.exists():
                instance.label_encoders_[target] = joblib.load(encoder_path)

        instance.numeric_feature_columns_ = manifest["numeric_feature_columns"]
        instance.categorical_feature_columns_ = manifest["categorical_feature_columns"]
        instance.dropped_constant_columns_ = manifest["dropped_constant_columns"]
        instance.onehot_output_columns_ = manifest["onehot_output_columns"]
        instance.final_feature_names_ = manifest["final_feature_names"]
        instance.feature_columns_ = instance.final_feature_names_
        instance._outlier_bounds_ = {
            k: tuple(v) for k, v in manifest.get("outlier_bounds", {}).items()
        }
        instance._is_fitted = True

        logger.info(
            "DataPreprocessor loaded from %s (%d final feature(s), %d target(s)).",
            in_dir, len(instance.final_feature_names_), len(instance.label_encoders_),
        )
        return instance