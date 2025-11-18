"""Core abstract base classes for the imputer package.

This module defines the fundamental interfaces for imputation strategies
that work across multiple ML frameworks (NumPy, PyTorch, JAX).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Type variables for generic typing
ArrayLike = TypeVar("ArrayLike", np.ndarray, "torch.Tensor", "jax.Array")
Predictions = TypeVar("Predictions")
Model = TypeVar("Model")

class ImputationStrategy(str, Enum):
    """Imputation strategies for BaselineImputer.
    
    Strategies:
        CONSTANT: Use fixed reference values
        MEAN: Use mean from background data
        MEDIAN: Use median from background data
        MODE: Use mode from background data (categorical features)
    """
    CONSTANT = "constant"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"

class CoalitionMatrix:
    """Represents a coalition matrix for game-theoretic model explanation.
    
    ⚠️ IMPORTANT: Coalition matrices MUST use boolean dtype.
    Numeric arrays (0/1) will be automatically converted, but this is 
    for backward compatibility only. Always use dtype=bool explicitly.
    
    Examples:
        # ✅ CORRECT
        S = CoalitionMatrix(np.array([[True, False]], dtype=bool))
        
        # ⚠️ DEPRECATED (auto-converted but discouraged)
        S = CoalitionMatrix(np.array([[1, 0]]))
    """
    
    def __init__(self, matrix: ArrayLike) -> None:
        """Initialize coalition matrix.
        
        Args:
            matrix: Binary matrix of shape (n_coalitions, n_features)
        
        Raises:
            ValueError: If matrix is not binary or has wrong dimensions
        """
        self.matrix = self._ensure_boolean(matrix)
        self._validate()
    def _ensure_boolean(self, matrix: ArrayLike) -> ArrayLike:
        '''convert to boolean dtype'''
        matrix_module = type(matrix).__module__.split('.')[0]
        
        if matrix_module == 'numpy':
            import numpy as np
            if matrix.dtype != np.bool_:
                return matrix.astype(np.bool_)
        
        elif matrix_module == 'torch':
            import torch
            if matrix.dtype != torch.bool:
                return matrix.to(torch.bool)
        
        elif matrix_module.startswith('jax'):
            import jax.numpy as jnp
            if matrix.dtype != jnp.bool_:
                return matrix.astype(jnp.bool_)
        
        return matrix
    
    def _validate(self) -> None:
        """Validate coalition matrix properties."""
        # make sure it's boolean dtype
        if not self._is_boolean_dtype(self.matrix):
            msg = (
                f"Coalition matrix must have boolean dtype, "
                f"got {self.matrix.dtype}. Use CoalitionMatrix constructor "
                f"for automatic conversion."
            )
            raise TypeError(msg)
        
        # make sure it's 2D
        if self.matrix.ndim != 2:
            msg = f"Coalition matrix must be 2D, got shape {self.matrix.shape}"
            raise ValueError(msg)

    def _is_boolean_dtype(self, matrix: ArrayLike) -> bool:
        """Check if matrix has boolean dtype across different backends."""
        matrix_module = type(matrix).__module__.split('.')[0]
        
        if matrix_module == 'numpy':
            import numpy as np
            return matrix.dtype == np.bool_
        elif matrix_module == 'torch':
            import torch
            return matrix.dtype == torch.bool
        elif matrix_module.startswith('jax'):
            import jax.numpy as jnp
            return matrix.dtype == jnp.bool_
        return False
    
    @property
    def n_coalitions(self) -> int:
        """Number of coalitions."""
        return self.matrix.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.matrix.shape[1]


class Imputer(ABC, Generic[ArrayLike, Predictions, Model]):
    """Abstract base class for imputation strategies.
    
    This class defines the interface for imputing missing values in data
    for the purpose of explaining machine learning models. It supports
    multiple ML frameworks through lazy dispatch.
    
    The general workflow is:
    1. __call__: Main entry point
    2. impute: Fill missing values based on coalition matrix
    3. predict: Get model predictions on imputed data
    4. postprocess: Transform predictions (e.g., averaging)
    
    Attributes:
        model: The ML model to explain (optional)
        predict_fn: Custom prediction function (optional)
    """
    
    def __init__(
        self,
        model: Model | None = None,
        predict_fn: Callable[[Model, ArrayLike], Predictions] | None = None,
    ) -> None:
        """Initialize the imputer.
        
        Args:
            model: ML model to explain. Can be None if only imputation is needed.
            predict_fn: Custom prediction function. If None, uses default.
        """
        self.model = model
        self.predict_fn = predict_fn
    
    def __call__(self, S: CoalitionMatrix) -> Predictions:
        """Main entry point for imputation and prediction.
        
        Args:
            S: Coalition matrix specifying which features to keep/impute
        
        Returns:
            Processed predictions for each coalition
        
        Workflow:
            1. Impute missing values based on coalition matrix
            2. Get model predictions on imputed data
            3. Post-process predictions (e.g., averaging over multiple imputations)
        """
        imputed_data = self.impute(S)
        predictions = self.predict(imputed_data)
        outputs = self.postprocess(predictions)
        return outputs
    
    @abstractmethod
    def impute(self, S: CoalitionMatrix) -> ArrayLike:
        """Impute missing values based on coalition matrix.
        
        Args:
            S: Coalition matrix specifying which features to keep (1) 
               and which to impute (0)
        
        Returns:
            Imputed data array. Shape depends on implementation:
            - Baseline: (n_coalitions, n_features)
            - Marginal: (n_coalitions, n_samples, n_features)
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Must be implemented by subclass")
    
    def predict(self, x: ArrayLike) -> Predictions:
        """Get model predictions on imputed data.
        
        Args:
            x: Imputed data array
        
        Returns:
            Model predictions
        
        Raises:
            ValueError: If model is None and predict_fn is None
        """
        if self.model is None and self.predict_fn is None:
            msg = "Either model or predict_fn must be provided"
            raise ValueError(msg)
        
        if self.predict_fn is not None:
            return self.predict_fn(self.model, x)
        
        # Default prediction (assumes model has __call__ or predict method)
        if hasattr(self.model, "predict"):
            return self.model.predict(x)
        return self.model(x)
    
    def postprocess(self, predictions: Predictions) -> Predictions:
        """Post-process predictions.
        
        Default implementation returns predictions unchanged.
        Subclasses can override to implement averaging, aggregation, etc.
        
        Args:
            predictions: Raw model predictions
        
        Returns:
            Processed predictions
        """
        return predictions


class BaselineImputer(Imputer[ArrayLike, Predictions, Model]):
    """Baseline imputation with multiple strategies.
    
    Supports:
        - constant: fixed reference values
        - mean: feature-wise mean from data
        - median: feature-wise median from data
        - mode: feature-wise mode from data
    
    Examples:
        # Strategy 1: Constant (current behavior, backward compatible)
        >>> imputer = BaselineImputer(
        ...     strategy="constant",
        ...     reference=np.zeros(4),
        ...     x=x
        ... )
        
        # Strategy 2: Mean
        >>> imputer = BaselineImputer(
        ...     strategy="mean",
        ...     data=background_data,  # (n_samples, n_features)
        ...     x=x
        ... )
        
        # Strategy 3: Median
        >>> imputer = BaselineImputer(
        ...     strategy="median",
        ...     data=background_data,
        ...     x=x
        ... )
    """
    
    def __init__(
        self,
        strategy: ImputationStrategy | Literal["constant", "mean", "median", "mode"] = "constant",
        reference: ArrayLike | None = None,  # For CONSTANT strategy
        data: ArrayLike | None = None,       # For MEAN/MEDIAN/MODE strategies
        x: ArrayLike | None = None,
        model: Model | None = None,
        predict_fn: Callable[[Model, ArrayLike], Predictions] | None = None,
    ) -> None:
        """Initialize baseline imputer with strategy.
        
        Args:
            strategy: Imputation strategy
            reference: Reference values (required for "constant")
            data: Background data (required for "mean"/"median"/"mode")
            x: Data point to explain
            model: ML model to explain
            predict_fn: Custom prediction function
        
        Raises:
            ValueError: If strategy="constant" but reference=None
            ValueError: If strategy in ["mean","median","mode"] but data=None
        """
        super().__init__(model=model, predict_fn=predict_fn)
        # Normalize strategy
        if isinstance(strategy, str):
            strategy = ImputationStrategy(strategy)
        
        self.strategy = strategy
        self.x = x
        self._reference = reference
        self._data = data

        # Validate inputs
        self._validate_strategy_inputs()

    def _validate_strategy_inputs(self) -> None:
        """Validate that required inputs are provided for strategy."""
        if self.strategy == ImputationStrategy.CONSTANT:
            if self._reference is None:
                msg = (
                    "strategy='constant' requires 'reference' argument. "
                    "Provide fixed values for imputation."
                )
                raise ValueError(msg)
        else:  # MEAN, MEDIAN, MODE
            if self._data is None:
                msg = (
                    f"strategy='{self.strategy.value}' requires 'data' argument. "
                    "Provide background dataset to compute statistics."
                )
                raise ValueError(msg)
        
    @property
    def reference(self) -> ArrayLike:
        """Get reference values based on strategy.
        
        Returns:
            Reference values for imputation
        
        Raises:
            ValueError: If data is not provided for statistical strategies
        """
        if self.strategy == ImputationStrategy.CONSTANT:
            return self._reference
        
        # Compute statistics from data
        if self._data is None:
            msg = f"Data must be provided for strategy '{self.strategy.value}'"
            raise ValueError(msg)
        
        if self.strategy == ImputationStrategy.MEAN:
            from imputer.core.implementations import compute_mean
            return compute_mean(self._data, axis=0)
        
        elif self.strategy == ImputationStrategy.MEDIAN:
            from imputer.core.implementations import compute_median
            return compute_median(self._data, axis=0)
        
        elif self.strategy == ImputationStrategy.MODE:
            from imputer.core.implementations import compute_mode
            return compute_mode(self._data, axis=0)
        
        else:
            msg = f"Unknown strategy: {self.strategy}"
            raise ValueError(msg)
    
    def impute(self, S: CoalitionMatrix) -> ArrayLike:
        """Impute using baseline strategy.
        
        Args:
            S: Coalition matrix (n_coalitions, n_features)
        
        Returns:
            Imputed data (n_coalitions, n_features)
        """
        if self.x is None:
            msg = "Data point x must be set before imputation"
            raise ValueError(msg)
        
        # Get reference based on strategy
        ref = self.reference
        
        # Delegate to backend-specific implementation
        from imputer.core.implementations import baseline_impute
        
        return baseline_impute(self.x, ref, S.matrix)


class MarginalImputer(Imputer[ArrayLike, Predictions, Model]):
    """Marginal imputation strategy.
    
    Samples missing features from their marginal distribution in the data.
    This approach better captures feature distributions but is more 
    computationally expensive.
    
    Reference:
        Aas et al. (2021): "Explaining individual predictions when features are dependent"
        Section on "Marginal imputation"
    
    Attributes:
        data: Background dataset for sampling (n_samples, n_features)
        x: Data point to explain (n_features,)
        n_samples: Number of samples to draw for each coalition
    """
    
    def __init__(
        self,
        data: ArrayLike,
        x: ArrayLike | None = None,
        model: Model | None = None,
        predict_fn: Callable[[Model, ArrayLike], Predictions] | None = None,
        n_samples: int = 100,
    ) -> None:
        """Initialize marginal imputer.
        
        Args:
            data: Background dataset (n_samples, n_features)
            x: Data point to explain (n_features,)
            model: ML model to explain
            predict_fn: Custom prediction function
            n_samples: Number of samples per coalition
        """
        super().__init__(model=model, predict_fn=predict_fn)
        self.data = data
        self.x = x
        self.n_samples = n_samples
    
    def impute(self, S: CoalitionMatrix) -> ArrayLike:
        """Impute using marginal strategy.
        
        For each coalition, keeps features where S[i,j]=1 from x,
        and samples features where S[i,j]=0 from the data distribution.
        
        Args:
            S: Coalition matrix (n_coalitions, n_features)
        
        Returns:
            Imputed data (n_coalitions, n_samples, n_features)
        
        Raises:
            ValueError: If x is not set
        """
        if self.x is None:
            msg = "Data point x must be set before imputation"
            raise ValueError(msg)
        
        # Delegate to backend-specific implementation via lazy dispatch
        from imputer.core.implementations import marginal_impute
        
        return marginal_impute(self.x, self.data, S.matrix, self.n_samples)
    
    def postprocess(self, predictions: Predictions) -> Predictions:
        """Average predictions over multiple samples.
        
        Since marginal imputation produces multiple samples per coalition,
        we average the predictions.
        
        Args:
            predictions: Predictions of shape (n_coalitions, n_samples, ...)
        
        Returns:
            Averaged predictions of shape (n_coalitions, ...)
        """
        from imputer.core.implementations import compute_mean
        
        # Average over the samples dimension (axis=1)
        return compute_mean(predictions, axis=1)
