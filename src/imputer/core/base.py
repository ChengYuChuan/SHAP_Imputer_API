"""Core abstract base classes for the imputer package.

This module defines the fundamental interfaces for imputation strategies
that work across multiple ML frameworks (NumPy, PyTorch, JAX).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Type variables for generic typing
ArrayLike = TypeVar("ArrayLike", np.ndarray, "torch.Tensor", "jax.Array")
Predictions = TypeVar("Predictions")
Model = TypeVar("Model")


class CoalitionMatrix:
    """Represents a coalition matrix for game-theoretic model explanation.
    
    A coalition matrix S is a binary matrix where:
    - Rows represent different coalitions (subsets of features)
    - Columns represent features
    - S[i,j] = 1 if feature j is in coalition i, 0 otherwise
    
    Attributes:
        matrix: The underlying binary matrix (n_coalitions, n_features)
        n_coalitions: Number of coalitions
        n_features: Number of features
    """
    
    def __init__(self, matrix: ArrayLike) -> None:
        """Initialize coalition matrix.
        
        Args:
            matrix: Binary matrix of shape (n_coalitions, n_features)
        
        Raises:
            ValueError: If matrix is not binary or has wrong dimensions
        """
        self.matrix = matrix
        self._validate()
    
    def _validate(self) -> None:
        """Validate coalition matrix properties."""
        # Implementation depends on backend
        pass
    
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
    """Baseline imputation strategy.
    
    Replaces missing features with values from a reference point.
    This is the simplest imputation strategy, often using a mean or 
    a specific data point as the reference.
    
    Reference:
        Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
        Section on "Baseline values"
    
    Attributes:
        reference: Reference values to use for imputation (n_features,)
        x: Data point to explain (n_features,)
    """
    
    def __init__(
        self,
        reference: ArrayLike,
        x: ArrayLike | None = None,
        model: Model | None = None,
        predict_fn: Callable[[Model, ArrayLike], Predictions] | None = None,
    ) -> None:
        """Initialize baseline imputer.
        
        Args:
            reference: Reference values for imputation (n_features,)
            x: Data point to explain (n_features,). Can be set later.
            model: ML model to explain
            predict_fn: Custom prediction function
        """
        super().__init__(model=model, predict_fn=predict_fn)
        self.reference = reference
        self.x = x
    
    def impute(self, S: CoalitionMatrix) -> ArrayLike:
        """Impute using baseline strategy.
        
        For each coalition, keeps features where S[i,j]=1 from x,
        and replaces features where S[i,j]=0 with reference values.
        
        Args:
            S: Coalition matrix (n_coalitions, n_features)
        
        Returns:
            Imputed data (n_coalitions, n_features)
        
        Raises:
            ValueError: If x is not set
        """
        if self.x is None:
            msg = "Data point x must be set before imputation"
            raise ValueError(msg)
        
        # Delegate to backend-specific implementation via lazy dispatch
        from imputer.core.implementations import baseline_impute
        
        return baseline_impute(self.x, self.reference, S.matrix)


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
