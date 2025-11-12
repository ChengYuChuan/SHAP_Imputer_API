"""Imputer: A multi-backend imputation library for model explanation.

This package provides efficient imputation strategies for explaining machine
learning models using game-theoretic concepts like Shapley values.

Supported backends:
    - NumPy (CPU)
    - PyTorch (GPU/CPU, autodiff)
    - JAX (TPU/GPU/CPU, XLA compilation)

Main classes:
    - Imputer: Abstract base class for imputation strategies
    - BaselineImputer: Replace missing features with reference values
    - MarginalImputer: Sample missing features from marginal distribution
    - CoalitionMatrix: Representation of feature coalitions

Example:
    >>> import numpy as np
    >>> from imputer import BaselineImputer, CoalitionMatrix
    >>>
    >>> # Setup
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> reference = np.zeros(4)
    >>> S = CoalitionMatrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0]]))
    >>>
    >>> # Create imputer
    >>> imputer = BaselineImputer(reference=reference, x=x)
    >>>
    >>> # Impute (without model prediction)
    >>> imputed = imputer.impute(S)
    >>> print(imputed)
    [[1. 2. 0. 0.]
     [1. 0. 3. 0.]]

For more examples, see the documentation at: https://imputer.readthedocs.io
"""

__version__ = "0.1.0"

from imputer.core import (
    BaselineImputer,
    CoalitionMatrix,
    Imputer,
    MarginalImputer,
    baseline_impute,
    compute_mean,
    marginal_impute,
)

__all__ = [
    "__version__",
    "Imputer",
    "BaselineImputer",
    "MarginalImputer",
    "CoalitionMatrix",
    "baseline_impute",
    "marginal_impute",
    "compute_mean",
]
