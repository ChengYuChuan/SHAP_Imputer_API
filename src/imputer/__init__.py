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
    - PatchGrouping: Group high-dimensional data into patches (NEW!)

Example (1D features):
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

Example (high-dimensional images):
    >>> import numpy as np
    >>> from imputer import BaselineImputer, CoalitionMatrix, PatchGrouping
    >>>
    >>> # Setup
    >>> image = np.random.randn(3, 224, 224)
    >>> reference = np.zeros((3, 224, 224))
    >>> 
    >>> # Group into patches
    >>> grouping = PatchGrouping(patch_size=(16, 16))
    >>> x_grouped = grouping.fit_transform(image)
    >>> ref_grouped = grouping.transform(reference)
    >>> 
    >>> # Coalition matrix operates on patches (not pixels!)
    >>> S = CoalitionMatrix(np.eye(196))  # 196 patches
    >>> 
    >>> # Impute on grouped data
    >>> imputer = BaselineImputer(reference=ref_grouped, x=x_grouped)
    >>> imputed_grouped = imputer.impute(S)
    >>> 
    >>> # Reconstruct to original shape
    >>> imputed_images = grouping.inverse_transform(imputed_grouped)
    >>> print(imputed_images.shape)  # (196, 3, 224, 224)

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
from imputer.grouping import FeatureGrouping, PatchGrouping

__all__ = [
    "__version__",
    # Core imputation
    "Imputer",
    "BaselineImputer",
    "MarginalImputer",
    "CoalitionMatrix",
    "baseline_impute",
    "marginal_impute",
    "compute_mean",
    # Feature grouping
    "FeatureGrouping",
    "PatchGrouping",
]