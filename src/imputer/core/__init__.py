"""Core imputation module.

This module provides the main API for the imputer package, including:
- Abstract base classes (Imputer, BaselineImputer, MarginalImputer)
- CoalitionMatrix data structure
- Backend-agnostic implementations via lazy dispatch
"""

from imputer.core.base import (
    BaselineImputer,
    CoalitionMatrix,
    Imputer,
    MarginalImputer,
)
from imputer.core.implementations import (
    baseline_impute,
    compute_mean,
    marginal_impute,
)

# Import backends to register implementations
from imputer.core import backends  # noqa: F401

__all__ = [
    "Imputer",
    "BaselineImputer",
    "MarginalImputer",
    "CoalitionMatrix",
    "baseline_impute",
    "marginal_impute",
    "compute_mean",
]
