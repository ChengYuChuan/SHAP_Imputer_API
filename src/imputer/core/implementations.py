"""Backend-agnostic implementations using lazy dispatch.

This module provides the core imputation functions that work across
NumPy, PyTorch, and JAX through lazy dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lazy_dispatch as ld

if TYPE_CHECKING:
    from collections.abc import Callable


# ==================== Baseline Imputation ====================

@ld.lazydispatch
def baseline_impute(
    x: object,
    reference: object,
    coalition_matrix: object,
) -> object:
    """Impute using baseline strategy (abstract dispatcher).
    
    Args:
        x: Data point to explain (n_features,)
        reference: Reference values (n_features,)
        coalition_matrix: Binary matrix (n_coalitions, n_features)
            where 1 = keep original, 0 = impute with reference
    
    Returns:
        Imputed data (n_coalitions, n_features)
    
    Raises:
        NotImplementedError: If backend not registered
    """
    msg = (
        f"baseline_impute not implemented for type {type(x).__name__}. "
        "Supported types: numpy.ndarray, torch.Tensor, jax.Array"
    )
    raise NotImplementedError(msg)


# ==================== Marginal Imputation ====================

@ld.lazydispatch
def marginal_impute(
    x: object,
    data: object,
    coalition_matrix: object,
    n_samples: int,
) -> object:
    """Impute using marginal strategy (abstract dispatcher).
    
    Args:
        x: Data point to explain (n_features,)
        data: Background dataset (n_background, n_features)
        coalition_matrix: Binary matrix (n_coalitions, n_features)
        n_samples: Number of samples to draw per coalition
    
    Returns:
        Imputed data (n_coalitions, n_samples, n_features)
    
    Raises:
        NotImplementedError: If backend not registered
    """
    msg = (
        f"marginal_impute not implemented for type {type(x).__name__}. "
        "Supported types: numpy.ndarray, torch.Tensor, jax.Array"
    )
    raise NotImplementedError(msg)


# ==================== Utility Functions ====================

@ld.lazydispatch
def compute_mean(data: object, axis: int) -> object:
    """Compute mean along specified axis (abstract dispatcher).
    
    Args:
        data: Input array
        axis: Axis along which to compute mean
    
    Returns:
        Mean values
    
    Raises:
        NotImplementedError: If backend not registered
    """
    msg = f"compute_mean not implemented for type {type(data).__name__}"
    raise NotImplementedError(msg)


@ld.lazydispatch
def sample_indices(
    data: object,
    n_samples: int,
) -> object:
    """Sample random indices (abstract dispatcher).
    
    Args:
        data: Dataset to sample from
        n_samples: Number of indices to sample
    
    Returns:
        Random indices
    
    Raises:
        NotImplementedError: If backend not registered
    """
    msg = f"sample_indices not implemented for type {type(data).__name__}"
    raise NotImplementedError(msg)


# ==================== Delayed Registration for Optional Dependencies ====================

# Register implementations for torch (delayed until torch is imported)
@baseline_impute.delayed_register("torch.Tensor")
def _register_torch_baseline(cls: type) -> None:
    """Register torch baseline implementation when torch.Tensor is encountered."""
    from imputer.core.backends.torch import baseline_impute_torch
    baseline_impute.eager_register(cls, baseline_impute_torch)


@marginal_impute.delayed_register("torch.Tensor")
def _register_torch_marginal(cls: type) -> None:
    """Register torch marginal implementation when torch.Tensor is encountered."""
    from imputer.core.backends.torch import marginal_impute_torch
    marginal_impute.eager_register(cls, marginal_impute_torch)


@compute_mean.delayed_register("torch.Tensor")
def _register_torch_mean(cls: type) -> None:
    """Register torch mean computation when torch.Tensor is encountered."""
    from imputer.core.backends.torch import compute_mean_torch
    compute_mean.eager_register(cls, compute_mean_torch)


# Register implementations for JAX (delayed until jax.Array is encountered)
@baseline_impute.delayed_register("jax.Array")
def _register_jax_baseline(cls: type) -> None:
    """Register JAX baseline implementation when jax.Array is encountered."""
    from imputer.core.backends.jax import baseline_impute_jax
    baseline_impute.eager_register(cls, baseline_impute_jax)


@marginal_impute.delayed_register("jax.Array")
def _register_jax_marginal(cls: type) -> None:
    """Register JAX marginal implementation when jax.Array is encountered."""
    from imputer.core.backends.jax import marginal_impute_jax
    marginal_impute.eager_register(cls, marginal_impute_jax)


@compute_mean.delayed_register("jax.Array")
def _register_jax_mean(cls: type) -> None:
    """Register JAX mean computation when jax.Array is encountered."""
    from imputer.core.backends.jax import compute_mean_jax
    compute_mean.eager_register(cls, compute_mean_jax)
