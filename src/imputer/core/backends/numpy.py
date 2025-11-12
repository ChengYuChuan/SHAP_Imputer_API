"""NumPy backend implementations for imputation.

This module provides NumPy-specific implementations of imputation strategies.
NumPy is always available as it's a core dependency.
"""

from __future__ import annotations

import numpy as np

from imputer.core.implementations import (
    baseline_impute,
    compute_mean,
    marginal_impute,
)


@baseline_impute.register(np.ndarray)
def baseline_impute_numpy(
    x: np.ndarray,
    reference: np.ndarray,
    coalition_matrix: np.ndarray,
) -> np.ndarray:
    """Baseline imputation for NumPy arrays.
    
    Args:
        x: Data point to explain, shape (n_features,)
        reference: Reference values, shape (n_features,)
        coalition_matrix: Binary coalition matrix, shape (n_coalitions, n_features)
    
    Returns:
        Imputed data, shape (n_coalitions, n_features)
    
    Algorithm:
        For each coalition i:
            imputed[i, j] = x[j] if S[i, j] == 1 else reference[j]
    
    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> ref = np.array([0.0, 0.0, 0.0])
        >>> S = np.array([[1, 1, 0], [1, 0, 1]])  # 2 coalitions
        >>> result = baseline_impute_numpy(x, ref, S)
        >>> result
        array([[1., 2., 0.],
               [1., 0., 3.]])
    """
    # Validate shapes
    n_features = x.shape[0]
    if reference.shape[0] != n_features:
        msg = f"Reference shape {reference.shape} doesn't match x shape {x.shape}"
        raise ValueError(msg)
    if coalition_matrix.shape[1] != n_features:
        msg = f"Coalition matrix has {coalition_matrix.shape[1]} features, expected {n_features}"
        raise ValueError(msg)
    
    n_coalitions = coalition_matrix.shape[0]
    
    # Broadcast x and reference to (n_coalitions, n_features)
    x_broadcast = np.tile(x, (n_coalitions, 1))
    reference_broadcast = np.tile(reference, (n_coalitions, 1))
    
    # Apply coalition matrix: S=1 keeps x, S=0 uses reference
    imputed = coalition_matrix * x_broadcast + (1 - coalition_matrix) * reference_broadcast
    
    return imputed


@marginal_impute.register(np.ndarray)
def marginal_impute_numpy(
    x: np.ndarray,
    data: np.ndarray,
    coalition_matrix: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Marginal imputation for NumPy arrays.
    
    Args:
        x: Data point to explain, shape (n_features,)
        data: Background dataset, shape (n_background, n_features)
        coalition_matrix: Binary coalition matrix, shape (n_coalitions, n_features)
        n_samples: Number of samples per coalition
    
    Returns:
        Imputed data, shape (n_coalitions, n_samples, n_features)
    
    Algorithm:
        For each coalition i and sample k:
            - Keep features where S[i, j] == 1 from x
            - Sample features where S[i, j] == 0 from data distribution
    
    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> data = np.random.randn(100, 3)  # background data
        >>> S = np.array([[1, 1, 0], [1, 0, 1]])
        >>> result = marginal_impute_numpy(x, data, S, n_samples=50)
        >>> result.shape
        (2, 50, 3)
    """
    # Validate shapes
    n_features = x.shape[0]
    n_background, data_features = data.shape
    
    if data_features != n_features:
        msg = f"Data has {data_features} features, expected {n_features}"
        raise ValueError(msg)
    if coalition_matrix.shape[1] != n_features:
        msg = f"Coalition matrix has {coalition_matrix.shape[1]} features, expected {n_features}"
        raise ValueError(msg)
    
    n_coalitions = coalition_matrix.shape[0]
    
    # Initialize output: (n_coalitions, n_samples, n_features)
    imputed = np.zeros((n_coalitions, n_samples, n_features))
    
    # Process each coalition
    for i in range(n_coalitions):
        coalition = coalition_matrix[i]
        
        # Sample background indices for this coalition
        background_indices = np.random.randint(0, n_background, size=n_samples)
        sampled_data = data[background_indices]  # (n_samples, n_features)
        
        # Broadcast x to (n_samples, n_features)
        x_broadcast = np.tile(x, (n_samples, 1))
        
        # Broadcast coalition to (n_samples, n_features)
        coalition_broadcast = np.tile(coalition, (n_samples, 1))
        
        # Apply coalition: S=1 keeps x, S=0 uses sampled data
        imputed[i] = coalition_broadcast * x_broadcast + (1 - coalition_broadcast) * sampled_data
    
    return imputed


@compute_mean.register(np.ndarray)
def compute_mean_numpy(data: np.ndarray, axis: int) -> np.ndarray:
    """Compute mean along specified axis for NumPy arrays.
    
    Args:
        data: Input array
        axis: Axis along which to compute mean
    
    Returns:
        Mean values along specified axis
    
    Example:
        >>> data = np.random.randn(10, 5, 3)
        >>> result = compute_mean_numpy(data, axis=1)
        >>> result.shape
        (10, 3)
    """
    return np.mean(data, axis=axis)
