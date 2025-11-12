"""PyTorch backend implementations for imputation.

This module provides PyTorch-specific implementations of imputation strategies.
Optimized for GPU acceleration and automatic differentiation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def baseline_impute_torch(
    x: torch.Tensor,
    reference: torch.Tensor,
    coalition_matrix: torch.Tensor,
) -> torch.Tensor:
    """Baseline imputation for PyTorch tensors.
    
    Args:
        x: Data point to explain, shape (n_features,)
        reference: Reference values, shape (n_features,)
        coalition_matrix: Binary coalition matrix, shape (n_coalitions, n_features)
    
    Returns:
        Imputed data, shape (n_coalitions, n_features)
    
    Note:
        - Preserves device (CPU/GPU)
        - Preserves gradient tracking
        - Supports batched operations for efficiency
    
    Example:
        >>> import torch
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> ref = torch.zeros(3)
        >>> S = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
        >>> result = baseline_impute_torch(x, ref, S)
        >>> result
        tensor([[1., 2., 0.],
                [1., 0., 3.]])
    """
    import torch
    
    # Validate shapes
    n_features = x.shape[0]
    if reference.shape[0] != n_features:
        msg = f"Reference shape {reference.shape} doesn't match x shape {x.shape}"
        raise ValueError(msg)
    if coalition_matrix.shape[1] != n_features:
        msg = f"Coalition matrix has {coalition_matrix.shape[1]} features, expected {n_features}"
        raise ValueError(msg)
    
    # Ensure all tensors are on the same device
    device = x.device
    if reference.device != device:
        reference = reference.to(device)
    if coalition_matrix.device != device:
        coalition_matrix = coalition_matrix.to(device)
    
    n_coalitions = coalition_matrix.shape[0]
    
    # Expand x and reference to (n_coalitions, n_features)
    x_expanded = x.unsqueeze(0).expand(n_coalitions, -1)
    reference_expanded = reference.unsqueeze(0).expand(n_coalitions, -1)
    
    # Apply coalition matrix: S=1 keeps x, S=0 uses reference
    # This preserves gradients through both x and reference
    imputed = coalition_matrix * x_expanded + (1 - coalition_matrix) * reference_expanded
    
    return imputed


def marginal_impute_torch(
    x: torch.Tensor,
    data: torch.Tensor,
    coalition_matrix: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Marginal imputation for PyTorch tensors.
    
    Args:
        x: Data point to explain, shape (n_features,)
        data: Background dataset, shape (n_background, n_features)
        coalition_matrix: Binary coalition matrix, shape (n_coalitions, n_features)
        n_samples: Number of samples per coalition
    
    Returns:
        Imputed data, shape (n_coalitions, n_samples, n_features)
    
    Note:
        - Efficient GPU implementation using random sampling
        - Preserves gradients for differentiable explanations
        - Uses torch.randint for efficient sampling
    
    Example:
        >>> import torch
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> data = torch.randn(100, 3)
        >>> S = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
        >>> result = marginal_impute_torch(x, data, S, n_samples=50)
        >>> result.shape
        torch.Size([2, 50, 3])
    """
    import torch
    
    # Validate shapes
    n_features = x.shape[0]
    n_background, data_features = data.shape
    
    if data_features != n_features:
        msg = f"Data has {data_features} features, expected {n_features}"
        raise ValueError(msg)
    if coalition_matrix.shape[1] != n_features:
        msg = f"Coalition matrix has {coalition_matrix.shape[1]} features, expected {n_features}"
        raise ValueError(msg)
    
    # Ensure all tensors are on the same device
    device = x.device
    if data.device != device:
        data = data.to(device)
    if coalition_matrix.device != device:
        coalition_matrix = coalition_matrix.to(device)
    
    n_coalitions = coalition_matrix.shape[0]
    
    # Sample background indices: (n_coalitions, n_samples)
    background_indices = torch.randint(
        0, n_background, 
        (n_coalitions, n_samples),
        device=device
    )
    
    # Sample from background data: (n_coalitions, n_samples, n_features)
    # For each coalition, sample n_samples rows from data
    sampled_data = data[background_indices]
    
    # Expand x to (n_coalitions, n_samples, n_features)
    x_expanded = x.unsqueeze(0).unsqueeze(0).expand(n_coalitions, n_samples, -1)
    
    # Expand coalition_matrix to (n_coalitions, n_samples, n_features)
    coalition_expanded = coalition_matrix.unsqueeze(1).expand(-1, n_samples, -1)
    
    # Apply coalition: S=1 keeps x, S=0 uses sampled data
    imputed = coalition_expanded * x_expanded + (1 - coalition_expanded) * sampled_data
    
    return imputed


def compute_mean_torch(data: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute mean along specified axis for PyTorch tensors.
    
    Args:
        data: Input tensor
        axis: Axis along which to compute mean
    
    Returns:
        Mean values along specified axis
    
    Note:
        Preserves gradients for backpropagation
    
    Example:
        >>> import torch
        >>> data = torch.randn(10, 5, 3)
        >>> result = compute_mean_torch(data, axis=1)
        >>> result.shape
        torch.Size([10, 3])
    """
    import torch
    
    return torch.mean(data, dim=axis)
