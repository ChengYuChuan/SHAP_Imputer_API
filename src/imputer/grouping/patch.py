"""Generic n-dimensional patch grouping for arbitrary spatial dimensions.

This module provides a unified PatchGrouping that works for:
- 2D images: (C, H, W)
- 3D videos/volumes: (C, T, H, W) or (C, D, H, W)  
- 4D+: (C, D1, D2, D3, ...)

Key insight: All operations can be expressed generically using:
- Dynamic reshape based on n_spatial_dims
- Generic transpose operations
- Automatic dimension detection from patch_size
"""

from __future__ import annotations

import numpy as np

from imputer.grouping.base import FeatureGrouping


class PatchGrouping(FeatureGrouping):
    """Generic n-dimensional patch grouping.
    
    Works with any number of spatial dimensions by treating the first
    dimension as channels and remaining dimensions as spatial.
    
    Examples:
        >>> # 2D image: (C, H, W)
        >>> grouping = PatchGrouping(patch_size=(16, 16))
        >>> x = np.random.randn(3, 224, 224)
        >>> grouped = grouping.fit_transform(x)
        >>> grouped.shape
        (196, 768)  # 196 patches of 3*16*16 features
        
        >>> # 3D video: (C, T, H, W)
        >>> grouping = PatchGrouping(patch_size=(8, 16, 16))
        >>> x = np.random.randn(3, 32, 224, 224)
        >>> grouped = grouping.fit_transform(x)
        >>> grouped.shape
        (784, 6144)  # 4*14*14 patches of 3*8*16*16 features
        
        >>> # 3D volume: (C, D, H, W)
        >>> grouping = PatchGrouping(patch_size=(4, 16, 16))
        >>> x = np.random.randn(1, 64, 128, 128)
        >>> grouped = grouping.fit_transform(x)
        >>> grouped.shape
        (1024, 1024)  # 16*8*8 patches of 1*4*16*16 features
    
    Attributes:
        patch_size: Tuple of patch sizes for each spatial dimension
        n_spatial_dims: Number of spatial dimensions (inferred from patch_size)
        original_shape: Original data shape
        n_groups: Total number of patches
    """
    
    def __init__(self, patch_size: tuple[int, ...]):
        """Initialize patch grouping.
        
        Args:
            patch_size: Patch sizes for each spatial dimension.
                - For 2D: (patch_h, patch_w)
                - For 3D: (patch_t, patch_h, patch_w)
                - For nD: (patch_d1, patch_d2, ..., patch_dn)
        """
        self.patch_size = patch_size
        self.n_spatial_dims = len(patch_size)
        self._original_shape: tuple[int, ...] | None = None
        self._n_groups: int | None = None
        self._grid_shape: tuple[int, ...] | None = None
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit to data shape and transform to patches.
        
        Args:
            x: Input array of shape (C, D1, D2, ..., Dn)
               where C is channels and D1...Dn are spatial dimensions
        
        Returns:
            Grouped patches of shape (n_patches, C * p1 * p2 * ... * pn)
        
        Raises:
            ValueError: If spatial dimensions don't match patch_size length
        """
        self._original_shape = x.shape
        return self.transform(x)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform array to patches.
        
        Args:
            x: Array of shape (C, D1, D2, ..., Dn)
        
        Returns:
            Patches of shape (n_patches, features_per_patch)
            where features_per_patch = C * p1 * p2 * ... * pn
        
        Algorithm (generalized for n dimensions):
            1. Extract shapes: channels + spatial dimensions
            2. Validate divisibility
            3. Reshape to: (C, n1, p1, n2, p2, ..., nn, pn)
            4. Transpose to: (n1, n2, ..., nn, C, p1, p2, ..., pn)
            5. Flatten to: (n1*n2*...*nn, C*p1*p2*...*pn)
        """
        if self._original_shape is None:
            self._original_shape = x.shape
        
        # Split into channels and spatial dimensions
        n_channels = x.shape[0]
        spatial_shape = x.shape[1:]
        
        # Validate
        if len(spatial_shape) != self.n_spatial_dims:
            msg = (
                f"Expected {self.n_spatial_dims} spatial dimensions, "
                f"got {len(spatial_shape)} (shape: {x.shape})"
            )
            raise ValueError(msg)
        
        # Check divisibility
        for i, (dim_size, patch_size) in enumerate(zip(spatial_shape, self.patch_size)):
            if dim_size % patch_size != 0:
                msg = (
                    f"Spatial dimension {i} (size {dim_size}) not divisible "
                    f"by patch size {patch_size}"
                )
                raise ValueError(msg)
        
        # Calculate grid shape (number of patches along each dimension)
        grid_shape = tuple(d // p for d, p in zip(spatial_shape, self.patch_size))
        self._grid_shape = grid_shape
        self._n_groups = int(np.prod(grid_shape))
        
        # === Core transformation logic ===
        # Step 1: Reshape to interleave grid and patch dimensions
        # (C, D1, D2, ..., Dn) -> (C, n1, p1, n2, p2, ..., nn, pn)
        new_shape = [n_channels]
        for n, p in zip(grid_shape, self.patch_size):
            new_shape.extend([n, p])
        
        x_reshaped = x.reshape(new_shape)
        
        # Step 2: Transpose to group patches
        # Move channel and patch dimensions to the end
        # (C, n1, p1, n2, p2, ...) -> (n1, n2, ..., nn, C, p1, p2, ..., pn)
        
        # Build transpose axes:
        # - Grid dimensions: [1, 3, 5, ...] (odd indices after channel)
        # - Channel: [0]
        # - Patch dimensions: [2, 4, 6, ...] (even indices after channel)
        grid_axes = list(range(1, len(new_shape), 2))  # [1, 3, 5, ...]
        patch_axes = list(range(2, len(new_shape), 2))  # [2, 4, 6, ...]
        transpose_axes = grid_axes + [0] + patch_axes
        
        x_transposed = x_reshaped.transpose(transpose_axes)
        
        # Step 3: Flatten to (n_patches, features_per_patch)
        n_patches = self._n_groups
        features_per_patch = n_channels * int(np.prod(self.patch_size))
        
        patches = x_transposed.reshape(n_patches, features_per_patch)
        
        return patches
    
    def inverse_transform(self, grouped: np.ndarray) -> np.ndarray:
        """Reconstruct from patches.
        
        Args:
            grouped: Patches of shape (n_patches, features) or
                     (n_coalitions, n_patches, features)
        
        Returns:
            Reconstructed array of original shape or
            (n_coalitions, *original_shape)
        
        Algorithm (reverse of transform):
            1. Reshape to: (n1, n2, ..., nn, C, p1, p2, ..., pn)
            2. Transpose to: (C, n1, p1, n2, p2, ..., nn, pn)
            3. Reshape to: (C, D1, D2, ..., Dn)
        """
        if self._original_shape is None:
            msg = "Must call fit_transform or transform before inverse_transform"
            raise ValueError(msg)
        
        # Handle batched input
        if grouped.ndim == 3:
            n_coalitions = grouped.shape[0]
            reconstructed = []
            for i in range(n_coalitions):
                rec = self._inverse_single(grouped[i])
                reconstructed.append(rec)
            return np.stack(reconstructed, axis=0)
        
        return self._inverse_single(grouped)
    
    def _inverse_single(self, patches: np.ndarray) -> np.ndarray:
        """Reconstruct single array from patches."""
        n_channels = self._original_shape[0]
        grid_shape = self._grid_shape
        
        # Step 1: Reshape to (n1, n2, ..., nn, C, p1, p2, ..., pn)
        intermediate_shape = list(grid_shape) + [n_channels] + list(self.patch_size)
        patches_reshaped = patches.reshape(intermediate_shape)
        
        # Step 2: Transpose to (C, n1, p1, n2, p2, ..., nn, pn)
        # Inverse of the forward transpose
        n_spatial = len(grid_shape)
        
        # Build inverse transpose axes:
        # Current: (n1, n2, ..., nn, C, p1, p2, ..., pn)
        # Target:  (C, n1, p1, n2, p2, ..., nn, pn)
        
        # Channel goes first (from position n_spatial)
        # Then interleave grid and patch dimensions
        new_axes = [n_spatial]  # Channel dimension
        for i in range(n_spatial):
            new_axes.append(i)  # Grid dimension i
            new_axes.append(n_spatial + 1 + i)  # Patch dimension i
        
        patches_transposed = patches_reshaped.transpose(new_axes)
        
        # Step 3: Reshape to original shape
        reconstructed = patches_transposed.reshape(self._original_shape)
        
        return reconstructed
    
    def expand_coalition_matrix(self, S: CoalitionMatrix) -> CoalitionMatrix:
        """Expand patch-level coalition matrix to feature-level."""
        from imputer.core.base import CoalitionMatrix as CM
        
        if self._original_shape is None:
            msg = "Must call fit_transform first"
            raise ValueError(msg)
        
        features_per_patch = self.features_per_patch
        
        if S.n_features != self.n_groups:
            msg = (
                f"Coalition matrix has {S.n_features} patches, "
                f"but grouping has {self.n_groups} patches"
            )
            raise ValueError(msg)
        
        S_expanded = np.repeat(S.matrix, features_per_patch, axis=1)
        return CM(S_expanded)

    @property
    def n_groups(self) -> int:
        """Number of patches."""
        if self._n_groups is None:
            msg = "Must call fit_transform or transform first"
            raise ValueError(msg)
        return self._n_groups
    
    @property
    def original_shape(self) -> tuple[int, ...]:
        """Original array shape."""
        if self._original_shape is None:
            msg = "Must call fit_transform or transform first"
            raise ValueError(msg)
        return self._original_shape
    
    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the patch grid (number of patches per dimension)."""
        if self._grid_shape is None:
            msg = "Must call fit_transform or transform first"
            raise ValueError(msg)
        return self._grid_shape
    
    @property
    def features_per_patch(self) -> int:
        """Number of features in each patch.
        
        Returns:
            Number of features = channels × patch_volume
            
        Example:
            >>> # RGB image (3 channels), 16×16 patches
            >>> # features_per_patch = 3 × 16 × 16 = 768
            >>> grouping = PatchGrouping(patch_size=(16, 16))
            >>> x = np.random.randn(3, 224, 224)
            >>> grouping.fit_transform(x)
            >>> grouping.features_per_patch
            768
        
        Raises:
            ValueError: If fit_transform has not been called yet
        """
        if self._original_shape is None:
            msg = "Must call fit_transform or transform first"
            raise ValueError(msg)
        
        n_channels = self._original_shape[0]
        patch_volume = int(np.prod(self.patch_size))
        return n_channels * patch_volume