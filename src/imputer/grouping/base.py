"""Abstract base class for feature grouping strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class FeatureGrouping(ABC):
    """Abstract base class for feature grouping strategies.
    
    Transforms high-dimensional data into grouped representations
    where each group can be treated as a single feature for imputation.
    
    The grouping process has two key operations:
    1. transform: high-dim → grouped representation
    2. inverse_transform: grouped representation → high-dim
    
    These operations must be invertible (i.e., perfect reconstruction).
    """
    
    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform high-dimensional data to grouped representation.
        
        Args:
            x: Input data with arbitrary shape
        
        Returns:
            Grouped data of shape (n_groups, group_size)
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(self, grouped: np.ndarray) -> np.ndarray:
        """Reconstruct original shape from grouped representation.
        
        Args:
            grouped: Grouped data of shape (n_groups, group_size) or
                     (n_coalitions, n_groups, group_size)
        
        Returns:
            Data in original shape
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def n_groups(self) -> int:
        """Number of feature groups.
        
        Returns:
            Number of groups
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def original_shape(self) -> tuple[int, ...]:
        """Original data shape.
        
        Returns:
            Shape tuple
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError