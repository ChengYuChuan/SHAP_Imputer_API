"""Feature grouping strategies for high-dimensional data.

This module provides strategies to group high-dimensional features
(e.g., pixels in images) into meaningful units for imputation.

Available grouping strategies:
    - PatchGrouping: Divide data into fixed-size patches (2D/3D/4D+)

Future strategies:
    - SuperpixelGrouping: SLIC-based superpixel segmentation
    - AdaptivePatchGrouping: Automatic patch size calculation
    
Example:
    >>> from imputer.grouping import PatchGrouping
    >>> import numpy as np
    >>> 
    >>> # 2D image
    >>> image = np.random.randn(3, 224, 224)
    >>> grouping = PatchGrouping(patch_size=(16, 16))
    >>> grouped = grouping.fit_transform(image)
    >>> print(grouped.shape)  # (196, 768)
    >>> 
    >>> # 3D video
    >>> video = np.random.randn(3, 32, 224, 224)
    >>> grouping = PatchGrouping(patch_size=(8, 16, 16))
    >>> grouped = grouping.fit_transform(video)
    >>> print(grouped.shape)  # (784, 6144)
"""

from imputer.grouping.base import FeatureGrouping
from imputer.grouping.patch import PatchGrouping

__all__ = [
    "FeatureGrouping",
    "PatchGrouping",
]