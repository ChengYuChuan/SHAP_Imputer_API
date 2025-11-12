# test_grouping_basic.py
import numpy as np
from imputer import PatchGrouping

# 2D 圖像
image = np.random.randn(3, 64, 64)
grouping = PatchGrouping(patch_size=(16, 16))

# Transform
grouped = grouping.fit_transform(image)
print(f"Grouped shape: {grouped.shape}")  # (16, 768)

# Inverse
reconstructed = grouping.inverse_transform(grouped)
print(f"Reconstructed shape: {reconstructed.shape}")  # (3, 64, 64)

# Verify
assert np.allclose(reconstructed, image)
print("✓ Basic test passed")