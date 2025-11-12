# test_grouping_integration.py
import numpy as np
from imputer import PatchGrouping, BaselineImputer, CoalitionMatrix

# Setup
image = np.random.randn(3, 224, 224)
reference = np.zeros((3, 224, 224))

# Grouping
grouping = PatchGrouping(patch_size=(16, 16))
x_grouped = grouping.fit_transform(image)
ref_grouped = grouping.transform(reference)

# Coalition matrix
S = CoalitionMatrix(np.eye(196, dtype=bool))  # 196 patches

# Imputation
imputer = BaselineImputer(reference=ref_grouped, x=x_grouped)
imputed_grouped = imputer.impute(S)

# Reconstruction
imputed_images = grouping.inverse_transform(imputed_grouped)

print(f"Output shape: {imputed_images.shape}")  # (196, 3, 224, 224)
print("✓ Integration test passed")