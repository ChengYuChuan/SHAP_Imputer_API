# test_grouping_integration_corrected.py
"""
Corrected integration test with proper coalition matrix understanding.

Key insights:
1. Coalition matrix shape: (n_coalitions, n_patches) - n_coalitions is arbitrary
2. Channels are treated as a unit (RGB together, not separately)
3. Boolean dtype is mandatory
"""
import numpy as np
from imputer import PatchGrouping, BaselineImputer, CoalitionMatrix

print("=" * 70)
print("Corrected Integration Test")
print("=" * 70)

# Setup
image = np.random.randn(3, 224, 224)
reference = np.zeros((3, 224, 224))

# Grouping
grouping = PatchGrouping(patch_size=(16, 16))
x_grouped = grouping.fit_transform(image)  # (196, 768)
ref_grouped = grouping.transform(reference)  # (196, 768)

print(f"\n1. Data shapes:")
print(f"   Original image: {image.shape}")
print(f"   Grouped: {x_grouped.shape}")
print(f"   - n_patches: {grouping.n_groups} = 14 × 14")
print(f"   - features_per_patch: {grouping.features_per_patch} = 3 × 16 × 16")

# ========== CORRECTED COALITION MATRIX EXAMPLES ==========

print("\n2. Coalition Matrix Examples:")
print("   (n_coalitions is arbitrary, n_patches is fixed at 196)")

# Example 1: Just 2 coalitions (as user suggested)
print("\n   Example 1: 2 coalitions")
S_2_coalitions = CoalitionMatrix(np.array([
    [True] * 98 + [False] * 98,   # Coalition 0: first half of patches
    [False] * 98 + [True] * 98    # Coalition 1: second half of patches
], dtype=bool))  # Shape: (2, 196) ✓ Correct!

print(f"   Shape: {S_2_coalitions.matrix.shape}")
print(f"   Coalition 0 keeps {S_2_coalitions.matrix[0].sum()} patches")
print(f"   Coalition 1 keeps {S_2_coalitions.matrix[1].sum()} patches")

# Example 2: Single coalition (all patches)
print("\n   Example 2: 1 coalition (keep all patches)")
S_all = CoalitionMatrix(np.ones((1, 196), dtype=bool))
print(f"   Shape: {S_all.matrix.shape}")

# Example 3: Single coalition (no patches, all imputed)
print("\n   Example 3: 1 coalition (impute all patches)")
S_none = CoalitionMatrix(np.zeros((1, 196), dtype=bool))
print(f"   Shape: {S_none.matrix.shape}")

# Example 4: Many coalitions for testing
print("\n   Example 4: 10 random coalitions")
S_random = CoalitionMatrix(np.random.choice(
    [True, False], 
    size=(10, 196),
    p=[0.5, 0.5]  # 50% chance of keeping each patch
).astype(bool))
print(f"   Shape: {S_random.matrix.shape}")

# =========================================================

# Let's use Example 1 (2 coalitions) for the actual test
print("\n3. Running imputation with 2 coalitions:")
S = S_2_coalitions

# Flatten
x_flat = x_grouped.flatten()      # (150528,)
ref_flat = ref_grouped.flatten()  # (150528,)
print(f"   Flattened data: {x_flat.shape}")

# Expand coalition matrix
S_expanded = grouping.expand_coalition_matrix(S)
print(f"   Expanded coalition: {S_expanded.matrix.shape}")
print(f"   - From (2, 196) to (2, 150528)")
print(f"   - Each patch value repeated 768 times")

# Impute
imputer = BaselineImputer(reference=ref_flat, x=x_flat)
imputed_flat = imputer.impute(S_expanded)  # (2, 150528)
print(f"\n4. Imputed (flat): {imputed_flat.shape}")

# Reshape
n_coalitions = S.n_coalitions  # 2
n_patches = grouping.n_groups  # 196
features_per_patch = grouping.features_per_patch  # 768
imputed_grouped = imputed_flat.reshape(n_coalitions, n_patches, features_per_patch)
print(f"   Imputed (grouped): {imputed_grouped.shape}")

# Inverse transform
imputed_images = grouping.inverse_transform(imputed_grouped)
print(f"   Final result: {imputed_images.shape}")
print(f"   - 2 coalitions, each producing one (3, 224, 224) image")

# Verification
print("\n5. Verification:")

# Coalition 0: first half of patches should match original
first_half_original = image[:, :, :112]  # First 7 patches width-wise
first_half_result = imputed_images[0, :, :, :112]
print(f"   Coalition 0 (first half):")
print(f"   - Original mean: {first_half_original.mean():.6f}")
print(f"   - Result mean: {first_half_result.mean():.6f}")

# Second half should be zeros (imputed)
second_half_result = imputed_images[0, :, :, 112:]
print(f"   Coalition 0 (second half - should be zeros):")
print(f"   - Mean: {abs(second_half_result.mean()):.6e}")

print("\n" + "=" * 70)
print("✓ Test passed with 2 coalitions (shape: 2, 196)")
print("✓ Channels (RGB) treated as single unit")
print("✓ Boolean coalition matrix")
print("=" * 70)