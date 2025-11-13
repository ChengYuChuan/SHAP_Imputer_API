"""
2D Image Imputation Demonstration Case

Demonstrates three different imputation scenarios:
1. All patches kept
2. Partial patches kept
3. All patches imputed

Image: tests/pics/cat.jpg
Format: RGB (3 channels) - actually 3D data (C, H, W)
Reference: zero-value (black image)
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from imputer import BaselineImputer, CoalitionMatrix, PatchGrouping


def load_and_prepare_image(image_path: str, target_size: tuple[int, int] = (224, 224)):
    """Load and prepare image data
    
    Args:
        image_path: path to the image
        target_size: target size (height, width)
    
    Returns:
        image_array: NumPy array, shape (C, H, W)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize
    img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    
    # Convert to NumPy array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Convert to (C, H, W) format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    return img_array


def create_coalition_scenarios(n_patches: int, seed: int = 55):
    """Create three coalition matrix scenarios
    
    Args:
        n_patches: total number of patches
        seed: random seed to ensure reproducibility
    
    Returns:
        dict: contains coalition matrices for three scenarios
    """
    # Set random seed to ensure reproducibility
    np.random.seed(seed)
    
    # Scenario 2: randomly select about 50% of patches
    random_selection = np.random.choice([True, False], size=n_patches, p=[0.5, 0.5])
    
    scenarios = {
        "scenario_1_all_kept": np.ones((1, n_patches), dtype=bool),
        "scenario_2_partial": random_selection.reshape(1, -1).astype(bool),
        "scenario_3_all_imputed": np.zeros((1, n_patches), dtype=bool)
    }
    
    return scenarios


def visualize_results(original_image, imputed_images, scenarios_info, save_path=None):
    """Visualize results
    
    Args:
        original_image: original image (C, H, W)
        imputed_images: dictionary containing imputed images for each scenario
        scenarios_info: descriptive information about the scenarios
        save_path: optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Image Imputation: 3 Scenarios', fontsize=16, fontweight='bold')
    
    # Convert to (H, W, C) for display
    original_display = np.transpose(original_image, (1, 2, 0))
    
    # Original image
    axes[0, 0].imshow(original_display)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Three scenarios
    titles = [
        'Scenario 1: All Patches Kept\n(Coalition = all True)',
        'Scenario 2: Random ~50% Patches Kept\n(Coalition = random True/False with p=0.5)',
        'Scenario 3: All Patches Imputed\n(Coalition = all False, Reference = 0)'
    ]
    
    positions = [(0, 1), (1, 0), (1, 1)]
    
    for idx, (scenario_name, title, pos) in enumerate(zip(
        ['scenario_1_all_kept', 'scenario_2_partial', 'scenario_3_all_imputed'],
        titles,
        positions
    )):
        imputed_display = np.transpose(imputed_images[scenario_name], (1, 2, 0))
        axes[pos].imshow(imputed_display.clip(0, 1))
        axes[pos].set_title(title, fontsize=10)
        axes[pos].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Image saved to: {save_path}")
    
    plt.show()


def main():
    print("=" * 80)
    print("2D Image Imputation Demonstration (actually 3D data: C×H×W)")
    print("=" * 80)
    
    # 1. Load image
    image_path = "tests/pics/catSmall.jpg"
    print(f"\n1. Loading image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found {image_path}")
        print("Please make sure the image exists at the specified path")
        return
    
    image = load_and_prepare_image(image_path, target_size=(224, 224))
    print(f"   ✓ Image shape: {image.shape}")
    print(f"   ✓ Data dimensions: 3D (channels={image.shape[0]}, height={image.shape[1]}, width={image.shape[2]})")
    print(f"   ✓ Value range: [{image.min():.3f}, {image.max():.3f}]")
    
    # 2. Create reference (all zeros = black image)
    reference = np.zeros_like(image)
    print(f"\n2. Reference (all zeros):")
    print(f"   ✓ Reference shape: {reference.shape}")
    print(f"   ✓ Visual appearance: black image")
    
    # 3. Use PatchGrouping
    patch_size = (16, 16)
    print(f"\n3. Create PatchGrouping:")
    print(f"   ✓ Patch size: {patch_size}")
    
    grouping = PatchGrouping(patch_size=patch_size)
    x_grouped = grouping.fit_transform(image)
    ref_grouped = grouping.transform(reference)
    
    print(f"   ✓ Grouped shape: {x_grouped.shape}")
    print(f"   ✓ Number of patches: {grouping.n_groups}")
    print(f"   ✓ Features per patch: {grouping.features_per_patch}")
    print(f"   ✓ Grid shape: {grouping.grid_shape}")
    
    # 4. Create three coalition scenarios
    print(f"\n4. Create three Coalition Matrix scenarios:")
    scenarios = create_coalition_scenarios(grouping.n_groups)
    
    for name, S_matrix in scenarios.items():
        n_kept = S_matrix.sum()
        percentage = (n_kept / grouping.n_groups) * 100
        print(f"   ✓ {name}:")
        print(f"      - Shape: {S_matrix.shape}")
        print(f"      - Dtype: {S_matrix.dtype} {'✅' if S_matrix.dtype == bool else '❌'}")
        print(f"      - Patches kept: {n_kept}/{grouping.n_groups} ({percentage:.1f}%)")
    
    # 5. Perform Imputation
    print(f"\n5. Performing Imputation:")
    
    # Flatten data
    x_flat = x_grouped.flatten()
    ref_flat = ref_grouped.flatten()
    
    imputed_images = {}
    
    for scenario_name, S_matrix in scenarios.items():
        print(f"\n   Processing {scenario_name}...")
        
        # Create CoalitionMatrix
        S = CoalitionMatrix(S_matrix)
        
        # Expand coalition matrix
        S_expanded = grouping.expand_coalition_matrix(S)
        
        # Create imputer
        imputer = BaselineImputer(reference=ref_flat, x=x_flat)
        
        # Impute
        imputed_flat = imputer.impute(S_expanded)
        
        # Reshape
        n_coalitions = S.n_coalitions
        imputed_grouped = imputed_flat.reshape(
            n_coalitions, 
            grouping.n_groups, 
            grouping.features_per_patch
        )
        
        # Inverse transform
        imputed_image = grouping.inverse_transform(imputed_grouped)
        
        # Save result (remove batch dimension)
        imputed_images[scenario_name] = imputed_image[0]
        
        print(f"      ✓ Imputed shape: {imputed_image.shape}")
    
    # 6. Validate results
    print(f"\n6. Validate results:")
    
    # Scenario 1: should be identical to the original image
    diff_1 = np.abs(imputed_images['scenario_1_all_kept'] - image).mean()
    print(f"   ✓ Scenario 1 (all kept) difference from original: {diff_1:.6e}")
    assert diff_1 < 1e-5, "Scenario 1 should be identical to the original image"
    
    # Scenario 3: should be all black
    diff_3 = np.abs(imputed_images['scenario_3_all_imputed']).mean()
    print(f"   ✓ Scenario 3 (all imputed) mean value: {diff_3:.6e}")
    assert diff_3 < 1e-5, "Scenario 3 should be all zeros (black)"
    
    # Scenario 2: about 50% kept, 50% imputed
    partial = imputed_images['scenario_2_partial']
    partial_coalition = scenarios['scenario_2_partial'][0]
    
    # Calculate actual kept ratio
    kept_ratio = partial_coalition.sum() / len(partial_coalition)
    
    # Compute mean difference (kept areas should match original)
    diff_2 = np.abs(partial - image).mean()
    max_possible_diff = image.mean()  # if all imputed, max difference ≈ mean of original
    
    print(f"   ✓ Scenario 2 actual kept ratio: {kept_ratio:.1%}")
    print(f"   ✓ Scenario 2 mean difference from original: {diff_2:.6f}")
    print(f"   ✓ Scenario 2 theoretical max difference: {max_possible_diff:.6f}")
    print(f"   ✓ Scenario 2 difference ratio: {(diff_2/max_possible_diff):.1%}")
    
    # Check that difference is within reasonable range
    assert 0 <= diff_2 <= max_possible_diff * 1.5, "Scenario 2 difference should be reasonable"
    
    # 7. Visualization
    print(f"\n7. Visualizing results...")
    visualize_results(
        image, 
        imputed_images,
        scenarios,
        save_path='example/image_imputation_demo.png'
    )
    
    print("\n" + "=" * 80)
    print("✅ Demonstration complete!")
    print("=" * 80)
    print("\nSummary:")
    print("  • Data format: 3D (C, H, W) - 3-channel RGB image")
    print("  • Coalition Matrix: Boolean dtype ✅")
    print("  • Three scenarios demonstrated ✅")
    print("  • Validation passed ✅")


if __name__ == "__main__":
    main()
