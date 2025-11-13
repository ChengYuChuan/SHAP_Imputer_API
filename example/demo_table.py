"""
2D Tabular Data Imputation Demonstration

Showcases three different imputation scenarios:
1. All features kept
2. Partial features kept (random)
3. All features imputed

Data format: 20 customers × 4 features
Features: age, income, credit_score, years_employed
Reference: zero values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from imputer import BaselineImputer, CoalitionMatrix


def generate_customer_data(n_customers: int = 20, seed: int = 42):
    """Generate simulated customer data table
    
    Args:
        n_customers: number of customers
        seed: random seed
    
    Returns:
        data: NumPy array (n_customers, 4)
        df: Pandas DataFrame
    """
    np.random.seed(seed)
    
    # Generate 4 features
    age = np.random.randint(18, 80, n_customers)
    income = np.random.lognormal(10, 0.5, n_customers)
    credit_score = np.random.normal(700, 50, n_customers).clip(300, 850)
    years_employed = np.random.exponential(5, n_customers).clip(0, 40)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'years_employed': years_employed
    })
    
    # Convert to NumPy array (n_customers, 4)
    data = df.values
    
    # Normalize to [0, 1] for visualization
    data_normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    
    return data_normalized, df


def save_customer_table(df, save_path):
    """Save customer data table
    
    Args:
        df: Pandas DataFrame
        save_path: file path to save
    """
    # Save as CSV
    csv_path = save_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"   ✓ Full dataset saved to: {csv_path}")
    print(f"   ✓ Data size: {df.shape[0]} customers × {df.shape[1]} features")
    
    # Create table preview image
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Show first 10 customers, format values
    preview_df = df.head(10).copy()
    preview_df['age'] = preview_df['age'].astype(int)
    preview_df['income'] = preview_df['income'].round(0).astype(int)
    preview_df['credit_score'] = preview_df['credit_score'].round(0).astype(int)
    preview_df['years_employed'] = preview_df['years_employed'].round(1)
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=preview_df.values,
        colLabels=preview_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.25, 0.25, 0.25]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Header style
    for i in range(len(preview_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Title
    ax.set_title(f'Customer Data Preview (First 10 of {len(df)} customers)', 
                 fontweight='bold', pad=20, fontsize=12)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Table preview image saved to: {save_path}")
    plt.close()


def create_coalition_scenarios(n_features: int, seed: int = 42):
    """Create three coalition matrix scenarios
    
    Args:
        n_features: total number of features
        seed: random seed for reproducibility
    
    Returns:
        dict: contains three coalition matrices
    """
    np.random.seed(seed)
    
    # Scenario 2: randomly select ~50% of features
    random_selection = np.random.choice([True, False], size=n_features, p=[0.5, 0.5])
    
    scenarios = {
        "scenario_1_all_kept": np.ones((1, n_features), dtype=bool),
        "scenario_2_partial": random_selection.reshape(1, -1).astype(bool),
        "scenario_3_all_imputed": np.zeros((1, n_features), dtype=bool)
    }
    
    return scenarios


def visualize_results(original_data, imputed_data_dict, feature_names, scenarios, save_path=None):
    """Visualize imputation results"""
    n_customers, n_features = original_data.shape
    
    fig = plt.figure(figsize=(16, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Tabular Data Imputation: 3 Scenarios', fontsize=16, fontweight='bold')
    
    scenario_names = ['scenario_1_all_kept', 'scenario_2_partial', 'scenario_3_all_imputed']
    titles = [
        'Scenario 1: All Features Kept\n(Coalition = all True)',
        'Scenario 2: Random ~50% Features Kept\n(Coalition = random True/False with p=0.5)',
        'Scenario 3: All Features Imputed\n(Coalition = all False, Reference = 0)'
    ]
    
    for idx, (scenario_name, title) in enumerate(zip(scenario_names, titles)):
        # Left: heatmap
        ax_heatmap = fig.add_subplot(gs[idx, 0])
        imputed_data = imputed_data_dict[scenario_name]
        
        im = ax_heatmap.imshow(imputed_data.T, aspect='auto', cmap='YlOrRd', 
                               vmin=0, vmax=1, interpolation='nearest')
        
        ax_heatmap.set_xlabel('Customer Index', fontsize=10)
        ax_heatmap.set_ylabel('Features', fontsize=10)
        ax_heatmap.set_yticks(range(n_features))
        ax_heatmap.set_yticklabels(feature_names, fontsize=9)
        ax_heatmap.set_title(title, fontsize=11, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Value', rotation=270, labelpad=15, fontsize=9)
        
        # Right: mean value comparison
        ax_dist = fig.add_subplot(gs[idx, 1])
        x_positions = np.arange(n_features)
        width = 0.35
        
        original_means = original_data.mean(axis=0)
        imputed_means = imputed_data.mean(axis=0)
        
        ax_dist.bar(x_positions - width/2, original_means, width, label='Original', alpha=0.8, color='steelblue')
        ax_dist.bar(x_positions + width/2, imputed_means, width, label='Imputed', alpha=0.8, color='coral')
        
        ax_dist.set_xlabel('Features', fontsize=10)
        ax_dist.set_ylabel('Mean Value (Normalized)', fontsize=10)
        ax_dist.set_title(f'{scenario_name.replace("_", " ").title()}\nFeature Means', 
                         fontsize=10, fontweight='bold')
        ax_dist.set_xticks(x_positions)
        ax_dist.set_xticklabels(feature_names, fontsize=9, rotation=15, ha='right')
        ax_dist.legend(fontsize=9)
        ax_dist.grid(True, alpha=0.3, axis='y')
        ax_dist.set_ylim([0, 1])
        
        if scenario_name == 'scenario_2_partial':
            coalition = scenarios[scenario_name][0]
            kept_features = [feature_names[i] for i, keep in enumerate(coalition) if keep]
            imputed_features = [feature_names[i] for i, keep in enumerate(coalition) if not keep]
            
            info_text = f"Kept: {', '.join(kept_features) if kept_features else 'None'}\n"
            info_text += f"Imputed: {', '.join(imputed_features) if imputed_features else 'None'}"
            
            ax_dist.text(0.5, -0.25, info_text, transform=ax_dist.transAxes,
                        ha='center', va='top', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {save_path}")
    
    plt.show()


def main():
    print("=" * 80)
    print("2D Tabular Data Imputation Demonstration")
    print("Real Tabular Data: 20 customers × 4 features")
    print("=" * 80)

    # 1. Generate table data
    n_customers = 20
    n_features = 4
    feature_names = ['age', 'income', 'credit_score', 'years_employed']

    print(f"\n1. Generating simulated customer data:")
    print(f"   ✓ Number of customers: {n_customers}")
    print(f"   ✓ Number of features: {n_features}")
    print(f"   ✓ Feature names: {', '.join(feature_names)}")

    data_normalized, df_original = generate_customer_data(n_customers=n_customers)

    print(f"   ✓ Data shape: {data_normalized.shape}")
    print(f"   ✓ Data dimensions: 2D (customers={data_normalized.shape[0]}, features={data_normalized.shape[1]})")
    print(f"   ✓ Value range (normalized): [{data_normalized.min():.3f}, {data_normalized.max():.3f}]")

    # Display sample statistics
    print(f"\n   Original data sample (first 3 customers):")
    print(df_original.head(3).to_string(index=False))

    # 2. Save table
    print(f"\n2. Saving customer table:")
    save_customer_table(df_original, 'example/customer_data_preview.png')

    # 3. Create zero reference
    reference = np.zeros_like(data_normalized)
    print(f"\n3. Reference (all zeros):")
    print(f"   ✓ Reference shape: {reference.shape}")
    print(f"   ✓ Meaning: features imputed will take value 0")

    # 4. Select one customer for demonstration
    customer_idx = 0
    x = data_normalized[customer_idx:customer_idx+1, :]  # shape: (1, 4)
    ref = reference[customer_idx:customer_idx+1, :]      # shape: (1, 4)

    print(f"\n4. Selecting a customer for imputation:")
    print(f"   ✓ Customer index selected: {customer_idx}")
    print(f"   ✓ Original data for this customer:")
    for i, (fname, val) in enumerate(zip(feature_names, df_original.iloc[customer_idx])):
        print(f"      - {fname}: {val:.2f}")

    # 5. Create three coalition scenarios
    print(f"\n5. Creating three Coalition Matrix scenarios:")
    print(f"   Note: Coalition acts directly on the 4 feature dimensions (no PatchGrouping used)")

    scenarios = create_coalition_scenarios(n_features)

    for name, S_matrix in scenarios.items():
        n_kept = S_matrix.sum()
        percentage = (n_kept / n_features) * 100
        kept_features = [feature_names[i] for i, keep in enumerate(S_matrix[0]) if keep]

        print(f"   ✓ {name}:")
        print(f"      - Shape: {S_matrix.shape}")
        print(f"      - Dtype: {S_matrix.dtype} {'✅' if S_matrix.dtype == bool else '❌'}")
        print(f"      - Features kept: {n_kept}/{n_features} ({percentage:.0f}%)")
        if kept_features:
            print(f"      - Which kept: {', '.join(kept_features)}")

    # 6. Perform imputation
    print(f"\n6. Performing imputation:")

    x_flat = x.flatten()
    ref_flat = ref.flatten()

    imputed_data_dict = {}

    for scenario_name, S_matrix in scenarios.items():
        print(f"\n   Processing {scenario_name}...")

        S = CoalitionMatrix(S_matrix)
        imputer = BaselineImputer(reference=ref_flat, x=x_flat)
        imputed_flat = imputer.impute(S)
        imputed = imputed_flat.reshape(1, n_features)

        imputed_data_dict[scenario_name] = imputed

        print(f"      ✓ Imputed shape: {imputed.shape}")
        print(f"      ✓ Imputed values: {imputed[0]}")

    # 7. Verify results
    print(f"\n7. Verifying results:")

    diff_1 = np.abs(imputed_data_dict['scenario_1_all_kept'] - x).mean()
    print(f"   ✓ Scenario 1 (All kept) difference from original: {diff_1:.6e}")
    assert diff_1 < 1e-10, "Scenario 1 should be identical to original data"

    diff_3 = np.abs(imputed_data_dict['scenario_3_all_imputed']).mean()
    print(f"   ✓ Scenario 3 (All imputed) mean value: {diff_3:.6e}")
    assert diff_3 < 1e-10, "Scenario 3 should be all zeros"

    partial = imputed_data_dict['scenario_2_partial'][0]
    partial_coalition = scenarios['scenario_2_partial'][0]

    kept_ratio = partial_coalition.sum() / len(partial_coalition)
    kept_features_values = partial[partial_coalition]
    original_kept_features = x[0, partial_coalition]
    diff_2_kept = np.abs(kept_features_values - original_kept_features).mean()

    imputed_features_values = partial[~partial_coalition]
    diff_2_imputed = np.abs(imputed_features_values).mean()

    print(f"   ✓ Scenario 2 actual kept ratio: {kept_ratio:.0%}")
    print(f"   ✓ Scenario 2 kept features: {[feature_names[i] for i, k in enumerate(partial_coalition) if k]}")
    print(f"   ✓ Scenario 2 imputed features: {[feature_names[i] for i, k in enumerate(partial_coalition) if not k]}")
    print(f"   ✓ Scenario 2 difference (kept vs original): {diff_2_kept:.6e}")
    print(f"   ✓ Scenario 2 mean value (imputed features): {diff_2_imputed:.6e}")

    if len(kept_features_values) > 0:
        assert diff_2_kept < 1e-10, "Kept features should match original values"
    if len(imputed_features_values) > 0:
        assert diff_2_imputed < 1e-10, "Imputed features should be zero"

    # 8. Impute all customers for visualization
    print(f"\n8. Imputing all {n_customers} customers for visualization...")

    all_customers_imputed = {}

    for scenario_name, S_matrix in scenarios.items():
        S = CoalitionMatrix(S_matrix)
        all_imputed = []
        for i in range(n_customers):
            customer_data = data_normalized[i, :]
            customer_ref = reference[i, :]
            imputer = BaselineImputer(reference=customer_ref, x=customer_data)
            imputed = imputer.impute(S)
            all_imputed.append(imputed[0])
        all_customers_imputed[scenario_name] = np.array(all_imputed)

    print(f"   ✓ Completed imputation for all customers")

    # 9. Visualization
    print(f"\n9. Visualizing results...")
    visualize_results(
        data_normalized,
        all_customers_imputed,
        feature_names,
        scenarios,
        save_path='example/tabular_imputation_demo.png'
    )

    print("\n" + "=" * 80)
    print("✅ Demonstration complete!")
    print("=" * 80)
    print("\nSummary:")
    print("  • Data format: 2D tabular (20 customers × 4 features)")
    print("  • Features: age, income, credit_score, years_employed")
    print("  • Coalition Matrix: Boolean dtype ✅")
    print("  • Coalition acts directly on feature level ✅")
    print("  • Three scenarios demonstrated ✅")
    print("  • All verifications passed ✅")
    print(f"\nOutput files:")
    print(f"  • customer_data_preview.csv - full dataset (20 customers)")
    print(f"  • customer_data_preview.png - data table preview")
    print(f"  • tabular_imputation_demo.png - imputation visualization")


if __name__ == "__main__":
    main()
