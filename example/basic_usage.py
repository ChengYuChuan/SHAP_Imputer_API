"""Examples demonstrating the imputer package with different backends.

This script shows how to use the imputer with NumPy, PyTorch, and JAX.
"""

import numpy as np


def example_numpy():
    """Example using NumPy backend."""
    print("=" * 60)
    print("NumPy Backend Example")
    print("=" * 60)
    
    from imputer import BaselineImputer, CoalitionMatrix, MarginalImputer
    
    # Create sample data
    x = np.array([1.0, 2.0, 3.0, 4.0])
    reference = np.array([0.0, 0.0, 0.0, 0.0])
    background_data = np.random.randn(100, 4)
    
    # Create coalition matrix (2 coalitions)
    # Coalition 1: Keep features 0,1 (set to 1), impute features 2,3 (set to 0)
    # Coalition 2: Keep features 0,2 (set to 1), impute features 1,3 (set to 0)
    S_matrix = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0]
    ])
    S = CoalitionMatrix(S_matrix)
    
    # Example 1: Baseline Imputation
    print("\n1. Baseline Imputation:")
    print(f"   Original x: {x}")
    print(f"   Reference:  {reference}")
    print(f"   Coalition matrix:\n{S_matrix}")
    
    baseline_imputer = BaselineImputer(reference=reference, x=x)
    imputed = baseline_imputer.impute(S)
    print(f"   Imputed data:\n{imputed}")
    
    # Example 2: Marginal Imputation
    print("\n2. Marginal Imputation:")
    print(f"   Background data shape: {background_data.shape}")
    
    marginal_imputer = MarginalImputer(data=background_data, x=x, n_samples=10)
    imputed_marginal = marginal_imputer.impute(S)
    print(f"   Imputed data shape: {imputed_marginal.shape}")
    print(f"   (n_coalitions, n_samples, n_features) = {imputed_marginal.shape}")
    print(f"   First coalition, first sample: {imputed_marginal[0, 0, :]}")


def example_pytorch():
    """Example using PyTorch backend."""
    try:
        import torch
    except ImportError:
        print("\nPyTorch not installed, skipping PyTorch example")
        return
    
    print("\n" + "=" * 60)
    print("PyTorch Backend Example")
    print("=" * 60)
    
    from imputer import BaselineImputer, CoalitionMatrix
    
    # Create sample data (on GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    reference = torch.zeros(4, device=device)
    
    S_matrix = torch.tensor([
        [1., 1., 0., 0.],
        [1., 0., 1., 0.]
    ], device=device)
    S = CoalitionMatrix(S_matrix)
    
    print("\nBaseline Imputation with PyTorch:")
    print(f"   Original x: {x}")
    print(f"   Reference:  {reference}")
    
    baseline_imputer = BaselineImputer(reference=reference, x=x)
    imputed = baseline_imputer.impute(S)
    print(f"   Imputed data:\n{imputed}")
    print(f"   Device: {imputed.device}")


def example_jax():
    """Example using JAX backend."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("\nJAX not installed, skipping JAX example")
        return
    
    print("\n" + "=" * 60)
    print("JAX Backend Example")
    print("=" * 60)
    
    from imputer import BaselineImputer, CoalitionMatrix
    
    # Create sample data
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    reference = jnp.zeros(4)
    
    S_matrix = jnp.array([
        [1., 1., 0., 0.],
        [1., 0., 1., 0.]
    ])
    S = CoalitionMatrix(S_matrix)
    
    print("\nBaseline Imputation with JAX:")
    print(f"   Original x: {x}")
    print(f"   Reference:  {reference}")
    
    baseline_imputer = BaselineImputer(reference=reference, x=x)
    imputed = baseline_imputer.impute(S)
    print(f"   Imputed data:\n{imputed}")
    
    print("\n   JIT-compiled version:")
    
    # way1: manual JIT
    @jax.jit
    def baseline_impute_jit(x, ref, S):
        n_coalitions = S.shape[0]
        x_bc = jnp.tile(x, (n_coalitions, 1))
        ref_bc = jnp.tile(ref, (n_coalitions, 1))
        return S * x_bc + (1 - S) * ref_bc
    
    # first run to compile
    imputed_jit = baseline_impute_jit(x, reference, S_matrix)
    print(f"   Result (JIT): {imputed_jit}")
    
    # way2: use imputer  JIT 
    print("\n   Using Imputer (automatically uses JAX backend):")
    result = baseline_imputer.impute(S)
    print(f"   Result: {result}")


def example_with_model():
    """Example with model prediction."""
    print("\n" + "=" * 60)
    print("Example with Model Prediction")
    print("=" * 60)
    
    from imputer import BaselineImputer, CoalitionMatrix
    
    # Create a simple model
    class SimpleModel:
        def predict(self, X):
            # Sum all features
            return np.sum(X, axis=-1)
    
    model = SimpleModel()
    
    # Setup
    x = np.array([1.0, 2.0, 3.0, 4.0])
    reference = np.zeros(4)
    S_matrix = np.array([
        [1, 1, 0, 0],  # Keep first 2 features
        [1, 0, 1, 0],  # Keep 1st and 3rd features
        [1, 1, 1, 1],  # Keep all features
        [0, 0, 0, 0],  # Impute all features
    ])
    S = CoalitionMatrix(S_matrix)
    
    # Create imputer with model
    imputer = BaselineImputer(reference=reference, x=x, model=model)
    
    # Use __call__ to get predictions
    predictions = imputer(S)
    
    print("\nModel predictions for different coalitions:")
    print(f"   Coalition matrix:\n{S_matrix}")
    print(f"   Predictions: {predictions}")
    print(f"\n   Explanation:")
    print(f"   - Coalition [1,1,0,0]: sum([1,2,0,0]) = {predictions[0]}")
    print(f"   - Coalition [1,0,1,0]: sum([1,0,3,0]) = {predictions[1]}")
    print(f"   - Coalition [1,1,1,1]: sum([1,2,3,4]) = {predictions[2]}")
    print(f"   - Coalition [0,0,0,0]: sum([0,0,0,0]) = {predictions[3]}")


def main():
    """Run all examples."""
    print("Imputer Package Examples")
    print("=" * 60)
    
    example_numpy()
    example_pytorch()
    example_jax()
    example_with_model()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
