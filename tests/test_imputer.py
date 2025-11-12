"""Unit tests for the imputer package.

Tests cover:
1. CoalitionMatrix creation and validation
2. Baseline imputation across all backends
3. Marginal imputation across all backends
4. Model integration
5. Edge cases and error handling
"""

import numpy as np
import pytest

from imputer import BaselineImputer, CoalitionMatrix, MarginalImputer


class TestCoalitionMatrix:
    """Tests for CoalitionMatrix class."""
    
    def test_creation(self):
        """Test basic creation of coalition matrix."""
        S_array = np.array([[1, 1, 0], [1, 0, 1]])
        S = CoalitionMatrix(S_array)
        
        assert S.n_coalitions == 2
        assert S.n_features == 3
        assert np.array_equal(S.matrix, S_array)
    
    def test_single_coalition(self):
        """Test with a single coalition."""
        S = CoalitionMatrix(np.array([[1, 0, 1, 0]]))
        
        assert S.n_coalitions == 1
        assert S.n_features == 4


class TestBaselineImputerNumpy:
    """Tests for BaselineImputer with NumPy backend."""
    
    def test_basic_imputation(self):
        """Test basic baseline imputation."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        reference = np.array([0.0, 0.0, 0.0, 0.0])
        S = CoalitionMatrix(np.array([
            [1, 1, 0, 0],  # Keep first 2
            [1, 0, 1, 0],  # Keep 1st and 3rd
            [0, 0, 0, 0],  # Impute all
        ]))
        
        imputer = BaselineImputer(reference=reference, x=x)
        result = imputer.impute(S)
        
        expected = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        
        assert result.shape == (3, 4)
        assert np.allclose(result, expected)
    
    def test_no_imputation(self):
        """Test when all features are kept."""
        x = np.array([1.0, 2.0, 3.0])
        reference = np.array([9.0, 9.0, 9.0])
        S = CoalitionMatrix(np.array([[1, 1, 1]]))
        
        imputer = BaselineImputer(reference=reference, x=x)
        result = imputer.impute(S)
        
        assert np.allclose(result, x)
    
    def test_full_imputation(self):
        """Test when all features are imputed."""
        x = np.array([1.0, 2.0, 3.0])
        reference = np.array([9.0, 9.0, 9.0])
        S = CoalitionMatrix(np.array([[0, 0, 0]]))
        
        imputer = BaselineImputer(reference=reference, x=x)
        result = imputer.impute(S)
        
        assert np.allclose(result, reference)
    
    def test_error_no_x(self):
        """Test error when x is not provided."""
        reference = np.array([0.0, 0.0, 0.0])
        S = CoalitionMatrix(np.array([[1, 0, 1]]))
        
        imputer = BaselineImputer(reference=reference, x=None)
        
        with pytest.raises(ValueError, match="Data point x must be set"):
            imputer.impute(S)


class TestMarginalImputerNumpy:
    """Tests for MarginalImputer with NumPy backend."""
    
    def test_basic_marginal_imputation(self):
        """Test basic marginal imputation."""
        x = np.array([1.0, 2.0, 3.0])
        data = np.random.randn(100, 3)
        S = CoalitionMatrix(np.array([
            [1, 1, 0],
            [1, 0, 1],
        ]))
        
        imputer = MarginalImputer(data=data, x=x, n_samples=10)
        result = imputer.impute(S)
        
        # Check shape
        assert result.shape == (2, 10, 3)
        
        # Check that kept features match x
        assert np.allclose(result[0, :, 0], x[0])  # Coalition 0, feature 0
        assert np.allclose(result[0, :, 1], x[1])  # Coalition 0, feature 1
        assert np.allclose(result[1, :, 0], x[0])  # Coalition 1, feature 0
        assert np.allclose(result[1, :, 2], x[2])  # Coalition 1, feature 2
    
    def test_marginal_samples_from_data(self):
        """Test that imputed values come from data distribution."""
        x = np.array([999.0, 999.0, 999.0])  # Very different from data
        data = np.random.randn(100, 3)
        S = CoalitionMatrix(np.array([[0, 0, 0]]))  # Impute all
        
        imputer = MarginalImputer(data=data, x=x, n_samples=50)
        result = imputer.impute(S)
        
        # Imputed values should be within data range (roughly)
        # Not exactly equal to x
        assert not np.allclose(result, x)
        
        # Mean should be close to data mean
        data_mean = data.mean(axis=0)
        imputed_mean = result[0].mean(axis=0)
        
        # With 50 samples, should be reasonably close
        assert np.allclose(imputed_mean, data_mean, atol=1.0)


@pytest.mark.parametrize("backend,skip_condition", [
    ("torch", lambda: pytest.importorskip("torch")),
    ("jax", lambda: pytest.importorskip("jax")),
])
class TestBackendCompatibility:
    """Test compatibility across different backends."""
    
    def test_baseline_imputation_consistency(self, backend, skip_condition):
        """Test that all backends produce same results."""
        skip_condition()
        
        # NumPy reference
        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        ref_np = np.array([0.0, 0.0, 0.0, 0.0])
        S_np = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
        
        imputer_np = BaselineImputer(reference=ref_np, x=x_np)
        result_np = imputer_np.impute(CoalitionMatrix(S_np))
        
        # Convert to backend
        if backend == "torch":
            import torch
            x = torch.tensor(x_np)
            ref = torch.tensor(ref_np)
            S_matrix = torch.tensor(S_np)
        elif backend == "jax":
            import jax.numpy as jnp
            x = jnp.array(x_np)
            ref = jnp.array(ref_np)
            S_matrix = jnp.array(S_np)
        
        # Test backend
        imputer = BaselineImputer(reference=ref, x=x)
        result = imputer.impute(CoalitionMatrix(S_matrix))
        
        # Convert back to numpy for comparison
        if backend == "torch":
            result = result.cpu().numpy()
        elif backend == "jax":
            result = np.array(result)
        
        assert np.allclose(result, result_np)


class TestModelIntegration:
    """Tests for integration with models."""
    
    def test_with_simple_model(self):
        """Test imputer with a simple model."""
        
        class SumModel:
            def predict(self, X):
                return np.sum(X, axis=-1)
        
        model = SumModel()
        x = np.array([1.0, 2.0, 3.0, 4.0])
        reference = np.zeros(4)
        S = CoalitionMatrix(np.array([
            [1, 1, 1, 1],  # All features: sum = 10
            [1, 1, 0, 0],  # First 2: sum = 3
            [0, 0, 0, 0],  # None: sum = 0
        ]))
        
        imputer = BaselineImputer(reference=reference, x=x, model=model)
        predictions = imputer(S)
        
        expected = np.array([10.0, 3.0, 0.0])
        assert np.allclose(predictions, expected)
    
    def test_with_custom_predict_fn(self):
        """Test with custom prediction function."""
        
        def custom_predict(model, X):
            # Custom logic: multiply by 2
            return model.predict(X) * 2
        
        class SimpleModel:
            def predict(self, X):
                return np.sum(X, axis=-1)
        
        model = SimpleModel()
        x = np.array([1.0, 2.0])
        reference = np.zeros(2)
        S = CoalitionMatrix(np.array([[1, 1]]))
        
        imputer = BaselineImputer(
            reference=reference,
            x=x,
            model=model,
            predict_fn=custom_predict
        )
        
        predictions = imputer(S)
        # sum([1, 2]) * 2 = 6
        assert np.allclose(predictions, 6.0)
    
    def test_postprocess_marginal(self):
        """Test post-processing with marginal imputer."""
        
        class ArrayModel:
            def predict(self, X):
                # Return array with shape matching input
                return np.sum(X, axis=-1)
        
        model = ArrayModel()
        x = np.array([1.0, 2.0, 3.0])
        data = np.random.randn(100, 3)
        S = CoalitionMatrix(np.array([[1, 1, 1]]))
        
        imputer = MarginalImputer(
            data=data,
            x=x,
            model=model,
            n_samples=10
        )
        
        predictions = imputer(S)
        
        # Should average over samples
        # Original shape would be (1, 10), after postprocess: (1,)
        assert predictions.shape == (1,)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_feature(self):
        """Test with single feature."""
        x = np.array([5.0])
        reference = np.array([0.0])
        S = CoalitionMatrix(np.array([[1], [0]]))
        
        imputer = BaselineImputer(reference=reference, x=x)
        result = imputer.impute(S)
        
        expected = np.array([[5.0], [0.0]])
        assert np.allclose(result, expected)
    
    def test_large_coalition(self):
        """Test with many coalitions."""
        n_features = 10
        n_coalitions = 100
        
        x = np.random.randn(n_features)
        reference = np.zeros(n_features)
        S_matrix = np.random.randint(0, 2, (n_coalitions, n_features))
        S = CoalitionMatrix(S_matrix)
        
        imputer = BaselineImputer(reference=reference, x=x)
        result = imputer.impute(S)
        
        assert result.shape == (n_coalitions, n_features)
    
    def test_shape_mismatch(self):
        """Test error handling for shape mismatch."""
        x = np.array([1.0, 2.0, 3.0])
        reference = np.array([0.0, 0.0])  # Wrong shape!
        S = CoalitionMatrix(np.array([[1, 1, 0]]))
        
        imputer = BaselineImputer(reference=reference, x=x)
        
        with pytest.raises(ValueError, match="doesn't match"):
            imputer.impute(S)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
