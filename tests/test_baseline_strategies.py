import numpy as np
import pytest
from imputer import BaselineImputer, CoalitionMatrix


class TestBaselineStrategies:
    """Test different imputation strategies."""
    
    def test_constant_strategy(self):
        """Test constant strategy (backward compatible)."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        reference = np.array([0.0, 0.0, 0.0, 0.0])
        S = CoalitionMatrix(np.array([[True, True, False, False]], dtype=bool))
        
        imputer = BaselineImputer(
            strategy="constant",
            reference=reference,
            x=x
        )
        result = imputer.impute(S)
        
        expected = np.array([[1.0, 2.0, 0.0, 0.0]])
        assert np.allclose(result, expected)
    
    def test_mean_strategy(self):
        """Test mean strategy."""
        x = np.array([10.0, 20.0, 30.0, 40.0])
        data = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 9.0, 12.0],
        ])  # Means: [2.0, 4.0, 6.0, 8.0]
        
        S = CoalitionMatrix(np.array([[True, True, False, False]], dtype=bool))
        
        imputer = BaselineImputer(
            strategy="mean",
            data=data,
            x=x
        )
        result = imputer.impute(S)
        
        # Keep first 2 features, impute last 2 with mean
        expected = np.array([[10.0, 20.0, 6.0, 8.0]])
        assert np.allclose(result, expected)
    
    def test_median_strategy(self):
        """Test median strategy."""
        x = np.array([10.0, 20.0, 30.0, 40.0])
        data = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 9.0, 12.0],
            [100.0, 100.0, 100.0, 100.0],  # Outlier
        ])  # Medians: [2.5, 5.0, 7.5, 10.0]
        
        S = CoalitionMatrix(np.array([[True, True, False, False]], dtype=bool))
        
        imputer = BaselineImputer(
            strategy="median",
            data=data,
            x=x
        )
        result = imputer.impute(S)
        
        expected = np.array([[10.0, 20.0, 7.5, 10.0]])
        assert np.allclose(result, expected)
    
    def test_mode_strategy(self):
        """Test mode strategy for categorical data."""
        x = np.array([5, 5, 5, 5])
        data = np.array([
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 2, 1, 3],
            [1, 3, 1, 3],
        ])  # Modes: [1, 2, 1, 2] or [1, 2, 1, 3]
        
        S = CoalitionMatrix(np.array([[True, True, False, False]], dtype=bool))
        
        imputer = BaselineImputer(
            strategy="mode",
            data=data,
            x=x
        )
        result = imputer.impute(S)
        
        # Keep first 2, impute last 2 with mode
        assert result[0, 0] == 5  # Kept
        assert result[0, 1] == 5  # Kept
        assert result[0, 2] == 1  # Mode
        # Note: mode for feature 3 could be 2 or 3 (tie)
    
    def test_backward_compatibility(self):
        """Test that old API still works."""
        x = np.array([1.0, 2.0, 3.0])
        reference = np.zeros(3)
        S = CoalitionMatrix(np.array([[True, False, True]], dtype=bool))
        
        # Old way (should default to constant)
        imputer_old = BaselineImputer(reference=reference, x=x)
        result_old = imputer_old.impute(S)
        
        # New way (explicit)
        imputer_new = BaselineImputer(
            strategy="constant",
            reference=reference,
            x=x
        )
        result_new = imputer_new.impute(S)
        
        assert np.allclose(result_old, result_new)
    
    def test_validation_errors(self):
        """Test validation of strategy inputs."""
        x = np.array([1.0, 2.0, 3.0])
        
        # Constant without reference
        with pytest.raises(ValueError, match="requires 'reference'"):
            BaselineImputer(strategy="constant", x=x)
        
        # Mean without data
        with pytest.raises(ValueError, match="requires 'data'"):
            BaselineImputer(strategy="mean", x=x)
        
        # Median without data
        with pytest.raises(ValueError, match="requires 'data'"):
            BaselineImputer(strategy="median", x=x)


@pytest.mark.parametrize("backend,skip_condition", [
    ("torch", lambda: pytest.importorskip("torch")),
    ("jax", lambda: pytest.importorskip("jax")),
])
class TestStrategiesAcrossBackends:
    """Test strategies work across backends."""
    
    def test_mean_backend_consistency(self, backend, skip_condition):
        """Test mean strategy produces consistent results."""
        skip_condition()
        
        # NumPy reference
        x_np = np.array([10.0, 20.0, 30.0])
        data_np = np.random.randn(100, 3)
        S_np = np.array([[True, False, False]], dtype=bool)
        
        imputer_np = BaselineImputer(strategy="mean", data=data_np, x=x_np)
        result_np = imputer_np.impute(CoalitionMatrix(S_np))
        
        # Convert to backend
        if backend == "torch":
            import torch
            x = torch.tensor(x_np)
            data = torch.tensor(data_np)
            S = torch.tensor(S_np)
        elif backend == "jax":
            import jax.numpy as jnp
            x = jnp.array(x_np)
            data = jnp.array(data_np)
            S = jnp.array(S_np)
        
        imputer = BaselineImputer(strategy="mean", data=data, x=x)
        result = imputer.impute(CoalitionMatrix(S))
        
        # Convert back to NumPy
        if backend == "torch":
            result = result.cpu().numpy()
        elif backend == "jax":
            result = np.array(result)
        
        assert np.allclose(result, result_np, rtol=1e-5)