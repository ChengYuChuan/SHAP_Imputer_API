"""JAX backend implementations for imputation.

This module provides JAX-specific implementations of imputation strategies.
Optimized for XLA compilation, TPU/GPU acceleration, and functional programming.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp


def baseline_impute_jax(
    x: jax.Array,
    reference: jax.Array,
    coalition_matrix: jax.Array,
) -> jax.Array:
    """Baseline imputation for JAX arrays.
    
    Args:
        x: Data point to explain, shape (n_features,)
        reference: Reference values, shape (n_features,)
        coalition_matrix: Boolean coalition matrix (dtype=jnp.bool_), shape (n_coalitions, n_features)
    
    Returns:
        Imputed data, shape (n_coalitions, n_features)
    
    Note:
        - JIT-compilable for performance
        - Supports vmap for automatic vectorization
        - Pure functional implementation
    
    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> ref = jnp.zeros(3)
        >>> S = jnp.array([[1., 1., 0.], [1., 0., 1.]])
        >>> result = baseline_impute_jax(x, ref, S)
        >>> result
        Array([[1., 2., 0.],
               [1., 0., 3.]], dtype=float32)
    """
    import jax.numpy as jnp
    
    # Validate shapes
    n_features = x.shape[0]
    if reference.shape[0] != n_features:
        msg = f"Reference shape {reference.shape} doesn't match x shape {x.shape}"
        raise ValueError(msg)
    if coalition_matrix.shape[1] != n_features:
        msg = f"Coalition matrix has {coalition_matrix.shape[1]} features, expected {n_features}"
        raise ValueError(msg)
    
    n_coalitions = coalition_matrix.shape[0]
    
    # Broadcast x and reference to (n_coalitions, n_features)
    x_broadcast = jnp.tile(x, (n_coalitions, 1))
    reference_broadcast = jnp.tile(reference, (n_coalitions, 1))
    
    # make sure that coalition_matrix is boolean
    if coalition_matrix.dtype != jnp.bool_:
        coalition_matrix = coalition_matrix.astype(jnp.bool_)

    imputed = jnp.where(coalition_matrix, x_broadcast, reference_broadcast)    
    
    return imputed


def marginal_impute_jax(
    x: jax.Array,
    data: jax.Array,
    coalition_matrix: jax.Array,
    n_samples: int,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    """Marginal imputation for JAX arrays.
    
    Args:
        x: Data point to explain, shape (n_features,)
        data: Background dataset, shape (n_background, n_features)
        coalition_matrix: Boolean coalition matrix (dtype=jnp.bool_), shape (n_coalitions, n_features)
        n_samples: Number of samples per coalition
        rng_key: JAX random key. If None, creates a new key with seed 0.
    
    Returns:
        Imputed data, shape (n_coalitions, n_samples, n_features)
    
    Note:
        - JIT-compilable with static n_samples
        - Uses JAX's functional random number generation
        - Can be vmapped over multiple data points
    
    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> data = jax.random.normal(key, (100, 3))
        >>> S = jnp.array([[1., 1., 0.], [1., 0., 1.]])
        >>> result = marginal_impute_jax(x, data, S, n_samples=50, rng_key=key)
        >>> result.shape
        (2, 50, 3)
    """
    import jax
    import jax.numpy as jnp
    
    # Validate shapes
    n_features = x.shape[0]
    n_background, data_features = data.shape
    
    if data_features != n_features:
        msg = f"Data has {data_features} features, expected {n_features}"
        raise ValueError(msg)
    if coalition_matrix.shape[1] != n_features:
        msg = f"Coalition matrix has {coalition_matrix.shape[1]} features, expected {n_features}"
        raise ValueError(msg)
    
    # Create RNG key if not provided
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    n_coalitions = coalition_matrix.shape[0]
    
    # Initialize output: (n_coalitions, n_samples, n_features)
    def impute_coalition(i: int, key: jax.Array) -> jax.Array:
        """Impute a single coalition."""
        coalition = coalition_matrix[i]
        
        # Sample background indices
        indices = jax.random.randint(
            key,
            shape=(n_samples,),
            minval=0,
            maxval=n_background
        )
        
        # Sample from background data: (n_samples, n_features)
        sampled_data = data[indices]
        
        # Broadcast x to (n_samples, n_features)
        x_broadcast = jnp.tile(x, (n_samples, 1))
        
        # Broadcast coalition to (n_samples, n_features)
        coalition_broadcast = jnp.tile(coalition, (n_samples, 1))
        
        return jnp.where(coalition_broadcast, x_broadcast, sampled_data)
    
    # Split RNG key for each coalition
    keys = jax.random.split(rng_key, n_coalitions)
    
    # Use vmap for efficient vectorization
    impute_all = jax.vmap(impute_coalition, in_axes=(0, 0))
    coalition_indices = jnp.arange(n_coalitions)
    imputed = impute_all(coalition_indices, keys)
    
    return imputed


def compute_mean_jax(data: jax.Array, axis: int) -> jax.Array:
    """Compute mean along specified axis for JAX arrays.
    
    Args:
        data: Input array
        axis: Axis along which to compute mean
    
    Returns:
        Mean values along specified axis
    
    Note:
        JIT-compilable and differentiable
    
    Example:
        >>> import jax.numpy as jnp
        >>> data = jax.random.normal(jax.random.PRNGKey(0), (10, 5, 3))
        >>> result = compute_mean_jax(data, axis=1)
        >>> result.shape
        (10, 3)
    """
    import jax.numpy as jnp
    
    return jnp.mean(data, axis=axis)

def compute_median_jax(data: jax.Array, axis: int) -> jax.Array:
    """Compute median along axis for JAX arrays.
    
    Args:
        data: Input array (n_samples, n_features)
        axis: Axis along which to compute median
    
    Returns:
        Median values
    
    Note:
        JIT-compilable and differentiable.
    
    Example:
        >>> import jax.numpy as jnp
        >>> data = jax.random.normal(jax.random.PRNGKey(0), (100, 4))
        >>> medians = compute_median_jax(data, axis=0)
        >>> medians.shape
        (4,)
    """
    import jax.numpy as jnp
    
    return jnp.median(data, axis=axis)


def compute_mode_jax(data: jax.Array, axis: int) -> jax.Array:
    """Compute mode along axis for JAX arrays.
    
    Args:
        data: Input array (n_samples, n_features)
        axis: Axis along which to compute mode
    
    Returns:
        Mode values
    
    Note:
        For continuous data, uses most frequent binned value.
        Not as robust as scipy.stats.mode.
    
    Warning:
        This is an approximation for JAX compatibility.
        Consider rounding continuous data before calling.
    
    Example:
        >>> import jax.numpy as jnp
        >>> data = jax.random.randint(
        ...     jax.random.PRNGKey(0), (100, 4), 0, 5
        ... )
        >>> modes = compute_mode_jax(data, axis=0)
        >>> modes.shape
        (4,)
    """
    import jax.numpy as jnp
    from jax.scipy import stats as jax_stats
    
    # JAX doesn't have built-in mode, use workaround
    # For each feature, find most common value
    def mode_1d(x):
        # Round to handle floating point
        x_int = jnp.round(x).astype(jnp.int32)
        values, counts = jnp.unique(x_int, return_counts=True)
        return values[jnp.argmax(counts)].astype(x.dtype)
    
    # Apply along axis
    return jnp.apply_along_axis(mode_1d, axis, data)