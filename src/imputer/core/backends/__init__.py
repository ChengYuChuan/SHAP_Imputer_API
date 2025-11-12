"""Backend implementations for different array libraries.

This package contains backend-specific implementations for:
- NumPy (always available)
- PyTorch (optional)
- JAX (optional)

Implementations are registered via lazy dispatch, so optional dependencies
are only imported when actually used.
"""

# Import NumPy implementations (always available)
from imputer.core.backends import numpy  # noqa: F401

# Optional backends are registered via delayed_register in implementations.py
# They will be imported automatically when needed
