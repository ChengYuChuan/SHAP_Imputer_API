imputer_architecture/
│
├── overview.md                           # Main project documentation
│
├── src/imputer/                        
│   ├── __init__.py                       # Public API
│   │   └── Exports: Imputer, BaselineImputer, MarginalImputer,
│   │              CoalitionMatrix, baseline_impute, etc.
│   │
│   └── core/                             # Core functionality module
│       ├── __init__.py                   # Entry point of core module
│       │
│       ├── base.py                       # [Level 1: Abstraction Layer]
│       │   ├── CoalitionMatrix           # Data structure for coalition matrices
│       │   ├── Imputer                   # Abstract base class
│       │   ├── BaselineImputer           # Baseline imputation implementation
│       │   └── MarginalImputer           # Marginal imputation implementation
│       │
│       ├── implementations.py            # [Level 2: Dispatch Layer]
│       │   ├── baseline_impute()         # Lazy dispatch interface
│       │   ├── marginal_impute()         # Lazy dispatch interface
│       │   ├── compute_mean()            # Helper function
│       │   └── delayed_register()        # Lazy backend registration for PyTorch/JAX
│       │
│       └── backends/                     # [Level 3: Backend Implementation Layer]
│           ├── __init__.py               # Backend package entry
│           │
│           ├── numpy.py                  # NumPy backend (mandatory)
│           │   ├── baseline_impute_numpy()
│           │   ├── marginal_impute_numpy()
│           │   └── compute_mean_numpy()
│           │
│           ├── torch.py                  # PyTorch backend (optional)
│           │   ├── baseline_impute_torch()
│           │   ├── marginal_impute_torch()
│           │   └── compute_mean_torch()
│           │
│           └── jax.py                    # JAX backend (optional)
│               ├── baseline_impute_jax()
│               ├── marginal_impute_jax()
│               └── compute_mean_jax()
│
├── examples/                             # Usage examples
│   └── basic_usage.py                    # Basic usage demo
│       ├── example_numpy()               # NumPy example
│       ├── example_pytorch()             # PyTorch example
│       ├── example_jax()                 # JAX example
│       ├── example_with_model()          # Example with model prediction
│       └── main()                        # Run all examples
│
└── tests/                                # Test suite
    └── test_imputer.py                   # Main test file
        ├── TestCoalitionMatrix           # Tests for CoalitionMatrix
        ├── TestBaselineImputerNumpy      # NumPy baseline tests
        ├── TestMarginalImputerNumpy      # NumPy marginal tests
        ├── TestBackendCompatibility      # Cross-backend consistency tests
        ├── TestModelIntegration          # Model integration tests
        └── TestEdgeCases                 # Edge case tests



User code
  │
  │  imputer = BaselineImputer(reference=ref, x=x, model=model)
  │  predictions = imputer(S)  # calls __call__ method
  │
  ▼
┌─────────────────────────────────────────────────┐
│  Imputer.__call__(S)                            │
│  ┌───────────────────────────────────────────┐  │
│  │ 1. imputed = self.impute(S)              │  │
│  │ 2. predictions = self.predict(imputed)   │  │
│  │ 3. output = self.postprocess(predictions)│  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
  │
  │  Step 1: impute(S)
  ▼
┌─────────────────────────────────────────────────┐
│  BaselineImputer.impute(S)                      │
│  ┌───────────────────────────────────────────┐  │
│  │ from implementations import               │  │
│  │     baseline_impute                       │  │
│  │                                            │  │
│  │ return baseline_impute(                   │  │
│  │     self.x, self.reference, S.matrix)     │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
  │
  │  Calls baseline_impute
  ▼
┌─────────────────────────────────────────────────┐
│  implementations.baseline_impute()              │
│  (lazy dispatch)                                │
│  ┌───────────────────────────────────────────┐  │
│  │ Detects input type                        │  │
│  │   ├─ np.ndarray    → NumPy backend        │  │
│  │   ├─ torch.Tensor  → PyTorch backend      │  │
│  │   └─ jax.Array     → JAX backend          │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
  │
  │  Example: torch.Tensor
  ▼
┌─────────────────────────────────────────────────┐
│  delayed_register triggered                     │
│  ┌───────────────────────────────────────────┐  │
│  │ On first encounter with torch.Tensor:      │  │
│  │ 1. Import backends/torch.py                │  │
│  │ 2. Register baseline_impute_torch()        │  │
│  │ 3. Call baseline_impute_torch(x, ref, S)   │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
  │
  │  Returns imputed data
  ▼
┌─────────────────────────────────────────────────┐
│  backends/torch.py                              │
│  baseline_impute_torch()                        │
│  ┌───────────────────────────────────────────┐  │
│  │ x_expanded = x.unsqueeze(0).expand(...)   │  │
│  │ ref_expanded = ref.unsqueeze(0).expand()  │  │
│  │ return S * x_expanded +                   │  │
│  │        (1-S) * ref_expanded               │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
  │
  │  Returns to Imputer.__call__
  ▼
┌─────────────────────────────────────────────────┐
│  Step 2: self.predict(imputed)                  │
│  ┌───────────────────────────────────────────┐  │
│  │ if self.predict_fn:                       │  │
│  │     return self.predict_fn(model, imputed)│  │
│  │ else:                                     │  │
│  │     return self.model.predict(imputed)    │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
  │
  │  Returns to Imputer.__call__
  ▼
┌─────────────────────────────────────────────────┐
│  Step 3: self.postprocess(predictions)          │
│  ┌───────────────────────────────────────────┐  │
│  │ BaselineImputer:  direct return           │  │
│  │ MarginalImputer:  computes mean           │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
  │
  │  Final result
  ▼
Returned to user
