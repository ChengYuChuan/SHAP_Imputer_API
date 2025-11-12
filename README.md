# Imputer Package

The goal of the hackathon is to implement a Python package that allows us to apply game-theoretic concepts such as Shapley Values and Shapley Interactions to machine learning models.

You should implement the issues listed in the repository as thoroughly as possible, ensuring sufficient tests and documentation.

You should also implement as modularly as possible so that the package can be extended at a later time.

As references, you can use [Explaining by Removing: A Unified Framework for Model Explanation](https://jmlr.csail.mit.edu/papers/volume22/20-1316/20-1316.pdf) (Section 4.2) and [Explaining Machine Learning Models with Conditional Shapley Values in R and Python](https://arxiv.org/pdf/2504.01842) (Section 2.2).

## Next Steps

Package management is handled via [uv](https://docs.astral.sh/uv/getting-started/installation/), which is one of the fastest package managers for Python.

After installation, you can install the required packages based on **pyproject.toml** using `uv sync --dev`.

We have already added some required packages; additional necessary packages can be added using `uv add <package>`, see [here](https://docs.astral.sh/uv/getting-started/features/#projects).
We aim for high modularity even in the package management, thus we have different groups for the dependencies.
Those which are core dependencies, needed in every case, are added via `uv add <package>`.
Others which would only be needed in certainty cases, e.g. see the torch dependency group in **pyproject.toml**, are added via `uv add <package> --group <dependency_group_name>`.

Finally, you can start the implementation; issues for this are defined [here](https://github.com/Advueu963/imputer).

## References
Packages (XAI pacakges) that use imputation/masking.

- shap.maskers: https://github.com/shap/shap
- shapiq.imputer: https://github.com/mmschlk/shapiq
- fippy.samplers: https://github.com/gcskoenig/fippy
- sklearn.impute: https://scikit-learn.org/stable/api/sklearn.impute.html
- shapr: https://github.com/NorskRegnesentral/shapr
