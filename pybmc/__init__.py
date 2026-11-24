# This file makes the `pybmc` directory a package.
"""
pybmc: Bayesian Model Combination toolkit

Classes:
- Model: A model defined by input/output data
- Dataset: Handles loading and preparing nuclear model datasets
- BayesianModelCombination: Combines models using Bayesian inference

Error models (see `pybmc.error_models`):
- homoscedastic (constant variance) and six heteroscedastic variants
  whose variance depends on distance in principal-component space
  and/or the spread among model predictions. All of them share one
  likelihood and sampler; homoscedastic is the constant-only case.

Randomness (see `pybmc.rng`):
- All samplers and posterior-predictive draws are driven by a single
  seeded package-wide generator (`DEFAULT_SEED`), so runs are
  reproducible end to end; use `set_seed` to re-seed mid-session.
"""

from .data import Dataset
from .bmc import BayesianModelCombination
from .inference_utils import (
    gibbs_sampler,
    gibbs_sampler_simplex,
    gibbs_sampler_heteroscedastic,
    USVt_hat_extraction,
)
from .sampling_utils import (
    coverage,
    coverage_quality,
    diagnose_coverage_shape,
    mace,
    reduced_chi_square,
    DEFAULT_PREDICTIVE_SEED,
)
from .error_models import (
    VARIANCE_MODELS,
    HeteroscedasticMetrics,
    required_metrics,
    variance_basis,
    variance_parameter_names,
)
from .rng import DEFAULT_SEED, get_rng, set_seed


__all__ = [
    "Model",
    "Dataset",
    "BayesianModelCombination",
    "gibbs_sampler",
    "gibbs_sampler_simplex",
    "gibbs_sampler_heteroscedastic",
    "USVt_hat_extraction",
    "coverage",
    "coverage_quality",
    "diagnose_coverage_shape",
    "mace",
    "reduced_chi_square",
    "DEFAULT_PREDICTIVE_SEED",
    "DEFAULT_SEED",
    "get_rng",
    "set_seed",
    "VARIANCE_MODELS",
    "HeteroscedasticMetrics",
    "required_metrics",
    "variance_basis",
    "variance_parameter_names",
]
