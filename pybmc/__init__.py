# This file makes the `pybmc` directory a package.
"""
pybmc: Bayesian Model Combination toolkit

Classes:
- Model: A model defined by input/output data
- Dataset: Handles loading and preparing nuclear model datasets
- BayesianModelCombination: Combines models using Bayesian inference
"""

from .data import Dataset
from .bmc import BayesianModelCombination
from .inference_utils import gibbs_sampler, gibbs_sampler_simplex, USVt_hat_extraction
from .sampling_utils import coverage


__all__ = [
    "Model",
    "Dataset",
    "BayesianModelCombination",
    "gibbs_sampler",
    "gibbs_sampler_simplex",
    "USVt_hat_extraction",
    "coverage",
]
