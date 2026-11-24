# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `pybmc.rng` module: a single seeded package-wide random generator
  (`DEFAULT_SEED`, `get_rng`, `set_seed`). All MCMC samplers now draw
  from it by default (previously training used the unseeded legacy
  global RNG while only predictions were seeded), and every sampler
  accepts an explicit `seed` argument (also available as the `'seed'`
  training option of `BayesianModelCombination.train()`)
- Heteroscedastic error models: the noise variance can now depend on the
  distance from the training data in principal-component space
  (`pc_dist`) and/or on the disagreement among model predictions
  (`model_var`), linearly or quadratically (new `error_model` parameter
  of `BayesianModelCombination`, new `pybmc.error_models` module and
  `gibbs_sampler_heteroscedastic` Gibbs-within-Metropolis sampler with
  burn-in proposal adaptation toward a target acceptance rate)
- Posterior predictive sampling with per-point variances
  (`rndm_m_heteroscedastic_calculator`)
- Calibration diagnostics: `coverage_quality` (mean |empirical - nominal|
  coverage) and `diagnose_coverage_shape` (under-/over-dispersion
  classification)
- `mh_acceptance_rate_` attribute on `BayesianModelCombination` after
  heteroscedastic training

### Fixed
- `get_weights()` now slices the coefficient columns by the number of
  kept components instead of assuming a single trailing noise parameter
- `evaluate()` now excludes points without truth values, as documented
- Comprehensive docstrings for all public classes and functions
- CONTRIBUTING.md with contribution guidelines
- CHANGELOG.md to track project changes
- AUTHORS.md to credit contributors
- Improved README with better documentation and examples

### Changed
- Unified the error-model likelihoods: the homoscedastic model is now
  trained and predicted as the constant-only special case of the
  heteroscedastic machinery (`gibbs_sampler_heteroscedastic` with a
  constant variance basis) instead of a separate implementation;
  `gibbs_sampler` and `rndm_m_random_calculator` remain as thin
  wrappers around the unified code paths
- All samplers now parametrize the noise on the variance (sigma^2)
  scale: posterior samples store `sigma^2` in the trailing column(s)
  for every error model, including the simplex sampler (previously the
  homoscedastic and simplex samplers stored `sigma`)
- Homoscedastic training now uses a Gamma prior on `sigma^2`
  (`prior_spec`) like the other error models; `nu0_chosen` and
  `sigma20_chosen` now apply to the simplex sampler only
- Variance floors are applied only where positivity is not guaranteed
  by construction (the sampler's data-derived initial value and
  prediction-time variances of extrapolated points, both using the
  single `VARIANCE_FLOOR` constant); the redundant hardcoded floors
  inside the sampling loops (1e-6 homoscedastic / 1e-9 heteroscedastic)
  were removed and the sampler now validates that the variance basis is
  non-negative
- License changed from GPL-3.0 to MIT

### Fixed
- The simplex sampler's Metropolis acceptance ratio was missing the
  factor 1/2 of the Gaussian log-likelihood (it used
  `exp(-dSSR/sigma^2)` instead of `exp(-dSSR/(2 sigma^2))`), slightly
  over-concentrating the constrained posterior

### Changed (docs)
- Updated API documentation in docs/api_reference.md
- Improved usage examples in docs/usage.md
- Standardized docstrings to Google style throughout the codebase

### Fixed
- Minor bug fixes in orthogonalization implementation
- Improved error handling in data loading functions

## [0.1.0] - 2025-07-24

### Added
- Initial release of pybmc
- Core functionality for Bayesian model combination
- Data loading and preprocessing capabilities
- Orthogonalization using SVD
- Gibbs sampling implementation
- Prediction with uncertainty quantification
- Basic documentation structure
