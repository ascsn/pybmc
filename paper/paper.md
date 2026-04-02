750-1750 words
https://joss.readthedocs.io/en/latest/paper.html
https://joss.readthedocs.io/en/latest/example_paper.html

---
title: "pybmc: A Python package for Bayesian model combination"

tags:
  - Python
  - Bayesian inference
  - Machine learning
  - Uncertainty quantification

authors:
  - name: Troy Dasher
    orcid: 0009-0001-4814-2185
    affiliation: "1, 2"
    
  - name: Kyle Godbey
    orcid: 0000-0003-0622-3646
    affiliation: "1"
    
  - name: An Le
    corresponding: true
    affiliation: "1, 2"

affiliations:
  - name: Facility for Rare Isotope Beams, Michigan State University, East Lansing, MI 48824, USA\\\\
    index: 1
    
  - name: Department of Physics and Astronomy, Michigan State University, East Lansing, MI 48824, USA
    index: 2

date: 2026-01-21

bibliography: paper.bib
---


# Summary
Bayesian model combination (BMC) provides a principled framework for combining predictions from multiple models while quantifying uncertainty. Unlike simple averaging methods, BMC explicitly accounts for correlations among models and infers optimal model weights through Bayesian inference, producing predictions with calibrated uncertainty estimates.

The pybmc package implements this methodology in Python, allowing researchers to apply Bayesian model combination to ensembles of predictive models defined on a shared domain. The package is designed for applications in which different theoretical or computational models produce systematically different predictions.

By providing a lightweight and flexible interface for constructing model ensembles and generating uncertainty-aware predictions, pybmc makes Bayesian model combination accessible for research workflows. While motivated by applications in nuclear physics, the approach is general and applicable to a broad range of problems involving ensemble modeling and uncertainty-aware prediction.


# Statement of need - An
A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work.

# State of the fields
This package represents the first user-focused software implementation of the model orthogonalization and combination strategy detailed in [@PhysRevResearch.6.033266]. While there exist other Bayesian model mixing software, including the Taweret software package [@ingles2024taweret], the current package is primarily designed to be used in scenarios where practitioners have precomputed databases of model predictions (with or without uncertainties) and wish to combine the results efficiently without needing to run additional simulations. This is particularly beneficial in nuclear physics, for instance, where global calculations of nuclear properties for all possible isotopes are computationally demanding. 

# Bayesian Model Combination - Pablo
Bayesian model combination (BMC) provides a framework for constructing a single predictive model from a collection of $m$ existing models $\mathcal{M}_k$, each producing predictions $y_k(x)$ for an observable defined over a common domain. The objective is to leverage the collective information contained in the ensemble while accounting for correlations and redundancies among models, which often arise when models share theoretical assumptions or calibration data.

We begin by assembling the predictions of the $m$ models over $n$ input locations into a matrix $X \in \mathbb{R}^{n \times m}$, where each column corresponds to a model and each row corresponds to a point in the domain. From this matrix, we define the ensemble mean prediction $\phi_0(x_i) = \frac{1}{m} \sum_{k=1}^{m} y_k(x_i)$, which captures the common structure shared across models. Subtracting this mean from the model predictions produces a centered representation that isolates the deviations between models.

To identify the dominant patterns in these deviations, we perform a singular value decomposition (SVD) of the centered matrix and retain a truncated set of components. This procedure yields an orthogonal basis ${\phi_j(x)}_{j=1}^r$ that spans the principal directions of variability among the models, with $r \leq m$ controlling the effective dimensionality of the representation. This step reduces the dimensionality of the problem and mitigates the impact of redundant or highly correlated models by filtering out directions associated with minor variations.

The combined model is then constructed as a linear expansion in this reduced basis, $f^\dagger(x; \mathbf{b}) = \phi_0(x) + \sum_{j=1}^{r} b_j , \phi_j(x)$, where $\mathbf{b} = (b_1, \ldots, b_r)$ are coefficients to be inferred from data. Because each basis function $\phi_j$ is itself a linear combination of the original models, this representation defines an implicit combination of the model ensemble.

The relationship between the combined model and experimental observations $y_i$ is described through $y_i = f^\dagger(x_i; \mathbf{b}) + \epsilon_i$, with $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$, where $\sigma$ represents the typical scale of the discrepancy between the model and the data and is inferred within the Bayesian framework.

Given a set of observations $\mathbf{y}$, the likelihood function takes the form $p(\mathbf{y} \mid \mathbf{b}, \sigma) \propto \prod_{i=1}^{n} \frac{1}{\sigma} \exp\left( -\frac{\left(f^\dagger(x_i; \mathbf{b}) - y_i\right)^2}{2\sigma^2} \right)$. Combining this likelihood with suitable prior distributions for $\mathbf{b}$ and $\sigma$ yields the posterior distribution $p(\mathbf{b}, \sigma \mid \mathbf{y})$, which is sampled in pybmc using a Gibbs sampling procedure.

Predictions at new inputs are obtained by evaluating the combined model across posterior samples, resulting in a posterior predictive distribution $p(y(x) \mid \mathbf{y}) = \int p(y(x) \mid \mathbf{b}, \sigma), p(\mathbf{b}, \sigma \mid \mathbf{y}) , d\mathbf{b}, d\sigma$, from which summary statistics such as median predictions and credible intervals can be computed.

This formulation allows the BMC approach to reduce the effective dimensionality of the model space through orthogonalization, limit the impact of correlated or redundant models, and produce uncertainty-calibrated predictions by propagating posterior uncertainty through the combined model. A more detailed discussion of the methodology can be found in [@PhysRevResearch.6.033266].


# Software design

## Class architecture

pybmc is organized around two core classes that separate data management from Bayesian inference, supported by small utility modules that implement the numerical algorithms.

The `Dataset` class (module `pybmc.data`) encapsulates all data access and preprocessing. It loads model outputs and (optionally) experimental truth values from HDF5 or CSV files into synchronized, `pandas` DataFrames. Users specify a list of model identifiers, a list of physical properties (e.g., binding energy or charge radius), and the domain columns (e.g., nucleon numbers `N` and `Z`). `Dataset.load_data` then aligns the models on their common domain and, when a `truth_column_name` is supplied, left‑joins the truth data so that experimental information can live on a smaller domain than the model ensemble. Additional methods provide convenient views of the loaded data, domain‑aware filtering, and utilities for constructing training/validation/test splits.

The `BayesianModelCombination` class (module `pybmc.bmc`) implements the Bayesian model combination workflow. It operates on the synchronized DataFrames produced by `Dataset` and keeps track of the list of models to be combined and the column containing the experimental truth. The class exposes three main operations: `orthogonalize`, which constructs an orthogonal basis of model differences by applying singular value decomposition (SVD) to centered model predictions; `train`, which performs Gibbs sampling on the truncated SVD coefficients to obtain posterior samples of the combination weights and residual uncertainty; and `predict` / `evaluate`, which turn those samples into posterior predictive draws, credible intervals, and coverage diagnostics.

Low‑level numerical routines are factored into the `pybmc.inference_utils` and `pybmc.sampling_utils` modules. The former provides SVD post‑processing (`USVt_hat_extraction`) and a Gibbs sampler for Bayesian linear regression, while the latter provides utilities to turn posterior samples into predictive draws and credible intervals (`rndm_m_random_calculator`) and to compute coverage statistics (`coverage`). Keeping these routines outside the main classes keeps the public API small and makes it straightforward to swap or extend samplers and diagnostic tools without altering the high‑level workflow.

## Workflow pipeline

The overall workflow pipeline is designed to mirror typical scientific usage, starting from precomputed model tables and ending with uncertainty‑quantified predictions and diagnostics.

1. **Data loading and alignment.** The user constructs a `Dataset` pointing to a single HDF5 or CSV file that contains multiple predictive models and, optionally, an experimental or “truth” column. Calling `Dataset.load_data` with a list of model names, property keys, and domain columns returns a dictionary of aligned DataFrames, one per property, with columns for the domain and each model. When a truth column is specified, it is left‑joined to the common model domain so that the training set can be restricted to the subset of points with experimental data while still enabling prediction across the full model domain.

2. **Domain selection and splitting.** The same `Dataset` instance provides tools to select physically relevant subsets (for example, restricting to a range in `N` and `Z`) and to partition the data into training, validation, and test sets. The `split_data` method supports both simple random splits and an “inside‑to‑outside” algorithm based on distances in the (`N`, `Z`) plane, which is useful in nuclear physics applications where one wishes to train near stability and test predictions further from known nuclei. This design keeps domain knowledge and splitting strategies in the data layer rather than in the inference layer.

3. **Orthogonalization of model predictions.** The user then initializes a `BayesianModelCombination` instance with the list of models to be combined, the data dictionary returned by `Dataset`, and the name of the truth column. The `orthogonalize` method centers each model’s predictions around the ensemble mean and applies SVD to obtain an orthogonal basis of model differences. The user controls the effective model complexity through a `components_kept` parameter, which specifies how many singular vectors to retain. This step improves numerical stability and makes the subsequent Bayesian inference operate on a low‑dimensional set of latent components rather than on the raw model table.

4. **Bayesian inference.** Given the orthogonalized design matrix, the `train` method runs a Gibbs sampler (implemented in `pybmc.inference_utils.gibbs_sampler`) to draw posterior samples for the combination weights and a residual noise scale. The method accepts a `training_options` dictionary that allows users to control the number of iterations and the prior hyperparameters for the weights and variance. This separation between high‑level workflow (`BayesianModelCombination.train`) and low‑level Markov Chain Monte Carlo implementation keeps the public interface simple while leaving room for more advanced samplers.

5. **Prediction and diagnostics.** Once the sampler has converged, the `predict` method projects the posterior weights back onto the original model space to generate posterior predictive draws for any property present in the original data dictionary, returning both the raw draws and summary DataFrames with median predictions and 95% credible intervals. The `evaluate` method uses the same predictive draws to compute empirical coverage over a grid of credible interval widths, providing a simple, model‑agnostic diagnostic of how well the Bayesian model combination captures the experimental truth where it is available.

This layered architecture-data access and domain logic in `Dataset`, orthogonalization and sampling in `BayesianModelCombination`, and numerical routines in focused helper modules—was chosen to reflect how practitioners already work with precomputed model tables while making each step of the combination pipeline explicit, testable, and replaceable.

Users can adjust the workflow at several points. For example, users can modify or extend the sampling routines in pybmc.inference_utils (for example, to use the provided simplex‑constrained Gibbs sampler or custom MCMC kernels). The package is deliberately designed to be domain‑agnostic, so that the same architecture can be applied to any application where practitioners need to combine multiple, potentially correlated predictive models into a single, uncertainty‑quantified prediction.

More detailed tutorials, additional examples, and a complete API reference are available in the online documentation:

https://ascsn.github.io/pybmc/docs

# Research impact statement

The original methodology was published in a peer-reviewed journal [@PhysRevResearch.6.033266], and the first application of the methodology has also recently been published in an analysis of new experimental data from the Facility for Rare Isotope Beams [@article].
Once the package was completed, it began replacing the original, bespoke implementation in new scientific works, including a study on Q-alpha trends in superheavy nuclei, the impact of nuclear uncertainties in r-process nucleosynthesis, and a systematic study of charge radii. Students involved in these projects have presented results obtained using the software at the American Physical Society's annual Division of Nuclear Physics meeting and at presentations locally.
The package has also been adopted for release with the Bayesian Analysis for Nuclear Dynamics collaboration's V0.5 release and disseminated broadly to the nuclear physics community.


# AI usage disclosure

Generative AI was used in the creation of a package template early in the development process and during development via inline autocomplete within the developers' editors. No code was adopted without human oversight, and our metric for correctness was defined by detailed numerical comparison to the bespoke implementation already published in [cite].
AI tools were also used to expand the package documentation and write additional tests, with the resulting documentation going through a round of human review for completeness, correctness, and accessibility.

# Acknowledgments

# Citations 




