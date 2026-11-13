---
title: "pybmc: A Python package for Bayesian model combination"

tags:
  - Python
  - Bayesian inference
  - Machine learning
  - Uncertainty quantification

authors:
  - name: Pranav Agarwal
    
  - name: Troy Dasher
    orcid: 0009-0001-4814-2185
    affiliation: "1, 2"
    corresponding: true
    note: "Email: [dashert@msu.edu](mailto:dashert@msu.edu)"

  - name: Pablo Giuliani
    orcid: 0000-0002-8145-0745
    affiliation: "1"
    note: "Email: [giulia27@msu.edu](mailto:giulia27@msu.edu)"

  - name: Kyle Godbey
    orcid: 0000-0003-0622-3646
    affiliation: "1"
    note: "Email: [godbey@frib.msu.edu](mailto:godbey@frib.msu.edu)"

  - name: Bailey Knight
    orcid: 0009-0004-6777-1034
    affiliation: "1, 2"
    note: "Email: [knight@frib.msu.edu]"

  - name: Sudhanva Lalit
    orcid: 0000-0001-7758-492X
    affiliation: "1"

  - name: An Le
    affiliation: "1, 2"
    note: "Email: [lean2@msu.edu](mailto:lean2@msu.edu)"


affiliations:
  - name: Facility for Rare Isotope Beams, Michigan State University, East Lansing, MI 48824, USA
    index: 1
  - name: Department of Physics and Astronomy, Michigan State University, East Lansing, MI 48824, USA
    index: 2

date: 2026-01-21

bibliography: paper.bib
---


# Summary
Bayesian model combination (BMC) provides a framework for combining predictions from multiple models while quantifying uncertainty. Unlike Bayesian averaging methods, BMC explicitly accounts for correlations among models and infers optimal model weights through Bayesian inference, producing predictions with calibrated uncertainty estimates.

The pybmc package implements this methodology in Python, allowing researchers to easily apply BMC to ensembles of predictive models defined on a shared domain. The package is designed for applications in which different theoretical or computational models produce systematically different predictions.

By providing a lightweight and flexible interface for constructing model ensembles and generating uncertainty-aware predictions, pybmc makes BMC accessible for research workflows. While motivated by applications in nuclear physics, the approach is general and applicable to a broad range of problems involving ensemble modeling and uncertainty-aware prediction.


# Statement of need 
Across diverse scientific disciplines, it is increasingly common to encounter problems that rely on different theoretical models, which often yield systematically different predictions for the same observable. In these instances, the challenge is not only to compare models, but to combine them in a statistically meaningful and reproducible way that accounts for correlations, redundancy, and uncertainty. BMC offers a useful framework to address this problem by constructing weighted predictions from multiple models using dimensionality reduction and Bayesian inference. 

Practical applications of BMC are currently limited by the lack of accessible and convenient software implementations. This software addresses this problem by providing a user-friendly, object-oriented Python framework for BMC that supports the entire workflow. It integrates key steps such as data processing, dimensionality reduction, and Bayesian statistical modeling into a coherent and reusable workflow. By organizing these operations into modular components, the package enables fast implementation of BMC across different datasets, while allowing flexible selection or extension of different components in the workflow.

# State of the fields
This package represents the first user-focused software implementation of the model orthogonalization and combination strategy detailed in [@PhysRevResearch.6.033266]. While existing scientific Python tools such as scikit-learn [@scikit-learn] and PyMC [@pymc2023] provide components relevant to BMC, including dimensionality reduction and Bayesian inference, they do not offer an integrated implementation or any BMC workflow. Other Bayesian model mixing packages, such as Taweret [@ingles2024taweret], provide frameworks for combining model predictions using a variety of Bayesian weighting methods. In contrast, pybmc implements a specific approach based on model orthogonalization, using principal component analysis to identify independent patterns of variation among potentially correlated models before performing Bayesian inference. This provides a dedicated implementation of the methodology developed by @PhysRevResearch.6.033266, particularly suited to combining ensembles of correlated theoretical models.

# Bayesian Model Combination
BMC provides a framework for constructing a single predictive model from a collection of $m$ existing models $\mathcal{M}_k$, each producing predictions $M_k(x)$ for an observable defined over a common domain. The objective is to leverage the collective information contained in the ensemble while accounting for correlations and redundancies among models, which often arise when models share theoretical assumptions or calibration data.

We begin by assembling the predictions of the $m$ models over $n$ input locations into a matrix $X \in \mathbb{R}^{n \times m}$, where each column corresponds to a model, and each row corresponds to a point in the domain. From this matrix, we define the ensemble mean prediction:

$\phi_0(x_i) = \frac{1}{m} \sum_{k=1}^{m} M_k(x_i)$, 

which captures the common structure shared across models. Subtracting this mean from the model predictions produces a centered representation that isolates the deviations between models.

To identify the dominant patterns in these deviations, we perform a singular value decomposition (SVD) of the centered matrix and retain a truncated set of components. This procedure yields an orthogonal basis $\phi_j(x)$ that spans the principal directions of variability among the models, with $j =1, ..., p$ with $p\leq m$ controlling the effective dimensionality of the representation. This step reduces the dimensionality of the problem and mitigates the impact of redundant or highly correlated models by filtering out directions associated with minor variations.

The combined model is then constructed as a linear expansion in principal components:

$f^\dagger(x; \mathbf{b}) = \phi_0(x) + \sum_{j=1}^{p} b_j \phi_j(x)$, 

where $\mathbf{b} = (b_1, \ldots, b_r)$ are coefficients to be inferred from data. Because each basis function $\phi_j$ is itself a linear combination of the original models, this representation is itself a (restricted) combination of the model ensemble.

The relationship between the combined model and experimental observations $y_i$ is described through $y_i = f^\dagger(x_i; \mathbf{b}) + \epsilon_i$, with $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$, where $\sigma$ represents the typical scale of the discrepancy between the model and the data and is inferred within the Bayesian framework. Here, we assume a constant discrepancy variance, although extensions of BMC allow this uncertainty to depend on the input domain [@Knight_2026_HeEM_BMC].

Given a set of observations $\mathbf{y}$, the likelihood function takes the form:

$p(\mathbf{y} \mid \mathbf{b}, \sigma) \propto \prod_{i=1}^{n} \frac{1}{\sigma} \exp\left( -\frac{\left(f^\dagger(x_i; \mathbf{b}) - y_i\right)^2}{2\sigma^2} \right)$. 

Combining this likelihood with suitable prior distributions for $\mathbf{b}$ and $\sigma$ yields the posterior distribution $p(\mathbf{b}, \sigma \mid \mathbf{y})$, which is sampled in pybmc using conditional Gibbs updates for the coefficients and Metropolis–Hastings updates for the variance parameters.

Predictions at new inputs are obtained by evaluating the combined model across posterior samples, resulting in a posterior predictive distribution:

$p(y(x) \mid \mathbf{y}) = \int p(y(x) \mid \mathbf{b}, \sigma) p(\mathbf{b}, \sigma \mid \mathbf{y})d\mathbf{b} d\sigma$, 

from which summary statistics such as median predictions and credible intervals can be computed.

This formulation allows the BMC approach to reduce the effective dimensionality of the model space through orthogonalization, limit the impact of correlated or redundant models, and produce uncertainty-calibrated predictions. A more detailed discussion of the methodology can be found in [@PhysRevResearch.6.033266].


# Software design
pybmc is designed to exist as an open-source, user-friendly Python package that is available to the broader scientific community and can be applied to problems in various fields. To accomplish this, pybmc handles precomputed model predictions for a given quantity, allowing researchers to easily and quickly apply BMC to any problem that relies on various theoretical models. Additionally, this allows researchers to use existing calculations without modifying or rerunning the underlying simulators. This approach is useful when simulations are computationally expensive or available only as published prediction tables. 

Furthermore, pybmc is organized around two core classes that separate data management from Bayesian inference, supported by small utility modules that implement the numerical algorithms. This separation allows for each class to be independently adjusted, allowing for a more seamless application of BMC to different scientific datasets.

The `Dataset` class (module `pybmc.data`) is responsible for all data loading, organization, and preprocessing, allowing these operations to exist separately from the Bayesian inference procedure. The class supports input model predictions in HDF5 or CSV format and synchronizes them over a common domain, allowing enabling the combination of models with different prediction coverages. Experimental observations, or `truth_column_name`, are incorporated through a left join, preserving the full synchronized model domain even when experimental data are available only for a subset. This is particularly important for applications in which theoretical predictions extend beyond experimentally measured regions. The class also supports domain-specific filtering and the construction of training, validation, and testing sets, providing a consistent data structure for subsequent inference.

The `BayesianModelCombination` class (module `pybmc.bmc`) implements the BMC workflow. It operates on the synchronized DataFrames produced by `Dataset` and keeps track of the list of models to be combined and the column containing the experimental truth. The class exposes three main operations: `orthogonalize`, which constructs an orthogonal basis of model differences by applying singular value decomposition (SVD) to centered model predictions; `train`, which performs Gibbs sampling on the truncated SVD coefficients to obtain posterior samples of the combination weights and residual uncertainty; and `predict` / `evaluate`, which turn those samples into posterior predictive draws, credible intervals, and coverage diagnostics.

Low‑level numerical routines are factored into the `pybmc.inference_utils` and `pybmc.sampling_utils` modules. The former provides SVD post‑processing (`USVt_hat_extraction`) and a Gibbs sampler for Bayesian linear regression, while the latter provides utilities to turn posterior samples into predictive draws and credible intervals (`rndm_m_random_calculator`) and to compute coverage statistics (`coverage`). Keeping these routines outside the main classes keeps the public API small and makes it straightforward to swap or extend samplers and diagnostic tools without altering the high‑level workflow.

More detailed tutorials, additional examples, and a complete API reference are available in the online documentation:

[https://ascsn.github.io/pybmc/docs](https://github.com/ascsn/pybmc/tree/main/docs)

# Research impact statement

The original methodology was published in a peer-reviewed journal [@PhysRevResearch.6.033266], and the first application of the methodology has also recently been published in an analysis of new experimental data from the Facility for Rare Isotope Beams [@article]. Once the package was completed, it began replacing the original, bespoke implementation in new scientific works, including the extension of BMC to heteroscedastic uncertainties [@Knight_2026_HeEM_BMC]. The package is also being used in ongoing studies of Q-alpha trends in superheavy nuclei, the impact of nuclear uncertainties in r-process nucleosynthesis, and a systematic study of charge radii. Students involved in these projects have presented results obtained using the software at the American Physical Society's annual Division of Nuclear Physics meeting and at local presentations. The package has also been adopted for release with the Bayesian Analysis for Nuclear Dynamics collaboration's (BAND) V0.5 release [@bandframework] and disseminated broadly to the nuclear physics community.


# AI usage disclosure

Generative AI was used in the creation of a package template early in the development process and during development via inline autocomplete within the developers' editors. No code was adopted without human oversight, and our metric for correctness was defined by detailed numerical comparison to the bespoke implementation already published in [@PhysRevResearch.6.033266].
AI tools were also used to expand the package documentation and write additional tests, with the resulting documentation going through a round of human review for completeness, correctness, and accessibility.

# Acknowledgments
This work was supported in part by the National Science Foundation through the Cyberinfrastructure for Sustained Scientific Innovation (CSSI) program under Award No. OAC-2004601 (BAND Collaboration), and by the U.S. Department of Energy, Office of Science, Office of Nuclear Physics under Award No. DE-SC0023688 (Bayesian Framework for Mining of Evaluated Nuclear Mass Data).

# Citations 




