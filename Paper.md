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
  - name: Facility for Rare Isotope Beams, Michigan State University, East Lansing, MI 48824, USA
    index: 1
  - name: Department of Physics and Astronomy, Michigan State University, East Lansing, MI 48824, USA
    index: 2

date: 2026-01-21

bibliography: paper.bib
---


# Summary
A description of the high-level functionality and purpose of the software for a diverse, non-specialist audience.

pybmc is a Python package for Bayesian model combination (BMC), a Bayesian machine learning framework for combining predictions from multiple correlated models while quantifying predictive uncertainty. Unlike simple averaging, pybmc explicitly accounts for inter-model correlations and learns optimal model weights through Bayesian inference, yielding combined predictions with uncertainty estimates.

The package is designed for scientific applications in which models are defined on a shared domain but differ systematically in their assumptions, approximations, or parameterizations. pybmc provides a flexible API for data handling, orthogonalization, Gibbs sampling, and prediction with uncertainty quantification. 

pybmc is implemented in Python with minimal dependencies and is intended to integrate naturally into existing scientific workflows. While motivated by applications in nuclear physics, the package is applicable to a broad class of problems involving ensemble modeling and uncertainty-aware prediction.


# Statement of need - An
A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work.

# State of the fields - KYLE
A description of how this software compares to other commonly-used packages in the research area. If related tools exist, provide a clear “build vs. contribute” justification explaining your unique scholarly contribution and why existing alternatives are insufficient.

This package represents the first user-focused software implementation of the model orthogonalization and combination strategy detailed in [cite]. While there exist other Bayesian model mixing software, including the Taweret software package [cite Taweret JOSS], the current package is primarily designed to be used in scenarios where practicioners have precomputed databases of model predictions (with or without uncertainties) and wish to combine the results efficiently without needing to run additional simulations. This is particularly beneficial in nuclear physics, for instance, where global calculations of nuclear properties for all possible isotopes is computationally demanding. 

# Software design
An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description.


pybmc is organized to provide a user-friendly, object-oriented API through the use of community tools while keeping the the user interface in Python. It is centered around two classes that separate data handling and the BMC workflow, along with helper functions to assist with the training and uncertainty quantification. This design isolates the two distinct functions that pybmc provides, clarifying the flow from preprocessed model outputs to final BMC estimates.

Data management is handled by the Dataset class, which provides utilities for loading data from external sources, selecting valid domains, and partitioning available observations into training, validation, and testing subsets. This abstraction centralizes data preparation within pybmc, ensuring that the inference workflow operates on consistent and well-defined inputs.

The BMC workflow is encapsulated in the BayesianModelCombination class, which serves as the main area for BMC. Rather than embedding certain functions directly in the class, supporting functionality such as sampling and uncertainty quantification is factored into helper modules. This design keeps the core workflow explicit and composable, emphasizes clarity in the training–prediction–uncertainty pipeline, and allows individual components to be modified or replaced without restructuring the overall package.

# Research impact statement

The original methodology was published in a peer-reviewed journal [cite] and the first application of the methodolgy has also recently been published in an analysis of new experimental data from the Facility for Rare Isotope Beams [cite].
Once the package was completed, it began replacing the original, bespoke implementation in new scientific works including a study on Q-alpha trends in superheavy nuclei, the impact of nuclear uncertainties in r-process nucleosynthesis, and a systematic study of charge radii. Students involved in these projects have presented results obtained using the software at the American Physical Society's annual Division of Nuclear Physics meeting and at presentations locally.
The package has also been adopted for release with the Bayesian Analysis for Nuclear Dynamics collaboration's V0.5 release and disseminated broadly to the nuclear physics community.

# Mathematics (some formal theory here?) - I think we can just include the documentation already on the Github


# Figures (?)

# AI usage disclosure - We should each contribute what we used AI for

Generative AI was used in the creation of a package template early in the development process and during development via inline autocomplete within the developers' editors. No code was adopted without human oversight and our metric for correctness was defined by detailed numerical comparison to the bespoke implementation already published in [cite].
AI tools were also used to expand the package documentation and write additional tests, with the resulting documentation going through a round of human review for completeness, correctness, and accessibility.

# Acknowledgments

# Citations 

Example paper.bib file:

@article{Pearson:2017,
  	url = {http://adsabs.harvard.edu/abs/2017arXiv170304627P},
  	Archiveprefix = {arXiv},
  	Author = {{Pearson}, S. and {Price-Whelan}, A.~M. and {Johnston}, K.~V.},
  	Eprint = {1703.04627},
  	Journal = {ArXiv e-prints},
  	Keywords = {Astrophysics - Astrophysics of Galaxies},
  	Month = mar,
  	Title = {{Gaps in Globular Cluster Streams: Pal 5 and the Galactic Bar}},
  	Year = 2017
}




