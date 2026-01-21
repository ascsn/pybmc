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

bibliography: 
---


# Summary
A description of the high-level functionality and purpose of the software for a diverse, non-specialist audience.

pybmc is a Python package for Bayesian model combination (BMC), a Bayesian machine learning framework for combining predictions from multiple correlated models while quantifying predictive uncertainty. Unlike simple averaging, pybmc explicitly accounts for inter-model correlations and learns optimal model weights through Bayesian inference, yielding combined predictions with uncertainty estimates.

The package is designed for scientific applications in which models are defined on a shared domain but differ systematically in their assumptions, approximations, or parameterizations. pybmc provides a flexible API for data handling, orthogonalization, Gibbs sampling, and prediction with uncertainty quantification. 

pybmc is implemented in Python with minimal dependencies and is intended to integrate naturally into existing scientific workflows. While motivated by applications in nuclear physics, the package is applicable to a broad class of problems involving ensemble modeling and uncertainty-aware prediction.


# Statement of need
A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work.

# State of the fields
A description of how this software compares to other commonly-used packages in the research area. If related tools exist, provide a clear “build vs. contribute” justification explaining your unique scholarly contribution and why existing alternatives are insufficient.

# Software design
An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description.


pybmc is organized to provide a user-friendly, object-oriented API through the use of community tools while keeping the the user interface in Python. It is centered around two classes that separate data handling and the BMC workflow, along with helper functions to assist with the training and uncertainty quantification. This design isolates the two distinct functions that pybmc provides, clarifying the flow from preprocessed model outputs to final BMC estimates.

Data management is handled by the Dataset class, which provides utilities for loading data from external sources, selecting valid domains, and partitioning available observations into training, validation, and testing subsets. This abstraction centralizes data preparation within pybmc, ensuring that the inference workflow operates on consistent and well-defined inputs.

The BMC workflow is encapsulated in the BayesianModelCombination class, which serves as the main area for BMC. Rather than embedding certain functions directly in the class, supporting functionality such as sampling and uncertainty quantification is factored into helper modules. This design keeps the core workflow explicit and composable, emphasizes clarity in the training–prediction–uncertainty pipeline, and allows individual components to be modified or replaced without restructuring the overall package.

# Research impact statement
Evidence of realized impact (publications, external use, integrations) or credible near-term significance (benchmarks, reproducible materials, community-readiness signals). The evidence should be compelling and specific, not aspirational.

# Mathematics (some formal theory here?)


# Figures (?)

# AI usage disclosure
Transparent disclosure of any use of generative AI in the software creation, documentation, or paper authoring. If no AI tools were used, state this explicitly. If AI tools were used, describe how they were used and how the quality and correctness of AI-generated content was verified.

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

@book{Binney:2008,
  	url = {http://adsabs.harvard.edu/abs/2008gady.book.....B},
  	Author = {{Binney}, J. and {Tremaine}, S.},
  	Booktitle = {Galactic Dynamics: Second Edition, by James Binney and Scott Tremaine.~ISBN 978-0-691-13026-2 (HB).~Published by Princeton University Press, Princeton, NJ USA, 2008.},
  	Publisher = {Princeton University Press},
  	Title = {{Galactic Dynamics: Second Edition}},
  	Year = 2008
}

@article{gaia,
    author = {{Gaia Collaboration}},
    title = "{The Gaia mission}",
    journal = {Astronomy and Astrophysics},
    archivePrefix = "arXiv",
    eprint = {1609.04153},
    primaryClass = "astro-ph.IM",
    keywords = {space vehicles: instruments, Galaxy: structure, astrometry, parallaxes, proper motions, telescopes},
    year = 2016,
    month = nov,
    volume = 595,
    doi = {10.1051/0004-6361/201629272},
    url = {http://adsabs.harvard.edu/abs/2016A%26A...595A...1G},
}

@article{astropy,
    author = {{Astropy Collaboration}},
    title = "{Astropy: A community Python package for astronomy}",
    journal = {Astronomy and Astrophysics},
    archivePrefix = "arXiv",
    eprint = {1307.6212},
    primaryClass = "astro-ph.IM",
    keywords = {methods: data analysis, methods: miscellaneous, virtual observatory tools},
    year = 2013,
    month = oct,
    volume = 558,
    doi = {10.1051/0004-6361/201322068},
    url = {http://adsabs.harvard.edu/abs/2013A%26A...558A..33A}
}

@article{Hunt:2025,
  author = {{Hunt}, Jason A.~S. and {Vasiliev}, Eugene},
  title = {Milky Way dynamics in light of Gaia},
  journal = {New Astronomy Reviews},
  year = 2025,
  volume = 98,
  doi = {10.1016/j.newar.2024.101721},
  url = {https://www.sciencedirect.com/science/article/pii/S1387647324000289}
}

@misc{fidgit,
  author = {A. M. Smith and K. Thaney and M. Hahnel},
  title = {Fidgit: An ungodly union of GitHub and Figshare},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/arfon/fidgit}
}


