# pyBMC: A General Bayesian Model Combination Package

[![Coverage Status](https://img.shields.io/badge/Coverage-86%25-brightgreen)](https://ascsn.github.io/pybmc/coverage/)

pyBMC is a Python package for performing Bayesian Model Combination (BMC) on various predictive models. It provides tools for data handling, orthogonalization, Gibbs sampling, and prediction with uncertainty quantification. The model combination methodology follows [this paper](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033266) by Giuliani et al.

## Features

- **Data Management**: Load and preprocess nuclear mass data from HDF5 and CSV files
- **Orthogonalization**: Transform model predictions using Singular Value Decomposition (SVD)
- **Bayesian Inference**: Perform Gibbs sampling for model combination
- **Uncertainty Quantification**: Generate predictions with credible intervals
- **Model Evaluation**: Calculate coverage statistics for model validation

## Installation

```bash
pip install pybmc
```

## Quick Start

For a detailed walkthrough of how to use the package, please see the [Usage Guide](docs/usage.md).

## Documentation

Comprehensive documentation is available at [https://ascsn.github.io/pybmc/](https://ascsn.github.io/pybmc/), including:

- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)
- [Theory Background](docs/theory.md)
- [Contribution Guidelines](docs/CONTRIBUTING.md)

## Contributing

We welcome contributions! Please see our [Contribution Guidelines](docs/CONTRIBUTING.md) for details on how to contribute to the project.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use pyBMC in your research, please cite:

```bibtex
@software{pybmc,
  title = {pyBMC: Bayesian Model Combination},
  author = {Kyle Godbey and Troy Dasher and Pablo Giuliani and An Le},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ascsn/pybmc}}
}
```

## Support

For questions or support, please open an issue on our [GitHub repository](https://github.com/ascsn/pybmc/issues).
