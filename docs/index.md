# pyBMC Documentation

Welcome to the official documentation for pyBMC, a Python package for general Bayesian Model Combination (BMC).

## Overview

pyBMC provides a comprehensive framework for combining multiple predictive models using Bayesian statistics. Key features include:

- **Data Management**: Load and preprocess various types of data from HDF5 and CSV files
- **Flexible Truth Data**: Truth/experimental data can have a smaller domain than prediction models
- **Orthogonalization**: Transform model predictions using Singular Value Decomposition (SVD)
- **Bayesian Inference**: Perform Gibbs sampling for model combination
- **Uncertainty Quantification**: Generate predictions with credible intervals
- **Model Evaluation**: Calculate coverage statistics for model validation

## Getting Started

### Installation

```bash
pip install pybmc
```

### Quick Start

For a detailed walkthrough, please see the [Usage Guide](usage.md).

## Documentation Contents

- [Usage Guide](usage.md): Detailed examples and tutorials
- [API Reference](api_reference.md): Complete documentation of all classes and functions
- [Theory Background](theory.md): Mathematical foundations of Bayesian model combination
- [Contributing](contributing.md): How to contribute to pyBMC

## Support

For questions or support, please open an issue on our [GitHub repository](https://github.com/ascsn/pybmc/issues).

## License

This project is licensed under the GPL-3.0 License - see the [License](license.md) file for details.
