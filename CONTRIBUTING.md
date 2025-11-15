# Contributing to pybmc

We welcome contributions from the community! This document outlines how you can contribute to the pybmc project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/pybmc.git
   cd pybmc
   ```
3. **Set up the development environment**:
   ```bash
   poetry install
   ```

## Development Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and ensure tests pass:
   ```bash
   pytest
   ```
3. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add new feature for orthogonalization"
   ```
4. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request against the main repository

## Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public classes and functions using Google style
- Keep functions small and focused (under 50 lines when possible)
- Write unit tests for new features using pytest

## Documentation

- Update documentation in the `docs/` directory
- Add examples for new features in usage.md
- Update API reference in api_reference.md when adding new public interfaces

## Testing

- Write tests for new features in the `tests/` directory
- Ensure test coverage remains above 90%
- Run tests locally before submitting a PR:
  ```bash
  pytest --cov=pybmc
  ```

## Reporting Issues

When reporting issues, please include:
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc)

## Code Review Process

- All pull requests require at least one maintainer approval
- Maintainers will review for:
  - Code quality and style
  - Test coverage
  - Documentation updates
  - Backward compatibility
- Be prepared to make revisions based on feedback

## License

By contributing to pybmc, you agree that your contributions will be licensed under the MIT License.
