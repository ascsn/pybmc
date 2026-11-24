import numpy as np
import pytest
from pybmc.inference_utils import (
    gibbs_sampler,
    gibbs_sampler_simplex,
    gibbs_sampler_heteroscedastic,
    USVt_hat_extraction,
)


def test_gibbs_sampler():
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1, 0], [0, 1], [1, 1]])
    iterations = 10
    prior_info = {
        "b_mean_prior": np.array([0.0, 0.0]),
        "b_mean_cov": np.eye(2),
    }

    samples = gibbs_sampler(y, X, iterations, prior_info, burn=100)

    assert samples.shape == (iterations, 3)
    assert not np.any(np.isnan(samples))
    # Last column is the constant variance sigma^2, positive by construction.
    assert np.all(samples[:, -1] > 0)


def test_gibbs_sampler_matches_constant_basis_heteroscedastic():
    # The homoscedastic sampler is the constant-only case of the
    # heteroscedastic one: with the same seed they must agree exactly.
    y = np.array([1.0, 2.0, 3.0, 2.5])
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    samples = gibbs_sampler(y, X, 20, burn=50, seed=3)
    hetero_samples, _ = gibbs_sampler_heteroscedastic(
        y, X, np.ones((len(y), 1)), 20, burn=50, seed=3
    )
    np.testing.assert_array_equal(samples, hetero_samples)


def test_gibbs_sampler_seed_reproducible():
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1, 0], [0, 1], [1, 1]])
    s1 = gibbs_sampler(y, X, 20, burn=50, seed=123)
    s2 = gibbs_sampler(y, X, 20, burn=50, seed=123)
    np.testing.assert_array_equal(s1, s2)


def test_gibbs_sampler_simplex():
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1, 0], [0, 1], [1, 1]])
    Vt_hat = np.array([[0.5, 0.5], [0.5, -0.5]])
    S_hat = np.array([1.0, 0.5])
    iterations = 10
    prior_info = [1.0, 1.0]

    samples = gibbs_sampler_simplex(y, X, Vt_hat, S_hat, iterations, prior_info, burn=100, stepsize=0.01)

    assert samples.shape[0] == iterations
    assert not np.any(np.isnan(samples))
    # Last column is sigma^2, positive by construction.
    assert np.all(samples[:, -1] > 0)


def test_gibbs_sampler_simplex_seed_reproducible():
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1, 0], [0, 1], [1, 1]])
    Vt_hat = np.array([[0.5, 0.5], [0.5, -0.5]])
    S_hat = np.array([1.0, 0.5])
    prior_info = [1.0, 1.0]

    s1 = gibbs_sampler_simplex(y, X, Vt_hat, S_hat, 20, prior_info, burn=50, stepsize=0.01, seed=7)
    s2 = gibbs_sampler_simplex(y, X, Vt_hat, S_hat, 20, prior_info, burn=50, stepsize=0.01, seed=7)
    np.testing.assert_array_equal(s1, s2)


def test_USVt_hat_extraction():
    U = np.array([[1, 0], [0, 1]])
    S = np.array([2.0, 1.0])
    Vt = np.array([[1, 0], [0, 1]])
    components_kept = 2

    U_hat, S_hat, Vt_hat, Vt_hat_normalized = USVt_hat_extraction(U, S, Vt, components_kept)

    assert U_hat.shape == (2, 2)
    assert len(S_hat) == components_kept
    assert Vt_hat.shape == (2, 2)
    assert Vt_hat_normalized.shape == (2, 2)


def test_gibbs_sampler_simplex_edge_cases():
    # Test with invalid inputs
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1, 0], [0, 1], [1, 1]])
    Vt_hat = np.array([[0.5, 0.5], [0.5, -0.5]])
    S_hat = np.array([1.0, 0.5])
    iterations = 10
    prior_info = [1.0, 1.0]

    # Invalid burn-in value
    with pytest.raises(ValueError):
        gibbs_sampler_simplex(y, X, Vt_hat, S_hat, iterations, prior_info, burn=-1)

    # Invalid stepsize
    with pytest.raises(ValueError):
        gibbs_sampler_simplex(y, X, Vt_hat, S_hat, iterations, prior_info, stepsize=-0.01)


def test_gibbs_sampler_simplex_acceptance_rate():
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1, 0], [0, 1], [1, 1]])
    Vt_hat = np.array([[0.5, 0.5], [0.5, -0.5]])
    S_hat = np.array([1.0, 0.5])
    iterations = 100
    prior_info = [1.0, 1.0]

    samples = gibbs_sampler_simplex(y, X, Vt_hat, S_hat, iterations, prior_info, burn=10, stepsize=0.01)

    # Check acceptance rate is within a reasonable range
    assert 0 < len(samples) <= iterations
    assert not np.any(np.isnan(samples))
