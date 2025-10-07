import numpy as np


def gibbs_sampler(y, X, iterations, prior_info):
    """
    Performs Gibbs sampling for Bayesian linear regression.

    Args:
        y (numpy.ndarray): Response vector (centered).
        X (numpy.ndarray): Design matrix.
        iterations (int): Number of sampling iterations.
        prior_info (tuple[numpy.ndarray, numpy.ndarray, float, float]): Prior parameters:
            - `b_mean_prior` (numpy.ndarray): Prior mean for coefficients.
            - `b_mean_cov` (numpy.ndarray): Prior covariance matrix.
            - `nu0` (float): Prior degrees of freedom for variance.
            - `sigma20` (float): Prior variance.

    Returns:
        numpy.ndarray: Posterior samples `[beta, sigma]`.
    """
    b_mean_prior, b_mean_cov, nu0, sigma20 = prior_info
    b_mean_cov_inv = np.linalg.inv(b_mean_cov)
    n = len(y)

    X_T_X = X.T.dot(X)
    X_T_X_inv = np.linalg.inv(X_T_X)

    b_data = X_T_X_inv.dot(X.T).dot(y)
    supermodel = X.dot(b_data)
    residuals = y - supermodel
    sigma2 = np.sum(residuals**2) / len(residuals)
    cov_matrix = sigma2 * X_T_X_inv

    samples = []

    # Initialize sigma2 with a small positive value to avoid division by zero
    sigma2 = max(sigma2, 1e-6)

    for i in range(iterations):
        # Regularize the covariance matrix to ensure it is positive definite
        cov_matrix = np.linalg.inv(X_T_X / sigma2 + b_mean_cov_inv + np.eye(X_T_X.shape[0]) * 1e-6)
        mean_vector = cov_matrix.dot(
            b_mean_cov_inv.dot(b_mean_prior) + X.T.dot(y) / sigma2
        )
        b_current = np.random.multivariate_normal(mean_vector, cov_matrix)

        # Sample from the conditional posterior of sigma2 given bs and data
        supermodel = X.dot(b_current)
        residuals = y - supermodel
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 + np.sum(residuals**2)) / 2.0
        sigma2 = max(1 / np.random.default_rng().gamma(shape_post, 1 / scale_post), 1e-6)

        samples.append(np.append(b_current, np.sqrt(sigma2)))

    return np.array(samples)


def gibbs_sampler_simplex(
    y, X, Vt_hat, S_hat, iterations, prior_info, burn=10000, stepsize=0.001
):
    """
    Performs Gibbs sampling with simplex constraints on model weights.

    Args:
        y (numpy.ndarray): Centered response vector.
        X (numpy.ndarray): Design matrix of principal components.
        Vt_hat (numpy.ndarray): Normalized right singular vectors.
        S_hat (numpy.ndarray): Singular values.
        iterations (int): Number of sampling iterations.
        prior_info (list[float]): `[nu0, sigma20]` - prior parameters for variance.
        burn (int, optional): Burn-in iterations (default: 10000).
        stepsize (float, optional): Proposal step size (default: 0.001).

    Returns:
        numpy.ndarray: Posterior samples `[beta, sigma]`.
    """
    bias0 = np.full(len(Vt_hat.T), 1 / len(Vt_hat.T))
    nu0, sigma20 = prior_info
    cov_matrix_step = np.diag(S_hat**2 * stepsize**2)
    n = len(y)
    b_current = np.full(len(X.T), 0)
    supermodel_current = X.dot(b_current)
    residuals_current = y - supermodel_current
    log_likelihood_current = -np.sum(residuals_current**2)
    sigma2 = -log_likelihood_current / len(residuals_current)
    samples = []
    acceptance = 0

    # Validate inputs
    if burn < 0:
        raise ValueError("Burn-in iterations must be non-negative.")
    if stepsize <= 0:
        raise ValueError("Stepsize must be positive.")

    # Burn-in phase
    for i in range(burn):
        b_proposed = np.random.multivariate_normal(b_current, cov_matrix_step)
        omegas_proposed = np.dot(b_proposed, Vt_hat) + bias0

        # Skip proposals with negative weights
        if not np.any(omegas_proposed < 0):
            supermodel_proposed = X.dot(b_proposed)
            residuals_proposed = y - supermodel_proposed
            log_likelihood_proposed = -np.sum(residuals_proposed**2)
            acceptance_prob = min(
                1,
                np.exp((log_likelihood_proposed - log_likelihood_current) / sigma2),
            )
            if np.random.uniform() < acceptance_prob:
                b_current = np.copy(b_proposed)
                log_likelihood_current = log_likelihood_proposed

        # Sample variance
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 - log_likelihood_current) / 2.0
        sigma2 = 1 / np.random.default_rng().gamma(shape_post, 1 / scale_post)

    # Sampling phase
    for i in range(iterations):
        b_proposed = np.random.multivariate_normal(b_current, cov_matrix_step)
        omegas_proposed = np.dot(b_proposed, Vt_hat) + bias0

        if not np.any(omegas_proposed < 0):
            supermodel_proposed = X.dot(b_proposed)
            residuals_proposed = y - supermodel_proposed
            log_likelihood_proposed = -np.sum(residuals_proposed**2)
            acceptance_prob = min(
                1,
                np.exp((log_likelihood_proposed - log_likelihood_current) / sigma2),
            )
            if np.random.uniform() < acceptance_prob:
                b_current = np.copy(b_proposed)
                log_likelihood_current = log_likelihood_proposed
                acceptance += 1

        # Sample variance
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 - log_likelihood_current) / 2.0
        sigma2 = 1 / np.random.default_rng().gamma(shape_post, 1 / scale_post)
        samples.append(np.append(b_current, np.sqrt(sigma2)))

    return np.array(samples)


def USVt_hat_extraction(U, S, Vt, components_kept):
    """
    Extracts reduced-dimensionality matrices from SVD results.

    Args:
        U (numpy.ndarray): Left singular vectors.
        S (numpy.ndarray): Singular values.
        Vt (numpy.ndarray): Right singular vectors (transposed).
        components_kept (int): Number of components to retain.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            - `U_hat` (numpy.ndarray): Reduced left singular vectors.
            - `S_hat` (numpy.ndarray): Retained singular values.
            - `Vt_hat` (numpy.ndarray): Normalized right singular vectors.
            - `Vt_hat_normalized` (numpy.ndarray): Original right singular vectors.
    """
    U_hat = np.array([U.T[i] for i in range(components_kept)]).T
    S_hat = S[:components_kept]
    Vt_hat = np.array([Vt[i] / S[i] for i in range(components_kept)])
    Vt_hat_normalized = np.array([Vt[i] for i in range(components_kept)])
    return U_hat, S_hat, Vt_hat, Vt_hat_normalized
