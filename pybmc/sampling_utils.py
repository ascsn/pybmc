import numpy as np

from .error_models import VARIANCE_FLOOR
from .rng import DEFAULT_SEED, get_rng

#: Default seed for posterior predictive draws (subsampling of posterior
#: samples and the noise added on top) when the caller does not supply
#: one explicitly. This is the same seed that drives the package-wide
#: generator used by the MCMC samplers (see :mod:`pybmc.rng`), so one
#: constant governs all pybmc randomness. Callers that need independent
#: draws across repeated calls (e.g. outer Monte Carlo loops) can pass
#: their own ``seed``, or ``seed=None`` to draw from the shared
#: package-wide stream instead.
DEFAULT_PREDICTIVE_SEED = DEFAULT_SEED


def coverage(percentiles, rndm_m, models_output, truth_column):
    """
    Calculates coverage percentages for credible intervals.

    Args:
        percentiles (list[int]): Percentiles to evaluate (e.g., `[5, 10, ..., 95]`).
        rndm_m (numpy.ndarray): Posterior samples of predictions.
        models_output (pandas.DataFrame): DataFrame containing true values.
        truth_column (str): Name of column with true values.

    Returns:
        list[float]: Coverage percentages for each percentile.
    """
    #  How often the model's credible intervals actually contain the true value
    data_total = len(rndm_m.T)  # Number of data points
    M_evals = len(rndm_m)  # Number of samples
    data_true = models_output[truth_column].tolist()

    coverage_results = []

    for p in percentiles:
        count_covered = 0
        for i in range(data_total):
            # Sort model evaluations for the i-th data point
            sorted_evals = np.sort(rndm_m.T[i])
            # Find indices for lower and upper bounds of the credible interval
            lower_idx = int((0.5 - p / 200) * M_evals)
            upper_idx = int((0.5 + p / 200) * M_evals) - 1
            # Check if the true value y[i] is within this interval
            if sorted_evals[lower_idx] <= data_true[i] <= sorted_evals[upper_idx]:
                count_covered += 1
        coverage_results.append(count_covered / data_total * 100)

    return coverage_results


def rndm_m_random_calculator(
    filtered_model_predictions, samples, Vt_hat, seed=DEFAULT_PREDICTIVE_SEED
):
    """
    Posterior predictive samples for the homoscedastic model.

    The homoscedastic model is the constant-only special case of the
    heteroscedastic predictive distribution, so this delegates to
    `rndm_m_heteroscedastic_calculator` with a constant-only variance
    basis.

    Args:
        filtered_model_predictions (numpy.ndarray): Model predictions.
        samples (numpy.ndarray): Posterior samples `[beta, sigma^2]`.
        Vt_hat (numpy.ndarray): Normalized right singular vectors.
        seed (int | None, optional): Seed for the posterior predictive
            draws (subsampling of `samples` and the noise added on top).
            Defaults to `DEFAULT_PREDICTIVE_SEED`; pass None to draw
            from the shared package-wide stream instead.

    Returns:
        tuple[numpy.ndarray, list[numpy.ndarray]]:
            - `rndm_m` (numpy.ndarray): Posterior predictive samples.
            - `[lower, median, upper]` (list[numpy.ndarray]): Credible interval arrays.
    """
    constant_basis = np.ones((filtered_model_predictions.shape[0], 1))
    return rndm_m_heteroscedastic_calculator(
        filtered_model_predictions, samples, Vt_hat, constant_basis, seed=seed
    )


def rndm_m_heteroscedastic_calculator(
    filtered_model_predictions, samples, Vt_hat, variance_basis,
    seed=DEFAULT_PREDICTIVE_SEED,
):
    """
    Generates posterior predictive samples for any error model.

    The noise added to each prediction has a per-point variance
    ``sigma_i^2 = phi_i . theta`` where ``theta`` are the variance
    parameters of each posterior draw. The homoscedastic model is the
    special case of a constant-only basis (a single column of ones),
    handled by the `rndm_m_random_calculator` convenience wrapper.

    Args:
        filtered_model_predictions (numpy.ndarray): Model predictions,
            shape ``(n_points, n_models)``.
        samples (numpy.ndarray): Posterior samples
            ``[beta_1..beta_k, theta_1..theta_p]`` from
            `gibbs_sampler_heteroscedastic`.
        Vt_hat (numpy.ndarray): Normalized right singular vectors, shape
            ``(k, n_models)``.
        variance_basis (numpy.ndarray): Variance design matrix for the
            prediction points, shape ``(n_points, p)``.
        seed (int | None, optional): Seed for the posterior predictive
            draws (subsampling of `samples` and the noise added on top).
            Defaults to `DEFAULT_PREDICTIVE_SEED`; pass None to draw
            from the shared package-wide stream instead.

    Returns:
        tuple[numpy.ndarray, list[numpy.ndarray]]:
            - `rndm_m` (numpy.ndarray): Posterior predictive samples.
            - `[lower, median, upper]` (list[numpy.ndarray]): 95% credible
              interval bounds and median.
    """
    rng = get_rng(seed)

    n_draws = min(10000, len(samples))
    replace = len(samples) < 10000
    theta_rand_selected = rng.choice(samples, n_draws, replace=replace)

    n_components = Vt_hat.shape[0]
    betas = theta_rand_selected[:, :n_components]
    variance_params = theta_rand_selected[:, n_components:]

    # Model weights and noiseless central predictions.
    default_weights = np.full(Vt_hat.shape[1], 1 / Vt_hat.shape[1])
    model_weights_random = betas @ Vt_hat + default_weights
    yvals_central = model_weights_random @ filtered_model_predictions.T

    # Per-draw, per-point variances (n_draws, n_points). The floor is
    # needed here (unlike during sampling) because the normalized metrics
    # of extrapolated points can be negative, so phi . theta can dip
    # below zero even though every theta draw is positive.
    sigma2 = variance_params @ variance_basis.T
    sigma2 = np.maximum(sigma2, VARIANCE_FLOOR)

    noise = rng.standard_normal(yvals_central.shape) * np.sqrt(sigma2)
    rndm_m = yvals_central + noise

    lower = np.percentile(rndm_m, 2.5, axis=0)
    median = np.percentile(rndm_m, 50, axis=0)
    upper = np.percentile(rndm_m, 97.5, axis=0)

    return rndm_m, [lower, median, upper]


def coverage_quality(percentiles, coverage_results):
    """
    Scalar calibration score: mean |empirical - nominal| coverage.

    Args:
        percentiles (array-like): Nominal credible-interval widths in
            percent (as passed to `coverage`).
        coverage_results (array-like): Empirical coverage in percent.

    Returns:
        float: Mean absolute deviation in percentage points (lower is
        better; 0 is perfect calibration).
    """
    return float(
        np.mean(np.abs(np.asarray(coverage_results) - np.asarray(percentiles)))
    )


def mace(rndm_m, y_true, quantile_levels=None):
    """
    Mean Absolute Calibration Error (MACE): a quantile-based calibration score.

    Complementary to `coverage`/`coverage_quality`'s two-sided central-interval
    view: for each one-sided quantile level ``q``, compares the empirical
    fraction of true values at or below the predictive ``q``-th percentile
    against ``q`` itself.

    Args:
        rndm_m (numpy.ndarray): Posterior predictive draws, shape
            ``(n_samples, n_points)``.
        y_true (array-like): True/observed values, shape ``(n_points,)``.
        quantile_levels (array-like, optional): Quantile levels in percent,
            0-100 (default: 5, 10, ..., 95).

    Returns:
        float: Mean absolute deviation between empirical and nominal
        quantile coverage, in percentage points (lower is better; 0 is
        perfect calibration).
    """
    y_true = np.asarray(y_true, dtype=float)
    if quantile_levels is None:
        quantile_levels = np.arange(5, 100, 5)
    quantile_levels = np.asarray(quantile_levels, dtype=float)

    predicted_quantiles = np.percentile(rndm_m, quantile_levels, axis=0)
    empirical = np.mean(y_true[None, :] <= predicted_quantiles, axis=1) * 100.0
    return float(np.mean(np.abs(empirical - quantile_levels)))


def reduced_chi_square(rndm_m, y_true):
    """
    Reduced chi-squared statistic of the posterior predictive distribution.

    ``chi^2_red = mean_i[ (y_true_i - mean_i)^2 / var_i ]`` where
    ``mean_i``/``var_i`` are the posterior predictive mean/variance at
    point ``i``. A value near 1 indicates well-calibrated predictive
    uncertainties; > 1 means the intervals are too narrow (overconfident,
    "underdispersed" in the language of `diagnose_coverage_shape`); < 1
    means too wide ("overdispersed").

    Args:
        rndm_m (numpy.ndarray): Posterior predictive draws, shape
            ``(n_samples, n_points)``.
        y_true (array-like): True/observed values, shape ``(n_points,)``.

    Returns:
        float: Reduced chi-squared statistic.

    Raises:
        ValueError: If the posterior predictive variance is non-positive
            at any point (a degenerate/zero-noise predictive distribution).
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_mean = np.mean(rndm_m, axis=0)
    pred_var = np.var(rndm_m, axis=0)
    if np.any(pred_var <= 0):
        raise ValueError(
            "Posterior predictive variance must be positive at every point."
        )
    return float(np.mean((y_true - pred_mean) ** 2 / pred_var))


def diagnose_coverage_shape(
    percentiles, coverage_results, bias_tolerance=5.0, balance_threshold=0.7
):
    """
    Classifies credible intervals as under-/over-dispersed or calibrated.

    Args:
        percentiles (array-like): Nominal interval widths in percent.
        coverage_results (array-like): Empirical coverage in percent.
        bias_tolerance (float, optional): Mean absolute deviation
            (percentage points) below which the model counts as well
            calibrated (default: 5.0).
        balance_threshold (float, optional): Fraction of intervals that
            must lie consistently on one side of nominal to call the
            direction (default: 0.7).

    Returns:
        dict: Keys ``'diagnosis'`` (``'well_calibrated'`` |
        ``'underdispersed'`` | ``'overdispersed'`` | ``'mixed'``),
        ``'mean_bias'``, ``'mean_abs_error'``, ``'frac_below'``,
        ``'frac_above'`` and ``'residuals'``. Underdispersed means the
        intervals are too narrow (overconfident); overdispersed too wide.
    """
    percentiles = np.asarray(percentiles, dtype=float)
    coverage_results = np.asarray(coverage_results, dtype=float)

    residuals = coverage_results - percentiles
    mean_bias = float(np.mean(residuals))
    mean_abs_error = float(np.mean(np.abs(residuals)))
    frac_below = float(np.mean(residuals < 0))
    frac_above = float(np.mean(residuals > 0))

    if mean_abs_error <= bias_tolerance:
        diagnosis = "well_calibrated"
    elif mean_bias < -bias_tolerance and frac_below >= balance_threshold:
        diagnosis = "underdispersed"
    elif mean_bias > bias_tolerance and frac_above >= balance_threshold:
        diagnosis = "overdispersed"
    else:
        diagnosis = "mixed"

    return {
        "diagnosis": diagnosis,
        "mean_bias": mean_bias,
        "mean_abs_error": mean_abs_error,
        "frac_below": frac_below,
        "frac_above": frac_above,
        "residuals": residuals,
    }
