import numpy as np


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
    #  How often the model’s credible intervals actually contain the true value
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


def rndm_m_random_calculator(filtered_model_predictions, samples, Vt_hat, output_weights=True):
    """
    Generates posterior predictive samples and credible intervals.

    Args:
        filtered_model_predictions (numpy.ndarray): Model predictions.
        samples (numpy.ndarray): Gibbs samples `[beta, sigma]`.
        Vt_hat (numpy.ndarray): Normalized right singular vectors.

    Returns:
        tuple[numpy.ndarray, list[numpy.ndarray]]:
            - `rndm_m` (numpy.ndarray): Posterior predictive samples.
            - `[lower, median, upper]` (list[numpy.ndarray]): Credible interval arrays.
    """
    np.random.seed(142858)
    rng = np.random.default_rng()

    theta_rand_selected = rng.choice(samples, 10000, replace=False)

    # Extract betas and noise std deviations
    betas = theta_rand_selected[:, :-1]  # shape: (10000, num_models - 1)
    noise_stds = theta_rand_selected[:, -1]  # shape: (10000,)

    # Compute model weights: shape (10000, num_models)
    default_weights = np.full(Vt_hat.shape[1], 1 / Vt_hat.shape[1])
    model_weights_random = (
        betas @ Vt_hat + default_weights
    )  # broadcasting default_weights

    # Generate noiseless predictions: shape (10000, num_data_points)
    yvals_rand_radius = (
        model_weights_random @ filtered_model_predictions.T
    )  # dot product

    # Add Gaussian noise with std = noise_stds (assume diagonal covariance)
    # We'll use broadcasting: noise_stds[:, None] * standard normal noise
    noise = rng.standard_normal(yvals_rand_radius.shape) * noise_stds[:, None]
    rndm_m = yvals_rand_radius + noise

    # Compute credible intervals
    lower_radius = np.percentile(rndm_m, 2.5, axis=0)
    median_radius = np.percentile(rndm_m, 50, axis=0)
    upper_radius = np.percentile(rndm_m, 97.5, axis=0)
    if output_weights:
        return rndm_m, [lower_radius, median_radius, upper_radius], model_weights_random
    else:
        return rndm_m, [lower_radius, median_radius, upper_radius]
    
