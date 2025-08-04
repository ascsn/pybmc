import numpy as np
import matplotlib.pyplot as plt
import corner

def coverage(percentiles, rndm_m, models_output, truth_column):
    """
    Calculate the coverage of credible intervals for the model's predictions.

    :param percentiles: List of percentiles to evaluate the credible intervals.
    :param rndm_m: Array of random samples of model predictions.
    :param models_output: DataFrame containing the true values under the "truth" column.
    :param truth_column: Name of the column containing truth values.
    :return: List of coverage percentages for each percentile.
    """
    #  How often the model’s credible intervals actually contain the true value
    data_total = len(rndm_m.T)   # Number of data points
    M_evals = len(rndm_m)        # Number of samples
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


def rndm_m_random_calculator(filtered_model_predictions, samples, Vt_hat):
        """
        Efficient calculation of random samples of model predictions and their credible intervals.
        Assumes Gaussian noise with diagonal covariance.

        :param filtered_model_predictions: Matrix of filtered model predictions. BC
        :param samples: Array of sampled parameter sets from the Gibbs sampler.
        :param Vt_hat: Normalized right singular vectors from SVD.
        :return: A tuple containing:
                - rndm_m: Array of random samples of model predictions.
                - [lower_radius, median_radius, upper_radius]: Percentile-based credible intervals.
        """
        np.random.seed(142858)
        rng = np.random.default_rng()
       
        theta_rand_selected = rng.choice(samples, 10000, replace=False)

        # Extract betas and noise std deviations
        betas = theta_rand_selected[:, :-1]  # shape: (10000, principle_components_kept)
        noise_stds = theta_rand_selected[:, -1]  # shape: (10000,)

        # Compute model weights: shape (10000, num_models)
        default_weights = np.full(Vt_hat.shape[1], 1 / Vt_hat.shape[1])
        model_weights_random = betas @ Vt_hat + default_weights  # broadcasting default_weights

        # Generate noiseless predictions: shape (10000, num_data_points)
        yvals_rand_radius = model_weights_random @ filtered_model_predictions.T  # dot product

        # Add Gaussian noise with std = noise_stds (assume diagonal covariance)
        # We'll use broadcasting: noise_stds[:, None] * standard normal noise
        noise = rng.standard_normal(yvals_rand_radius.shape) * noise_stds[:, None]
        rndm_m = yvals_rand_radius + noise

        # Compute credible intervals
        lower_radius = np.percentile(rndm_m, 2.5, axis=0)
        median_radius = np.percentile(rndm_m, 50, axis=0)
        upper_radius = np.percentile(rndm_m, 97.5, axis=0)

        return rndm_m, [lower_radius, median_radius, upper_radius]
