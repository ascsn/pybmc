"""Heteroscedastic error models for Bayesian model combination.

All error models share the mean structure of the orthogonalized BMC
regression and differ only in how the noise variance depends on the
data point:

=============================  ====================================================
name                           sigma_i^2
=============================  ====================================================
``homoscedastic``              sigma^2 (constant)
``hetero_pc_dist``             alpha + beta * d_i
``hetero_model_var``           alpha + beta * v_i
``hetero_pc_dist_quad``        alpha + beta_1 d_i + beta_2 d_i^2
``hetero_model_var_quad``      alpha + beta_1 v_i + beta_2 v_i^2
``hetero_combined_linear``     alpha + beta_d d_i + beta_m v_i
``hetero_combined_quadratic``  alpha + beta_d1 d_i + beta_d2 d_i^2
                               + beta_m1 v_i + beta_m2 v_i^2
=============================  ====================================================

with two physics-informed, per-point metrics:

- ``pc_dist`` (``d``): Euclidean distance from the training-data centroid
  in principal-component space — grows away from the fitted region.
- ``model_var`` (``v``): variance among the individual model predictions —
  grows where the models disagree.

Both metrics are min-max normalized using the *training* points only, so
values on extrapolated points can exceed 1.

Every model — the homoscedastic one included — is parametrized on the
variance (sigma^2) scale and shares the single likelihood
``y_i ~ N(X_i . b, sigma_i^2)`` sampled by
:func:`pybmc.inference_utils.gibbs_sampler_heteroscedastic`; the
homoscedastic model is simply the case where the variance basis is the
constant column alone.
"""

import numpy as np

#: Floor applied to a variance only where positivity is *not* guaranteed
#: by construction: the sampler's initial constant term (estimated from
#: data residuals, which can be exactly zero for a perfect fit) and
#: prediction-time variances (the normalized metrics of extrapolated
#: points can be negative, so ``phi . theta`` can dip below zero there).
#: During sampling no floor is needed: the variance parameters are kept
#: strictly positive by the Metropolis-Hastings step and the training
#: variance basis is non-negative with a leading column of ones, so
#: every per-point variance is positive by construction.
VARIANCE_FLOOR = 1e-9

#: Maps each error-model name to its variance-basis terms beyond the
#: constant, as ``(metric_name, power)`` tuples. ``homoscedastic`` has no
#: terms: its variance basis is the constant column alone, and it runs
#: through the same likelihood and sampler as every other error model.
VARIANCE_MODELS = {
    "homoscedastic": [],
    "hetero_pc_dist": [("pc_dist", 1)],
    "hetero_model_var": [("model_var", 1)],
    "hetero_pc_dist_quad": [("pc_dist", 1), ("pc_dist", 2)],
    "hetero_model_var_quad": [("model_var", 1), ("model_var", 2)],
    "hetero_combined_linear": [("pc_dist", 1), ("model_var", 1)],
    "hetero_combined_quadratic": [
        ("pc_dist", 1),
        ("pc_dist", 2),
        ("model_var", 1),
        ("model_var", 2),
    ],
}

#: Metropolis-Hastings tuning defaults per error model: random-walk
#: proposal scales (variances of the diagonal Gaussian proposal), initial
#: values for the non-constant variance parameters, and Gamma prior
#: ``(shape, scale)`` pairs for every variance parameter (constant term
#: first).
DEFAULT_SAMPLER_SETTINGS = {
    "homoscedastic": {
        "proposal_scales": [0.05],
        "init_params": [],
        "prior_spec": [(2, 10)],
    },
    "hetero_pc_dist": {
        "proposal_scales": [0.05, 0.005],
        "init_params": [0.01],
        "prior_spec": [(2, 10), (1, 1)],
    },
    "hetero_model_var": {
        "proposal_scales": [0.05, 0.005],
        "init_params": [0.01],
        "prior_spec": [(2, 10), (1, 1)],
    },
    "hetero_pc_dist_quad": {
        "proposal_scales": [5e-2, 5e-3, 5e-3],
        "init_params": [0.01, 0.001],
        "prior_spec": [(2, 10), (2, 10), (2, 10)],
    },
    "hetero_model_var_quad": {
        "proposal_scales": [5e-2, 5e-3, 5e-3],
        "init_params": [0.01, 0.001],
        "prior_spec": [(2, 10), (2, 10), (2, 10)],
    },
    "hetero_combined_linear": {
        "proposal_scales": [1e-2, 1e-3, 1e-3],
        "init_params": [0.01, 0.01],
        "prior_spec": [(2, 10), (2, 10), (2, 10)],
    },
    "hetero_combined_quadratic": {
        "proposal_scales": [1e-2, 1e-3, 1e-3, 1e-3, 1e-3],
        "init_params": [0.01, 0.001, 0.01, 0.001],
        "prior_spec": [(2, 10)] * 5,
    },
}


def required_metrics(error_model):
    """
    Returns the metric names an error model needs.

    Args:
        error_model (str): One of the keys of `VARIANCE_MODELS`.

    Returns:
        list[str]: Unique metric names (empty for ``homoscedastic``).
    """
    if error_model not in VARIANCE_MODELS:
        raise ValueError(
            f"Unknown error model '{error_model}'. "
            f"Must be one of {tuple(VARIANCE_MODELS)}."
        )
    return sorted({metric for metric, _ in VARIANCE_MODELS[error_model]})


def variance_parameter_names(error_model):
    """
    Returns human-readable names of the variance parameters of a model.

    Args:
        error_model (str): One of the keys of `VARIANCE_MODELS`.

    Returns:
        list[str]: Names such as ``['alpha', 'beta_pc_dist^1', ...]``;
        ``['sigma^2']`` for the homoscedastic model (whose constant term
        is the variance itself).
    """
    terms = VARIANCE_MODELS[error_model] if error_model in VARIANCE_MODELS else None
    if terms is None:
        raise ValueError(
            f"Unknown error model '{error_model}'. "
            f"Must be one of {tuple(VARIANCE_MODELS)}."
        )
    if not terms:
        return ["sigma^2"]
    return ["alpha"] + [f"beta_{metric}^{power}" for metric, power in terms]


def pc_distance_metric(model_predictions, Vt_hat, pc_centroid):
    """
    Distance of each point from a reference centroid in PC space.

    The principal-component coordinates of a point are obtained by
    projecting its raw model predictions with ``Vt_hat`` (for a
    row-centered SVD this is identical to projecting the centered
    predictions, since the right singular vectors are orthogonal to the
    all-ones direction).

    Args:
        model_predictions (numpy.ndarray): Model outputs, shape
            ``(n_points, n_models)``.
        Vt_hat (numpy.ndarray): Scaled right singular vectors, shape
            ``(components_kept, n_models)``.
        pc_centroid (numpy.ndarray): Training centroid in PC space,
            shape ``(components_kept,)``.

    Returns:
        numpy.ndarray: Euclidean distances, shape ``(n_points,)``.
    """
    pc_coords = model_predictions @ Vt_hat.T
    return np.linalg.norm(pc_coords - pc_centroid, axis=1)


def model_variance_metric(model_predictions):
    """
    Variance among the model predictions for each point (NaN-aware).

    Args:
        model_predictions (numpy.ndarray): Model outputs, shape
            ``(n_points, n_models)``.

    Returns:
        numpy.ndarray: Per-point variance across models, shape ``(n_points,)``.
    """
    return np.nanvar(model_predictions, axis=1)


def variance_basis(metrics, terms):
    """
    Builds the variance design matrix ``phi`` for a heteroscedastic model.

    The per-point variance is ``sigma_i^2 = phi[i] . theta`` where the
    first column of ``phi`` is ones (the constant term ``alpha``).

    Args:
        metrics (dict[str, numpy.ndarray]): Metric arrays keyed by name.
        terms (list[tuple[str, int]]): ``(metric_name, power)`` pairs.

    Returns:
        numpy.ndarray: Basis matrix, shape ``(n_points, 1 + len(terms))``.
    """
    n_points = len(next(iter(metrics.values())))
    columns = [np.ones(n_points)]
    for metric, power in terms:
        columns.append(np.asarray(metrics[metric], dtype=float) ** power)
    return np.column_stack(columns)


class HeteroscedasticMetrics:
    """
    Computes and normalizes the per-point metrics of the error models.

    The object is fit on the training model predictions: it stores the
    training centroid in PC space and the training min/max of every
    metric. Metrics for any other set of points are then computed
    consistently and scaled with the *training* bounds, so extrapolated
    points can legitimately exceed 1.
    """

    def __init__(self, metric_names):
        """
        :param metric_names: Metric names to compute
            (subset of ``{'pc_dist', 'model_var'}``).
        """
        unknown = set(metric_names) - {"pc_dist", "model_var"}
        if unknown:
            raise ValueError(f"Unknown metric names: {sorted(unknown)}")
        self.metric_names = list(metric_names)
        self.Vt_hat = None
        self.pc_centroid = None
        self.bounds = {}

    def fit(self, train_model_predictions, Vt_hat):
        """
        Fit the normalization on the training predictions.

        Args:
            train_model_predictions (numpy.ndarray): Training model
                outputs, shape ``(n_train, n_models)``.
            Vt_hat (numpy.ndarray): Scaled right singular vectors from
                the orthogonalization step.

        Returns:
            HeteroscedasticMetrics: ``self``, for chaining.
        """
        self.Vt_hat = np.asarray(Vt_hat)
        self.pc_centroid = np.mean(
            train_model_predictions @ self.Vt_hat.T, axis=0
        )
        train_metrics = self._raw_metrics(train_model_predictions)
        self.bounds = {}
        for name, values in train_metrics.items():
            lo, hi = float(np.min(values)), float(np.max(values))
            self.bounds[name] = (lo, hi)
        return self

    def compute(self, model_predictions):
        """
        Computes normalized metrics for a set of points.

        Args:
            model_predictions (numpy.ndarray): Model outputs, shape
                ``(n_points, n_models)``.

        Returns:
            dict[str, numpy.ndarray]: Normalized metric arrays keyed by name.
        """
        if self.Vt_hat is None:
            raise ValueError("Call `fit()` before `compute()`.")
        metrics = self._raw_metrics(model_predictions)
        for name, values in metrics.items():
            lo, hi = self.bounds[name]
            if hi > lo:
                metrics[name] = (values - lo) / (hi - lo)
            # A metric that is constant on the training set is left
            # unscaled; the sampler treats it like an extra constant.
        return metrics

    def _raw_metrics(self, model_predictions):
        metrics = {}
        if "pc_dist" in self.metric_names:
            metrics["pc_dist"] = pc_distance_metric(
                model_predictions, self.Vt_hat, self.pc_centroid
            )
        if "model_var" in self.metric_names:
            metrics["model_var"] = model_variance_metric(model_predictions)
        return metrics
