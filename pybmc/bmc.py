import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from .inference_utils import (
    gibbs_sampler_simplex,
    gibbs_sampler_heteroscedastic,
    USVt_hat_extraction,
)
from .sampling_utils import (
    coverage,
    rndm_m_heteroscedastic_calculator,
    DEFAULT_PREDICTIVE_SEED,
)
from .error_models import (
    VARIANCE_MODELS,
    DEFAULT_SAMPLER_SETTINGS,
    HeteroscedasticMetrics,
    required_metrics,
    variance_basis,
)


class BayesianModelCombination:
    """
    The main idea of this class is to perform Bayesian Model Combination (BMC) on the set of models that we choose
    from the dataset class. What should this class contain:
    + Orthogonalization step.
    + Perform Bayesian inference on the training data that we extract from the Dataset class.
    + Predictions for certain isotopes.
    """

    VALID_CONSTRAINTS = ("unconstrained", "simplex")
    VALID_ERROR_MODELS = tuple(VARIANCE_MODELS)

    def __init__(self, models_list, data_dict, truth_column_name, weights=None, constraint="unconstrained", error_model="homoscedastic"):
        """
        Initialize the BayesianModelCombination class.

        :param models_list: List of model names
        :param data_dict: Dictionary from `load_data()` where each key is a model name and each value is a DataFrame of properties
        :param truth_column_name: Name of the column containing the truth values.
        :param weights: Optional initial weights for the models.
        :param constraint: Weight constraint mode. Options:
            - ``"unconstrained"`` (default): No constraints on model weights.
            - ``"simplex"``: Forces weights to lie on the probability simplex
              (each weight between 0 and 1, weights sum to 1). Uses a
              Metropolis-within-Gibbs sampler to enforce the constraint.
        :param error_model: Noise structure of the combination. Options
            (see :mod:`pybmc.error_models` for the variance forms):
            - ``"homoscedastic"`` (default): A single constant variance.
            - ``"hetero_pc_dist"`` / ``"hetero_pc_dist_quad"``: Variance
              linear/quadratic in the distance from the training centroid
              in principal-component space.
            - ``"hetero_model_var"`` / ``"hetero_model_var_quad"``: Variance
              linear/quadratic in the spread among model predictions.
            - ``"hetero_combined_linear"`` / ``"hetero_combined_quadratic"``:
              Variance depending on both metrics.
            Heteroscedastic error models currently require the
            ``"unconstrained"`` weight mode.
        """

        if not isinstance(models_list, list) or not all(isinstance(model, str) for model in models_list):
            raise ValueError("The 'models' should be a list of model names (strings) for Bayesian Combination.")
        if not isinstance(data_dict, dict) or not all(isinstance(df, pd.DataFrame) for df in data_dict.values()):
            raise ValueError("The 'data_dict' should be a dictionary of pandas DataFrames, one per property.")
        if constraint not in self.VALID_CONSTRAINTS:
            raise ValueError(
                f"Invalid constraint '{constraint}'. "
                f"Must be one of {self.VALID_CONSTRAINTS}."
            )
        if error_model not in self.VALID_ERROR_MODELS:
            raise ValueError(
                f"Invalid error model '{error_model}'. "
                f"Must be one of {self.VALID_ERROR_MODELS}."
            )
        if error_model != "homoscedastic" and constraint == "simplex":
            raise ValueError(
                "Heteroscedastic error models are not supported with the "
                "'simplex' constraint; use constraint='unconstrained'."
            )

        self.data_dict = data_dict
        self.models_list = models_list
        self.models = [m for m in models_list if m != 'truth']
        self.weights = weights if weights is not None else None
        self.truth_column_name = truth_column_name
        self.constraint = constraint
        self.error_model = error_model
        self.samples = None
        self.Vt_hat = None
        self.mh_acceptance_rate_ = None
        self._trained_error_model = None
        self._metrics_calculator = None


    def orthogonalize(self, property, train_df, components_kept):
        """
        Perform orthogonalization for the specified property using training data.

        :param property: The nuclear property to orthogonalize on (e.g., 'BE').
        :param train_index: Training data from split_data
        :param components_kept: Number of SVD components to retain.
        """
        # Store selected property
        self.current_property = property

        # Extract the relevant DataFrame for that property
        df = self.data_dict[property].copy()
        self.selected_models_dataset = df  # Store for train() and predict()

        # Extract model outputs (only the model columns)
        models_output_train = train_df[self.models]
        model_predictions_train = models_output_train.values

        # Mean prediction across models (per nucleus)
        predictions_mean_train = np.mean(model_predictions_train, axis=1)

        # Experimental truth values for the property
        centered_experiment_train = train_df[self.truth_column_name].values - predictions_mean_train

        # Center model predictions
        model_predictions_train_centered = model_predictions_train - predictions_mean_train[:, None]

        # Perform SVD
        U, S, Vt = np.linalg.svd(model_predictions_train_centered)

        # Dimensionality reduction
        U_hat, S_hat, Vt_hat, Vt_hat_normalized = USVt_hat_extraction(U, S, Vt, components_kept) #type: ignore

        # Save for training
        self.centered_experiment_train = centered_experiment_train
        self.U_hat = U_hat
        self.Vt_hat = Vt_hat
        self.S_hat = S_hat
        self.Vt_hat_normalized = Vt_hat_normalized
        self._predictions_mean_train = predictions_mean_train
        # Raw training predictions, needed to fit the heteroscedasticity
        # metrics (PC-space centroid and normalization bounds).
        self._train_model_predictions = model_predictions_train


    def train(self, training_options=None):
        """
        Train the model combination using training data and optional training parameters.

        All error models (the homoscedastic one included) share the
        likelihood ``y_i ~ N(X_i . b, sigma_i^2)`` and are trained with
        the same Gibbs-within-Metropolis sampler; the homoscedastic
        model is simply the case where the variance basis is the
        constant column alone, so ``sigma_i^2 = sigma^2``.

        :param training_options: Dictionary of training options. Keys:
            - 'iterations': (int) Number of retained Gibbs samples (default 50000)
            - 'sampler': (str) Override the constraint mode for this training run.
              ``"unconstrained"`` or ``"simplex"``. If not provided, uses the
              instance-level ``self.constraint`` set at initialization.
            - 'error_model': (str) Override the error model for this training
              run (see ``VALID_ERROR_MODELS``). If not provided, uses the
              instance-level ``self.error_model`` set at initialization.
            - 'seed': (int) Seed for the sampler. If not provided, the
              sampler draws from the shared package-wide generator
              (see :mod:`pybmc.rng`), which is seeded at import.
            - 'b_mean_prior': (np.ndarray) Prior mean vector (default zeros)
              *(unconstrained sampler)*
            - 'b_mean_cov': (np.ndarray) Prior covariance matrix (default diag(S_hat²))
              *(unconstrained sampler)*
            - 'nu0_chosen': (float) Degrees of freedom for variance prior (default 1.0)
              *(simplex sampler only)*
            - 'sigma20_chosen': (float) Prior variance (default 0.02)
              *(simplex sampler only)*
            - 'burn': (int) Burn-in iterations (default 10000 for simplex,
              5000 otherwise)
            - 'stepsize': (float) Proposal step size (default 0.001)
              *(simplex sampler only)*
            - 'proposal_scales': (list) Diagonal of the Metropolis-Hastings
              proposal covariance for the variance parameters; sensible
              per-model defaults are used if omitted
              *(unconstrained sampler)*
            - 'init_params': (list) Initial values for the non-constant
              variance parameters *(unconstrained sampler)*
            - 'prior_spec': (list of (shape, scale)) Gamma priors for the
              variance parameters *(unconstrained sampler)*
            - 'adapt_proposal': (bool) Rescale the proposal during burn-in
              toward 'target_acceptance' (default True)
              *(unconstrained sampler)*
            - 'target_acceptance': (float) Acceptance rate targeted by the
              burn-in adaptation (default 0.25)
              *(unconstrained sampler)*

        After training with the unconstrained sampler, the
        Metropolis-Hastings acceptance rate of the variance parameters
        is available in ``self.mh_acceptance_rate_``.
        """
        if training_options is None:
            training_options = {}

        # Determine which sampler to use: training_options overrides instance default
        sampler_mode = training_options.get('sampler', self.constraint)
        if sampler_mode not in self.VALID_CONSTRAINTS:
            raise ValueError(
                f"Invalid sampler '{sampler_mode}'. "
                f"Must be one of {self.VALID_CONSTRAINTS}."
            )

        # Same override pattern for the error model.
        error_model_mode = training_options.get('error_model', self.error_model)
        if error_model_mode not in self.VALID_ERROR_MODELS:
            raise ValueError(
                f"Invalid error model '{error_model_mode}'. "
                f"Must be one of {self.VALID_ERROR_MODELS}."
            )
        if error_model_mode != "homoscedastic" and sampler_mode == "simplex":
            raise ValueError(
                "Heteroscedastic error models are not supported with the "
                "'simplex' sampler; use the unconstrained sampler."
            )

        iterations = training_options.get('iterations', 50000)
        num_components = self.U_hat.shape[1]
        S_hat = self.S_hat
        seed = training_options.get('seed')

        if sampler_mode == "simplex":
            nu0_chosen = training_options.get('nu0_chosen', 1.0)
            sigma20_chosen = training_options.get('sigma20_chosen', 0.02)
            burn = training_options.get('burn', 10000)
            stepsize = training_options.get('stepsize', 0.001)
            self._metrics_calculator = None
            self.mh_acceptance_rate_ = None
            self.samples = gibbs_sampler_simplex(
                self.centered_experiment_train,
                self.U_hat,
                self.Vt_hat,
                self.S_hat,
                iterations,
                [nu0_chosen, sigma20_chosen],
                burn=burn,
                stepsize=stepsize,
                seed=seed,
            )
        else:
            settings = DEFAULT_SAMPLER_SETTINGS[error_model_mode]
            terms = VARIANCE_MODELS[error_model_mode]
            metric_names = required_metrics(error_model_mode)

            if metric_names:
                # Fit the metrics (PC centroid, normalization bounds) on the
                # training predictions, then build the variance design matrix.
                self._metrics_calculator = HeteroscedasticMetrics(
                    metric_names
                ).fit(self._train_model_predictions, self.Vt_hat)
                metrics_train = self._metrics_calculator.compute(
                    self._train_model_predictions
                )
                basis_train = variance_basis(metrics_train, terms)
            else:
                # Homoscedastic: the variance basis is the constant column
                # alone, so theta = [sigma^2].
                self._metrics_calculator = None
                basis_train = np.ones(
                    (len(self.centered_experiment_train), 1)
                )

            b_mean_prior = training_options.get('b_mean_prior', np.zeros(num_components))
            b_mean_cov = training_options.get('b_mean_cov', np.diag(S_hat**2))
            self.samples, self.mh_acceptance_rate_ = gibbs_sampler_heteroscedastic(
                self.centered_experiment_train,
                self.U_hat,
                basis_train,
                iterations,
                burn=training_options.get('burn', 5000),
                proposal_scales=training_options.get(
                    'proposal_scales', settings['proposal_scales']
                ),
                init_params=training_options.get(
                    'init_params', settings['init_params']
                ),
                prior_spec=training_options.get(
                    'prior_spec', settings['prior_spec']
                ),
                b_mean_prior=b_mean_prior,
                b_mean_cov=b_mean_cov,
                adapt_proposal=training_options.get('adapt_proposal', True),
                target_acceptance=training_options.get('target_acceptance', 0.25),
                seed=seed,
            )

        # Remember which error model produced self.samples so that
        # predict()/evaluate() use the matching predictive distribution.
        self._trained_error_model = error_model_mode



    def predict(self, property, seed=DEFAULT_PREDICTIVE_SEED):
        """
        Predict a specified property using the model weights learned during training.

        :param property: The property name to predict (e.g., 'ChRad').
        :param seed: Seed for the posterior predictive draws (subsampling
            of the posterior samples and the noise added on top).
            Defaults to a fixed constant so repeated calls are
            reproducible; pass a different value for independent draws.
        :return:
            - rndm_m: array of shape (n_samples, n_points), full posterior draws
            - lower_df: DataFrame with columns domain_keys + ['Predicted_Lower']
            - median_df: DataFrame with columns domain_keys + ['Predicted_Median']
            - upper_df: DataFrame with columns domain_keys + ['Predicted_Upper']
        """
        if self.samples is None or self.Vt_hat is None:
            raise ValueError("Must call `orthogonalize()` and `train()` before predicting.")
        
        if property not in self.data_dict:
            raise KeyError(f"Property '{property}' not found in data_dict.")
        
        df = self.data_dict[property].copy()

        # Infer domain and model columns
        full_model_cols = self.models
        domain_keys = [col for col in df.columns if col not in full_model_cols and col != self.truth_column_name]

        # Determine which models are present
        available_models = [m for m in full_model_cols if m in df.columns]
        
        if len(available_models) == 0:
            raise ValueError("No available trained models are present in prediction DataFrame.")

        # Filter predictions and model weights
        model_preds = df[available_models].values
        domain_df = df[domain_keys].reset_index(drop=True)

        rndm_m, (lower, median, upper) = self._posterior_predictive(
            model_preds, seed=seed
        )

        # Build output DataFrames
        lower_df = domain_df.copy()
        
        lower_df["Predicted_Lower"] = lower

        median_df = domain_df.copy()
        median_df["Predicted_Median"] = median

        upper_df = domain_df.copy()
        upper_df["Predicted_Upper"] = upper

        return rndm_m, lower_df, median_df, upper_df

    def evaluate(self, domain_filter=None, seed=DEFAULT_PREDICTIVE_SEED):
        """
        Evaluate the model combination using coverage calculation.

        :param domain_filter: dict with optional domain key ranges, e.g., {"Z": (20, 30), "N": (20, 40)}
        :param seed: Seed for the posterior predictive draws underlying
            the coverage calculation. Defaults to a fixed constant so
            repeated calls are reproducible.
        :return: coverage list for each percentile
        """
        df = self.data_dict[self.current_property]

        if domain_filter:
            # Inline optimized filtering
            for col, cond in domain_filter.items():
                if col == 'multi' and callable(cond):
                    df = df[df.apply(cond, axis=1)]
                elif callable(cond):
                    df = df[cond(df[col])]
                elif isinstance(cond, tuple) and len(cond) == 2:
                    df = df[df[col].between(*cond)]
                elif isinstance(cond, list):
                    df = df[df[col].isin(cond)]
                else:
                    df = df[df[col] == cond]

        # Coverage is only defined where truth values exist.
        df = df.dropna(subset=[self.truth_column_name])

        preds = df[self.models].to_numpy()
        rndm_m, (lower, median, upper) = self._posterior_predictive(preds, seed=seed)

        return coverage(np.arange(0, 101, 5), rndm_m, df, truth_column=self.truth_column_name)

    def _posterior_predictive(self, model_preds, seed=DEFAULT_PREDICTIVE_SEED):
        """
        Posterior predictive draws for the given model predictions, using
        the predictive distribution matching the trained error model.

        :param model_preds: Array of shape (n_points, n_models) with one
            column per model in ``self.models`` order.
        :param seed: Seed for the posterior predictive draws.
        :return: Tuple ``(rndm_m, (lower, median, upper))``.
        """
        error_model = self._trained_error_model or "homoscedastic"
        terms = VARIANCE_MODELS[error_model]
        if terms:
            metrics = self._metrics_calculator.compute(model_preds)
            basis = variance_basis(metrics, terms)
        else:
            # Homoscedastic: constant-only variance basis.
            basis = np.ones((model_preds.shape[0], 1))
        return rndm_m_heteroscedastic_calculator(
            model_preds, self.samples, self.Vt_hat, basis, seed=seed
        )

    def get_weights(self, summary=True):
        """
        Compute model weights from posterior samples.

        Converts the sampled coefficient vectors (beta) into model weights
        using the transformation ``omega = beta @ Vt_hat + 1/M``, where M is
        the number of models.  In simplex-constrained mode, all weights are
        guaranteed to be non-negative and sum to 1.

        :param summary: If True (default), return a dictionary with
            ``'mean'``, ``'std'``, ``'median'`` arrays keyed by statistic.
            If False, return the full ``(n_samples, n_models)`` weight matrix.
        :return: Weight summary dict or full weight matrix.
        :raises ValueError: If ``train()`` has not been called.
        """
        if self.samples is None or self.Vt_hat is None:
            raise ValueError("Must call `orthogonalize()` and `train()` before getting weights.")

        # The first k columns are the PC coefficients; the remaining
        # columns are variance parameters (a single sigma^2 for the
        # homoscedastic model, several for heteroscedastic models).
        betas = self.samples[:, : self.Vt_hat.shape[0]]
        n_models = self.Vt_hat.shape[1]
        default_weights = np.full(n_models, 1.0 / n_models)
        weight_matrix = betas @ self.Vt_hat + default_weights

        if summary:
            return {
                "mean": np.mean(weight_matrix, axis=0),
                "std": np.std(weight_matrix, axis=0),
                "median": np.median(weight_matrix, axis=0),
                "models": self.models,
            }
        return weight_matrix




