import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from .inference_utils import gibbs_sampler,  USVt_hat_extraction
from .sampling_utils import coverage, rndm_m_random_calculator


class BayesianModelCombination:
    """
    The main idea of this class is to perform BMM on the set of models that we choose 
    from the dataset class. What should this class contain:
    + Orthogonalization step.
    + Perform Bayesian inference on the training data that we extract from the Dataset class.
    + Predictions for certain isotopes.
    """

    def __init__(self, models_list, data_dict, truth_column_name, weights=None):
        """ 
        Initialize the BayesianModelCombination class.

        :param models_list: List of model names
        :param data_dict: Dictionary from `load_data()` where each key is a model name and each value is a DataFrame of properties
        :param truth_column_name: Name of the column containing the truth values.
        :param weights: Optional initial weights for the models.
        """

        if not isinstance(models_list, list) or not all(isinstance(model, str) for model in models_list):
            raise ValueError("The 'models' should be a list of model names (strings) for Bayesian Combination.")    
        if not isinstance(data_dict, dict) or not all(isinstance(df, pd.DataFrame) for df in data_dict.values()):
            raise ValueError("The 'data_dict' should be a dictionary of pandas DataFrames, one per property.")

        self.data_dict = data_dict 
        self.models_list = models_list 
        self.models = [m for m in models_list if m != 'truth']
        self.weights = weights if weights is not None else None 
        self.truth_column_name = truth_column_name


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


    def train(self, training_options=None):
        """
        Train the model combination using training data and optional training parameters.

        :param training_data: Placeholder (not used).
        :param training_options: Dictionary of training options. Keys:
            - 'iterations': (int) Number of Gibbs iterations (default 50000)
            - 'b_mean_prior': (np.ndarray) Prior mean vector (default zeros)
            - 'b_mean_cov': (np.ndarray) Prior covariance matrix (default diag(S_hat²))
            - 'nu0_chosen': (float) Degrees of freedom for variance prior (default 1.0)
            - 'sigma20_chosen': (float) Prior variance (default 0.02)
            - 'sampler': (str) 'Gibbs_sampling' or 'simplex' (default 'Gibbs_sampling')
            - 'burn': (int) Burn-in iterations for simplex sampler (default 10000)
            - 'stepsize': (float) Proposal step size for simplex sampler (default 0.001)
        """

        # functions defined so that whenever a key not specified, we print out the default value for users
        def get_option(key, default):
            if key not in training_options:
                print(f"[INFO] Using default value for '{key}': {default}")
            return training_options.get(key, default)

        iterations = get_option('iterations', 50000)
        sampler = get_option('sampler', 'gibbs_sampling')
        burn = get_option('burn', 10000)
        stepsize = get_option('stepsize', 0.001)

        S_hat = self.S_hat
        num_components = self.U_hat.shape[1]

        b_mean_prior = get_option('b_mean_prior', np.zeros(num_components))
        b_mean_cov = get_option('b_mean_cov', np.diag(S_hat**2))
        nu0_chosen = get_option('nu0_chosen', 1.0)
        sigma20_chosen = get_option('sigma20_chosen', 0.02)

        if sampler == 'simplex':
            self.samples = gibbs_sampler_simplex(
                self.centered_experiment_train,
                self.U_hat,
                self.Vt_hat,
                self.S_hat,
                iterations,
                [nu0_chosen, sigma20_chosen],  # Note: no b_mean_prior/b_mean_cov needed
                burn = burn,
                stepsize = stepsize
            ) 
        else:
            self.samples = gibbs_sampler(
                self.centered_experiment_train,
                self.U_hat,
                iterations,
                [b_mean_prior, b_mean_cov, nu0_chosen, sigma20_chosen]
            )


    
    def predict(self, X):
        """
        Predict with full Bayesian posterior sampling, including model weight and data noise.

        :param property: A pandas DataFrame of the property to predict, containing domain columns and model predictions.
        :return:
            - rndm_m: array of shape (n_samples, n_points), full posterior draws
            - lower_df: DataFrame with columns domain_keys + ['Predicted_Lower']
            - median_df: DataFrame with columns domain_keys + ['Predicted_Median']
            - upper_df: DataFrame with columns domain_keys + ['Predicted_Upper']
        """
        if self.samples is None or self.Vt_hat is None:
            raise ValueError("Must call `orthogonalize()` and `train()` before predicting.")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame containing model predictions and domain info.")

        # Infer model columns vs. domain columns
        model_cols = self.models
        domain_keys = [col for col in X.columns if col not in model_cols]

        model_preds = X[model_cols].values
        rndm_m, (lower, median, upper) = rndm_m_random_calculator(model_preds, self.samples, self.Vt_hat)

        domain_df = X[domain_keys].reset_index(drop=True)

        lower_df = domain_df.copy()
        lower_df["Predicted_Lower"] = lower

        median_df = domain_df.copy()
        median_df["Predicted_Median"] = median

        upper_df = domain_df.copy()
        upper_df["Predicted_Upper"] = upper

        return rndm_m, lower_df, median_df, upper_df


    def predict2(self, property):
        """
        Predict a specified property using the model weights learned during training.

        :param property: The property name to predict (e.g., 'ChRad').
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

        # Find indices of available models in training order
        model_indices = [self.models.index(m) for m in available_models]

        # Reduce Vt_hat and samples to only use available models
        Vt_hat_reduced = self.Vt_hat[:, model_indices]

        rndm_m, (lower, median, upper) = rndm_m_random_calculator(model_preds, self.samples, Vt_hat_reduced)

        # Build output DataFrames
        lower_df = domain_df.copy()
        lower_df["Predicted_Lower"] = lower

        median_df = domain_df.copy()
        median_df["Predicted_Median"] = median

        upper_df = domain_df.copy()
        upper_df["Predicted_Upper"] = upper

        return rndm_m, lower_df, median_df, upper_df

    def evaluate(self, domain_filter=None):
        """
        Evaluate the model combination using coverage calculation.

        :param domain_filter: dict with optional domain key ranges, e.g., {"Z": (20, 30), "N": (20, 40)}
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

        preds = df[self.models].to_numpy()
        rndm_m, (lower, median, upper) = rndm_m_random_calculator(preds, self.samples, self.Vt_hat)

        return coverage(np.arange(0, 101, 5), rndm_m, df, truth_column=self.truth_column_name)


    def predict3(self, property):
        """
        Predict a specified property using the model weights learned during training.
        This version slices both Vt_hat and betas to match the available models.

        :param property: The property name to predict (e.g., 'ChRad').
        :return:
            - rndm_m: array of shape (n_samples, n_points), full posterior draws
            - lower_df: DataFrame with columns domain_keys + ['Predicted_Lower']
            - median_df: DataFrame with columns domain_keys + ['Predicted_Median']
            - upper_df: DataFrame with columns domain_keys + ['Predicted_Upper']
        """
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

        # Find indices of available models in training order
        model_indices = [self.models.index(m) for m in available_models]

        # Reduce Vt_hat to match available models
        Vt_hat_reduced = self.Vt_hat[:, model_indices]

        # Sample theta from training samples
        rng = np.random.default_rng()
        theta_rand_selected = rng.choice(self.samples, 10000, replace=False)

        # Slice betas to match reduced Vt_hat
        betas_full = theta_rand_selected[:, :-1]  # shape: (10000, num_components)
        num_components_used = Vt_hat_reduced.shape[0]
        betas_reduced = betas_full[:, :num_components_used]

        noise_stds = theta_rand_selected[:, -1][:, None]  # shape: (10000, 1)
        sliced_samples = np.hstack([betas_reduced, noise_stds])  # shape: (10000, num_components_used + 1)

        # Now call rndm_m_random_calculator with sliced samples:
        rndm_m, (lower, median, upper) = rndm_m_random_calculator(
            model_preds, sliced_samples, Vt_hat_reduced
        )

        lower_df = domain_df.copy()
        lower_df["Predicted_Lower"] = lower

        median_df = domain_df.copy()
        median_df["Predicted_Median"] = median

        upper_df = domain_df.copy()
        upper_df["Predicted_Upper"] = upper

        return rndm_m, lower_df, median_df, upper_df
