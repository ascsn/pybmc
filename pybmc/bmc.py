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
        """
        if training_options is None:
            training_options = {}

        iterations = training_options.get('iterations', 50000)
        num_components = self.U_hat.shape[1]
        S_hat = self.S_hat

        b_mean_prior = training_options.get('b_mean_prior', np.zeros(num_components))
        b_mean_cov = training_options.get('b_mean_cov', np.diag(S_hat**2))
        nu0_chosen = training_options.get('nu0_chosen', 1.0)
        sigma20_chosen = training_options.get('sigma20_chosen', 0.02)

        self.samples = gibbs_sampler(self.centered_experiment_train, self.U_hat, iterations, [b_mean_prior, b_mean_cov, nu0_chosen, sigma20_chosen])



    def predict(self, property):
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

        rndm_m, (lower, median, upper) = rndm_m_random_calculator(model_preds, self.samples, self.Vt_hat)

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






