import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class BayesianModelCombination:
    """
    The main idea of this class is to perform BMM on the set of models that we choose 
    from the dataset class. What should this class contain:
    + Orthogonalization step.
    + Perform Bayesian inference on the training data that we extract from the Dataset class.
    + Predictions for certain isotopes.
    """
    colors_sets = [
    "#ff7f0e",

    "#1f77b4",

    "#2ca02c",
    "#d62728",
    
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    '#1f77b4',  # muted blue
    'r',  
    ] 

    def __init__(self, models_truth, selected_models_dataset, weights = None):
        """ 
        This step should already specify which of the models that we are going to use for training, with the assumption that 
        our dataset have the experimental data for sampling
        """

        if not isinstance(models_truth, list) or not all(isinstance(model, str) for model in models_truth):
            raise ValueError("The 'models' should be a list of model names (strings) for Bayesian Combination.")
        
        if not isinstance(selected_models_dataset, pd.DataFrame):
            raise ValueError("The 'selected_models_dataset' should be a pandas dataframe")
        
        if not set(models_truth).issubset(selected_models_dataset.columns):
            raise KeyError("One or more selected models are missing in the dataset.")
        
        if 'truth' not in models_truth:
            raise KeyError("We need a 'truth' data column for the training algorithm")

        
        self.selected_models_dataset = selected_models_dataset # Dataset used for Bayesian Model Mixing

        self.models_truth = models_truth # Models and truth values of the BMC dataset

        self.models = [model for model in models_truth if model != 'truth'] # This is just the set of models without experimental data

        self.weights = weights if weights is not None else None # Weights of the models 

    def train(self, iterations, b_mean_prior, b_mean_cov, nu0_chosen, sigma20_chosen):
      # # These will be required parameters that we have to specify for our Gibbs sampling algorithm
        # required_params = ['b_mean_prior', 'b_mean_cov', 'nu0_chosen', 'sigma20_chosen']
        # if not all(param in kwargs for param in required_params):
        #     raise ValueError(f"Missing parameters for the gibbs sampling algorithm: {required_params}")
        
        # # These parameters will be used for gibbs_sampling algorithm
        # b_mean_prior = kwargs['b_mean_prior']
        # b_mean_cov = kwargs['b_mean_cov']
        # nu0_chosen = kwargs['nu0_chosen']
        # sigma20_chosen = kwargs['sigma20_chosen']

        # The Gibbs sampling algorithm should take in inputs that is defined from the orthogonalize method  
        self.samples = self.gibbs_sampler(self.centered_experiment_train, self.U_hat, iterations, [b_mean_prior, b_mean_cov, nu0_chosen, sigma20_chosen])

    
    def predict(self, N_range, Z_range):
        # The input of this method should just be the isotopes that you want to predict and the output should be the mean predictions

        # This code will help us to calculate the weight of the individual models from the weight of the principal components
        model_weights = []
        for beta in self.samples:
            model_weights.append(np.dot(beta[:-1], self.Vt_hat) + np.full(len(self.Vt_hat[0]) , 1/len(self.Vt_hat[0])))
        model_weights = np.array(model_weights)
        # The weights of corresponding models are now updated
        self.weights = model_weights 

        filtered_model_predictions = self.filtered_models_output_extraction(Z_range, N_range)

        lower, median, upper = self.rndm_m_random_calculator(filtered_model_predictions, self.samples)

        return median
    
    def predict_faster(self, N_range, Z_range):
        # The input of this method should just be the isotopes that you want to predict and the output should be the mean predictions

        # This code will help us to calculate the weight of the individual models from the weight of the principal components
        model_weights = []
        for beta in self.samples:
            model_weights.append(np.dot(beta[:-1], self.Vt_hat) + np.full(len(self.Vt_hat[0]) , 1/len(self.Vt_hat[0])))
        model_weights = np.array(model_weights)
        # The weights of corresponding models are now updated
        self.weights = model_weights 

        filtered_model_predictions = self.filtered_models_output_extraction(Z_range, N_range)

        lower, median, upper = self.rndm_m_random_calculator_faster(filtered_model_predictions, self.samples)

        return median

    def rndm_m_random_calculator(self, filtered_model_predictions, samples):
        np.random.seed(142857)
        rng = np.random.default_rng()
        
        theta_rand_selected = rng.choice(samples, 10000, replace = False)

        model_weights_random = []
        for beta in theta_rand_selected:
            model_weights_random.append(np.dot(beta[:-1], self.Vt_hat) + np.full(len(self.Vt_hat[0]) , 1/len(self.Vt_hat[0])))
        model_weights_random = np.array(model_weights_random)

        rndm_m = []
        for i in range(len(model_weights_random)):
            yvals_rand_radius= filtered_model_predictions.dot(model_weights_random[i].T)
            # else:
            #     yvals_rand= X_full_filtered.T.dot(theta_rand[i][0:-1])
        
            rndm_m.append(yvals_rand_radius +
                        np.random.multivariate_normal(np.full(
                            len(yvals_rand_radius)
                            ,0), np.diag(1.0 * np.full(len(yvals_rand_radius),1.0 * theta_rand_selected[i][-1]**2 ) )))  
            
        lower_radius = np.percentile(rndm_m, 2.5, axis = 0)
        median_radius = np.percentile(rndm_m, 50, axis = 0)
        upper_radius = np.percentile(rndm_m, 97.5, axis = 0)

        return [lower_radius, median_radius, upper_radius]
    
    def rndm_m_random_calculator_faster(self, filtered_model_predictions, samples):
        """
        Efficient calculation of random samples of model predictions and their credible intervals.
        Assumes Gaussian noise with diagonal covariance.
        """
        np.random.seed(142858)
        rng = np.random.default_rng()
        
        theta_rand_selected = rng.choice(samples, 10000, replace=False)

        # Extract betas and noise std deviations
        betas = theta_rand_selected[:, :-1]  # shape: (10000, num_models - 1)
        noise_stds = theta_rand_selected[:, -1]  # shape: (10000,)

        # Compute model weights: shape (10000, num_models)
        default_weights = np.full(self.Vt_hat.shape[1], 1 / self.Vt_hat.shape[1])
        model_weights_random = betas @ self.Vt_hat + default_weights  # broadcasting default_weights

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

        return [lower_radius, median_radius, upper_radius]
    
    def filtered_models_output_extraction(self, Z_range, N_range):
        models_output = self.selected_models_dataset

        # Extract models output dataframe for a specific isotope and separate them into training, validation, and testing regions
        filtered_models_output = models_output[(models_output['Z'] >= Z_range[0]) & (models_output['Z'] <= Z_range[1]) & 
                            (models_output['N'] >= N_range[0]) & (models_output['N'] <= N_range[1])]  
        
        # filtered_models_output_train = models_output.iloc[train_coordinates][(models_output.iloc[train_coordinates]['Z'] >= Z_range[0]) & (models_output.iloc[train_coordinates]['Z'] <= Z_range[1]) & 
        #                     (models_output.iloc[train_coordinates]['N'] >= N_range[0]) & (models_output.iloc[train_coordinates]['N'] <= N_range[1])]
        
        # filtered_models_output_validation = models_output.iloc[validation_coordinates][(models_output.iloc[validation_coordinates]['Z'] >= Z_range[0]) & (models_output.iloc[validation_coordinates]['Z'] <= Z_range[1]) & 
        #                     (models_output.iloc[validation_coordinates]['N'] >= N_range[0]) & (models_output.iloc[validation_coordinates]['N'] <= N_range[1])]
        
        # filtered_models_output_test = models_output.iloc[test_coordinates][(models_output.iloc[test_coordinates]['Z'] >= Z_range[0]) & (models_output.iloc[test_coordinates]['Z'] <= Z_range[1]) & 
        #                     (models_output.iloc[test_coordinates]['N'] >= N_range[0]) & (models_output.iloc[test_coordinates]['N'] <= N_range[1])]

        filtered_model_predictions = filtered_models_output[self.models].values

        return filtered_model_predictions




    def gibbs_sampler(self, y, X, iterations, prior_info):
        #Make sure that "y" has the correct structure. If data is being centered, it should have the mean already substracted
        b_mean_prior, b_mean_cov, nu0, sigma20  =  prior_info
        #From A_First_Course_in_Bayesian_Statistical_Methods (page ~ 159), 
        #nu0 represent the effective prior samples and sigma2_0 represents the expected prior variance
        
        b_mean_cov_inv=np.linalg.inv(b_mean_cov)
        n = len(y) # We are still taking mass and radius to be the same size
        
        X_T_X=X.T.dot(X)
        X_T_X_inv = np.linalg.inv(X_T_X)

        b_data = X_T_X_inv.dot(X.T).dot(y)

        
        
        supermodel=X.dot(b_data)
        

        residuals = y - supermodel 

        
        
        sigma2 = np.sum(residuals**2) / len(residuals) 
        cov_matrix = sigma2 * X_T_X_inv
        
        samples = []
        
        for i in range(iterations):
            # Sample from the conditional posterior of bs given sigma2 and data

            
            cov_matrix = np.linalg.inv(X_T_X/sigma2
                                    + b_mean_cov_inv)
            
            mean_vector = cov_matrix.dot(        b_mean_cov_inv.dot(b_mean_prior)+ X.T.dot(y)/sigma2  )
            
            
            b_current = np.random.multivariate_normal(mean_vector, cov_matrix)

            
            
            
            # Sample from the conditional posterior of sigma2 given bs and data
            supermodel=X.dot(b_current)
            
            residuals = y - supermodel 
            
            
            shape_post = (nu0 + n)/2.
            scale_post = (nu0*sigma20 + np.sum(residuals**2))/2.0
            # sigma2 = 1 / np.random.gamma(shape_post, 1/scale_post)
            sigma2 = 1 / np.random.default_rng().gamma(shape_post, 1/scale_post)
            
            
            samples.append(np.append(b_current,np.sqrt(sigma2)))
        


        return np.array(samples)


    def orthogonalize(self, train_index, components_kept):
        """
        For now this code requires that we have to orthogonalize our data to train since I have not thought of
        a training algorithm without orthogonalization.
        """

        # This gives you the dataframe containing only the training index
        models_truth_train  = self.selected_models_dataset.iloc[train_index]
        # This gives you the dataframe containing only the models that we choose to orthogonalize.
        # Note that self.models are the models we set to use in the investigation
        models_output_train = models_truth_train[self.models]
        # This give you the matrix that you want to perform SVD on
        model_predictions_train = models_output_train.values
        # This is the mean predictions of all models for training region
        predictions_mean_train = np.mean(model_predictions_train, axis = 1)

        # This give you the centered data that we are going to use for train algorithm
        centered_experiment_train = models_truth_train['truth'].values - predictions_mean_train

        # Centered data for orthogonalization
        model_predictions_train_centered = model_predictions_train - predictions_mean_train[:,None]

        U, S, Vt = np.linalg.svd(model_predictions_train_centered)
        # This extracts dimensionality-reduction matrices from SVD
        U_hat, S_hat, Vt_hat, Vt_hat_normalized = self.USVt_hat_extraction(U, S, Vt, components_kept)

        # Defining attribute that contain the centered experimental data later used for Gibss sampling
        self.centered_experiment_train = centered_experiment_train
        # Defining attribute that contain the singular value decomposition matrices 
        self.U_hat = U_hat
        self.Vt_hat = Vt_hat
        self.S_hat = S_hat
        self.Vt_hat_normalized = Vt_hat_normalized
        # Defining attribute that contain the mean predicitons of models from training data
        self._predictions_mean_train = predictions_mean_train
    
    def USVt_hat_extraction(self, U,S,Vt, components_kept):
        U_hat = np.array([U.T[i] for i in range(components_kept)]).T
        S_hat = S[:components_kept]
        Vt_hat = np.array([Vt[i]/S[i] for i in range(components_kept)])
        Vt_hat_normalized = np.array([Vt[i] for i in range(components_kept)])
        return U_hat, S_hat, Vt_hat, Vt_hat_normalized
    
    def plot_filtered_supermodel(self, supermodel_predictions_range, filtered_models_output_list, element):

        color_train=BayesianModelCombination.colors_sets[9]
        color_validation='orange'
        color_test=BayesianModelCombination.colors_sets[3]

        marker_train='s'
        marker_validation='*'
        marker_test='o'

        alpha_train=0.8
        alpha_validation=0.9
        alpha_test=0.4

        plt.rc("xtick", labelsize=30)
        plt.rc("ytick", labelsize=30)

        plt.rcParams['text.usetex'] = False
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        
        lower, median, upper = supermodel_predictions_range

        filtered_models_output= filtered_models_output_list[0]
        filtered_models_output_train= filtered_models_output_list[1]
        filtered_models_output_validation= filtered_models_output_list[2]
        filtered_models_output_test = filtered_models_output_list[3]
        # filtered_models_output_stable = filtered_models_output_list[4]
        
        fig, ax = plt.subplots(figsize=(10,8), dpi=150)
        
        plt.plot(filtered_models_output["N"], median, color="darkkhaki",linewidth=3)
        
        plt.plot(filtered_models_output["N"], lower, color="darkkhaki",linestyle="dashed",linewidth=2,alpha=0.5)
        plt.plot(filtered_models_output["N"], upper, color="darkkhaki",linestyle="dashed",linewidth=2,alpha=0.5)
        
        plt.fill_between(filtered_models_output["N"], lower, upper, color="darkkhaki",alpha=0.3)
        
        color_local= "khaki"
        
        
        # plt.plot(filtered_models_output["N"], -median_simplex_local/filtered_models_output['A'], color='purple', label='$f^\dagger$(Simplex Local)',linewidth=3)
        
        # # plt.plot(filtered_models_output["N"], lower, color=color_simplex,linestyle="dashed",linewidth=3,alpha=0.8)
        # # plt.plot(filtered_models_output["N"], upper, color=color_simplex,linestyle="dashed",linewidth=3,alpha=0.8)
        
        
        # plt.plot(filtered_models_output["N"], -lower_simplex_local/filtered_models_output['A'], color="purple",linestyle="dashed",linewidth=2,alpha=0.8)
        # plt.plot(filtered_models_output["N"], -upper_simplex_local/filtered_models_output['A'], color="purple",linestyle="dashed",linewidth=2,alpha=0.8)
        
        # plt.fill_between(filtered_models_output["N"], -lower_simplex_local/filtered_models_output['A'], -upper_simplex_local/filtered_models_output['A'], color='purple',alpha=0.3)
        
        ax.scatter(x = filtered_models_output_train["N"], 
           y = filtered_models_output_train['truth'], 
           label = r"$\mathcal{X}_0^{tr}$",  # <-- raw string here
           alpha = alpha_train,
           color=color_train,
           s=100,
           marker=marker_train,
           zorder=2)

        ax.scatter(x = filtered_models_output_validation["N"], 
           y = filtered_models_output_validation['truth'], 
           label = r"$\mathcal{X}_0^{va}$",  # <-- raw string
           alpha = alpha_validation,
           color=color_validation,
           s=100,
           marker=marker_validation,
           zorder=2)

        ax.scatter(x = filtered_models_output_test["N"], 
           y = filtered_models_output_test['truth'], 
           label = r"$\mathcal{X}_0^{te}$",  # <-- raw string
           alpha = alpha_test,
           color=color_test,
           s=100,
           marker=marker_test,
           zorder=2)
        
        # ax.scatter(x = filtered_models_output_stable["N"], y = filtered_models_output_stable['truth'], label = "Stable", alpha = 0.9,color='k',s=80,marker="s",zorder=2)
        
        
        
        
        
        plt.xlabel("Neutrons",fontsize=35)
        plt.ylabel(f"(Z= {element}) ", fontsize=33)
        # plt.ylabel(Selected_element_name+ " BE/A MeV",fontsize=25)
        
        plt.legend(fontsize=20,markerscale=1,ncol=2,columnspacing=0.5)

        # plt.title(f'Radius Calibration with {title} component(s)', fontsize = 25)
        # plt.savefig(f'{save_fig}')
        # plt.show()


        # def orthogonalize(self, selected_models_dataset = None):
        #     selected_models_dataset = selected_models_dataset if selected_models_dataset is not None else self.selected_models_dataset