# Usage Guide

This guide provides a comprehensive walkthrough of the `pybmc` package, demonstrating how to load data, combine models, and generate predictions with uncertainty quantification. We will use the `selected_data.h5` file included in the repository for this example.

## 1. Load and Prepare Data

First, we import the necessary classes and specify the path to our data file. We then load the data, specifying the models and properties we're interested in.

```python
import pandas as pd
from pybmc.data import Dataset
from pybmc.bmc import BayesianModelCombination

# Path to the data file
data_path = "pybmc/selected_data.h5"

# Initialize the dataset
dataset = Dataset(data_path)

# Load data for specified models and properties
data_dict = dataset.load_data(
    models=["FRDM12", "HFB24", "D1M", "UNEDF1", "BCPM", "AME2020"],
    keys=["BE"],
    domain_keys=["N", "Z"],
    truth_column_name="AME2020"  # Specify which model is the truth data
)
```

!!! note "Truth Data with Smaller Domain"
    The `truth_column_name` parameter allows the truth/experimental data to have a smaller domain than the prediction models. When specified:
    
    - Prediction models are inner-joined to find their common domain
    - Truth data is left-joined, allowing it to have fewer points
    - Domain points without truth data will have NaN values in the truth column
    
    This enables training on available experimental data while making predictions across the full model domain.

### Alternative: Traditional Loading (All Models Share Domain)

If you want all models to share the same domain, simply omit the `truth_column_name` parameter:

```python
# All models must have data at the same domain points
data_dict = dataset.load_data(
    models=["FRDM12", "HFB24", "D1M", "UNEDF1", "BCPM"],
    keys=["BE"],
    domain_keys=["N", "Z"]
)
```

## 2. Split the Data

Next, we split the data into training, validation, and test sets. `pybmc` supports random splitting as shown below.

!!! tip "Training with Smaller Truth Domain"
    When using `truth_column_name`, only rows where truth data is available (non-NaN) should be used for training. You can filter the data like this:
    
    ```python
    # Filter to only include rows where truth data is available
    df_with_truth = data_dict["BE"][data_dict["BE"]["AME2020"].notna()]
    
    # Split only the data with truth values
    train_df, val_df, test_df = dataset.split_data(
        {"BE": df_with_truth},
        "BE",
        splitting_algorithm="random",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )
    ```

For cases where all models share the same domain:

```python
# Split the data into training, validation, and test sets
train_df, val_df, test_df = dataset.split_data(
    data_dict,
    "BE",
    splitting_algorithm="random",
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
)
```

## 3. Initialize and Train the BMC Model

Now, we initialize the `BayesianModelCombination` class. We provide the list of models (excluding the truth column), the data dictionary, and the name of the column containing the ground truth values.

```python
# Initialize the Bayesian Model Combination
# Note: models_list should only include prediction models, not the truth data
bmc = BayesianModelCombination(
    models_list=["FRDM12", "HFB24", "D1M", "UNEDF1", "BCPM"],
    data_dict=data_dict,
    truth_column_name="AME2020",
)
```

Before training, we orthogonalize the model predictions. This is a crucial step that improves the stability and performance of the Bayesian inference.

```python
# Orthogonalize the model predictions
bmc.orthogonalize("BE", train_df, components_kept=3)
```

With the data prepared and the model orthogonalized, we can train the model combination. We use Gibbs sampling to infer the posterior distribution of the model weights.

```python
# Train the model
bmc.train(training_options={"iterations": 50000, "sampler": "gibbs_sampling"})
```

### Simplex Constraint Mode

By default, `pybmc` uses an unconstrained Gibbs sampler where model weights can take
any real value.  If you want to enforce that the weights lie on the **probability
simplex** — meaning each weight is between 0 and 1 and the weights sum to 1 — you can
enable the simplex constraint mode.

!!! tip "When to Use Simplex Constraints"
    Use simplex constraints when you want the model combination to behave as a
    **proper weighted average** of the constituent models.  This is appropriate when:

    - You want each model to contribute non-negatively to the prediction.
    - The combined prediction should remain within the range spanned by the individual models.
    - Physical interpretability of the weights matters for your application.

    The unconstrained mode is more flexible and may yield better predictive performance
    when some models systematically over- or under-predict, since negative weights can
    partially cancel out biased models.

There are two ways to enable simplex constraints:

**Option 1: Set at initialization (recommended when you always want simplex)**

```python
bmc = BayesianModelCombination(
    models_list=["FRDM12", "HFB24", "D1M", "UNEDF1", "BCPM"],
    data_dict=data_dict,
    truth_column_name="AME2020",
    constraint="simplex",   # <-- weights constrained to [0, 1], sum to 1
)

bmc.orthogonalize("BE", train_df, components_kept=3)
bmc.train(training_options={
    "iterations": 50000,
    "burn": 10000,        # burn-in iterations for the Metropolis step
    "stepsize": 0.001,    # proposal step size
})
```

**Option 2: Override per training call**

```python
# Initialize with default unconstrained mode
bmc = BayesianModelCombination(
    models_list=["FRDM12", "HFB24", "D1M", "UNEDF1", "BCPM"],
    data_dict=data_dict,
    truth_column_name="AME2020",
)

bmc.orthogonalize("BE", train_df, components_kept=3)

# Override to simplex for this specific training run
bmc.train(training_options={
    "iterations": 50000,
    "sampler": "simplex",
    "burn": 10000,
    "stepsize": 0.001,
})
```

### Inspecting Model Weights

After training, you can inspect the inferred model weights using `get_weights()`:

```python
# Get a summary (mean, std, median per model)
summary = bmc.get_weights()
for model, mean_w, std_w in zip(summary["models"], summary["mean"], summary["std"]):
    print(f"  {model}: {mean_w:.4f} ± {std_w:.4f}")

# Get the full weight matrix (n_samples × n_models) for custom analysis
weight_matrix = bmc.get_weights(summary=False)
```

In simplex mode, every row of the weight matrix is guaranteed to satisfy
\(w_k \ge 0\) and \(\sum_k w_k = 1\).

## 4. Make Predictions

After training, we can use the `predict` method to generate predictions with uncertainty quantification. The method returns the full posterior draws, as well as DataFrames for the lower, median, and upper credible intervals.

!!! note "Predictions Across Full Domain"
    When truth data has a smaller domain, predictions can still be made for all domain points (including those without truth data). This allows you to:
    
    - Train on available experimental data
    - Make predictions beyond the experimental coverage
    - Quantify uncertainty for all predictions

```python
# Make predictions with uncertainty quantification
# Predictions are made for ALL domain points, including those without truth data
rndm_m, lower_df, median_df, upper_df = bmc.predict("BE")

# Display the first 5 rows of the median predictions
print(median_df.head())
```

## 5. Evaluate the Model

Finally, we can evaluate the performance of our model combination using the `evaluate` method. This calculates the coverage of the credible intervals, which tells us how often the true values fall within the predicted intervals.

!!! note "Evaluation on Training Data"
    The `evaluate` method only evaluates on data points where truth values are available. Points with NaN truth values are automatically excluded from the evaluation.

```python
# Evaluate the model's coverage
coverage_results = bmc.evaluate()

# Print the coverage for a 95% credible interval
print(f"Coverage for 95% credible interval: {coverage_results[19]:.2f}%")
```

## Complete Example: Truth Data with Smaller Domain

Here's a complete example demonstrating the workflow when truth/experimental data is only available for a subset of domain points:

```python
import pandas as pd
from pybmc.data import Dataset
from pybmc.bmc import BayesianModelCombination

# Initialize dataset
dataset = Dataset(data_path="pybmc/selected_data.h5")

# Load data with truth_column_name parameter
# This allows AME2020 (truth) to have fewer domain points than the models
data_dict = dataset.load_data(
    models=["FRDM12", "HFB24", "D1M", "UNEDF1", "BCPM", "AME2020"],
    keys=["BE"],
    domain_keys=["N", "Z"],
    truth_column_name="AME2020"  # Identifies the truth data
)

# Check the data structure
df = data_dict["BE"]
print(f"Total domain points: {len(df)}")
print(f"Points with truth data: {df['AME2020'].notna().sum()}")
print(f"Points without truth data: {df['AME2020'].isna().sum()}")

# Filter to only rows with truth data for training
df_with_truth = df[df["AME2020"].notna()].copy()

# Split the data (only using points with truth)
train_df, val_df, test_df = dataset.split_data(
    {"BE": df_with_truth},
    "BE",
    splitting_algorithm="random",
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
)

# Initialize BMC (models_list excludes the truth column)
bmc = BayesianModelCombination(
    models_list=["FRDM12", "HFB24", "D1M", "UNEDF1", "BCPM"],
    data_dict=data_dict,
    truth_column_name="AME2020",
)

# Orthogonalize and train on the subset with truth data
bmc.orthogonalize("BE", train_df, components_kept=3)
bmc.train(training_options={"iterations": 50000})

# Make predictions for ALL domain points
# This includes points where AME2020 (truth) is NaN
rndm_m, lower_df, median_df, upper_df = bmc.predict("BE")

print(f"Predictions made for {len(median_df)} domain points")
print("This includes both points with and without experimental truth data!")

# Evaluate coverage (only on points with truth data)
coverage_results = bmc.evaluate()
print(f"Coverage for 95% credible interval: {coverage_results[19]:.2f}%")
```


