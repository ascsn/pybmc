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
    models=["FRDM12", "HFB24", "D1M",
    "UNEDF1", "BCPM", "AME2020"],
    keys=["BE"],
    domain_keys=["N", "Z"]
```

## 2. Split the Data

Next, we split the data into training, validation, and test sets. `pybmc` supports random splitting as shown below.

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

Now, we initialize the `BayesianModelCombination` class. We provide the list of models, the data dictionary, and the name of the column containing the ground truth values.

```python
# Initialize the Bayesian Model Combination
bmc = BayesianModelCombination(
    models_list=["FRDM12", "HFB24", "D1M",
    "UNEDF1", "BCPM", "AME2020"],
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

## 4. Make Predictions

After training, we can use the `predict` method to generate predictions with uncertainty quantification. The method returns the full posterior draws, as well as DataFrames for the lower, median, and upper credible intervals.

```python
# Make predictions with uncertainty quantification
rndm_m, lower_df, median_df, upper_df = bmc.predict("BE")

# Display the first 5 rows of the median predictions
print(median_df.head())
```

## 5. Evaluate the Model

Finally, we can evaluate the performance of our model combination using the `evaluate` method. This calculates the coverage of the credible intervals, which tells us how often the true values fall within the predicted intervals.

```python
# Evaluate the model's coverage
coverage_results = bmc.evaluate()

# Print the coverage for a 95% credible interval
print(f"Coverage for 95% credible interval: {coverage_results[19]:.2f}%")
```
