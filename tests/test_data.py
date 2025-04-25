import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pybmc_trials.dataset import Dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'N': [10, 20, 30, 10, 40],
            'Z': [10, 20, 30, 10, 50],
            'value': [100, 200, 300, 110, 400],
            'model': ['A', 'A', 'A', 'B', 'B']
        })
        self.dataset = Dataset()


    @patch("pybmc_trials.dataset.os.path.exists", return_value=True)
    @patch("pybmc_trials.dataset.pd.read_csv")
    def test_load_data_csv(self, mock_read_csv, mock_exists):
        mock_read_csv.return_value = self.sample_data

        self.dataset.data_source = "dummy.csv"
        result = self.dataset.load_data(
            models=["A", "B"],
            keys=["N", "Z", "value"],
            domain_keys=["N", "Z"]
        )

        # Only the domain (10, 10) should be retained after syncing
        expected_domain = pd.DataFrame({'N': [10], 'Z': [10]})

        # Check domain for each model
        for model in ["A", "B"]:
            model_df = result[result["model"] == model]
            domain = model_df[["N", "Z"]].drop_duplicates().reset_index(drop=True)
            pd.testing.assert_frame_equal(
                domain.sort_values(by=["N", "Z"]).reset_index(drop=True),
                expected_domain.sort_values(by=["N", "Z"]).reset_index(drop=True)
            )

        # Check that all expected columns are present
        for col in ["N", "Z", "value", "model"]:
            self.assertIn(col, result.columns)


    def test_split_data_random(self):
        df = self.sample_data[["N", "Z"]]  # use two columns for coordinate format
        train, val, test = self.dataset.split_data(data=df, splitting_algorithm="random", train_size=0.6, val_size=0.2, test_size=0.2)

        total_len = len(train) + len(val) + len(test)
        self.assertEqual(total_len, len(df))  # No rows should be lost
        self.assertAlmostEqual(len(train) / len(df), 0.6, delta=0.25)
        self.assertAlmostEqual(len(val) / len(df), 0.2, delta=0.25)
        self.assertAlmostEqual(len(test) / len(df), 0.2, delta=0.25)

    def test_split_data_inside_to_outside(self):
        df = self.sample_data[["N", "Z"]].drop_duplicates().reset_index(drop=True)

        # Define (N, Z) = (10, 10) as the "stable" center
        stable_points = [(10, 10)]
        distance1 = 0  # Only exact match goes to train
        distance2 = 100  # Everyone else goes to validation or test

        train, val, test = self.dataset.split_data(
            data=df,
            splitting_algorithm="inside_to_outside",
            stable_points=stable_points,
            distance1=0.1,
            distance2=100
        )

        total_len = len(train) + len(val) + len(test)
        self.assertEqual(total_len, len(df))  # No rows should be lost

        # Check that the one matching stable point went to train
        self.assertTrue(any((row["N"] == 10 and row["Z"] == 10) for _, row in train.iterrows()))


    def test_get_subset_filters_correctly(self):
    # Create a mock dataset with two models
        self.dataset.data = {
            "A": pd.DataFrame({
                "N": [10, 20, 30],
                "Z": [10, 20, 30],
                "value": [100, 200, 300]
            }),
            "B": pd.DataFrame({
                "N": [10, 15, 25],
                "Z": [10, 15, 25],
                "value": [110, 150, 250]
            })
        }

        # Define filter: select rows where N > 10
        filters = {"N": lambda x: x > 10}

        # Apply get_subset to both models
        result = self.dataset.get_subset(filters=filters, models_to_filter=["A", "B"])

        # Confirm result is a single DataFrame with correct models and filtered values
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("model", result.columns)

        # Check that only N > 10
        self.assertTrue((result["N"] > 10).all())

        # Confirm both models are represented
        models_in_result = result["model"].unique().tolist()
        self.assertIn("A", models_in_result)
        self.assertIn("B", models_in_result)



if __name__ == '__main__':
    unittest.main()

