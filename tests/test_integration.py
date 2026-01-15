"""Integration tests for the complete workflow with smaller truth domains."""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from pybmc.data import Dataset
from pybmc.bmc import BayesianModelCombination


class TestIntegrationWithSmallerTruthDomain(unittest.TestCase):
    """Test the full workflow: load data with smaller truth domain -> BMC training -> prediction."""
    
    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_hdf")
    def test_bmc_workflow_with_smaller_truth_domain(self, mock_read_hdf, mock_exists):
        """Test BMC can train and predict when truth data has a smaller domain than models."""
        
        # Setup mock data: models have 6 points, truth has only 4 points
        # Use realistic variation in model predictions
        def mock_hdf_reader(file, key):
            if key == "model1":
                return pd.DataFrame({
                    "N": [1, 2, 3, 4, 5, 6],
                    "Z": [10, 20, 30, 40, 50, 60],
                    "BE": [100, 205, 295, 410, 495, 605]  # Varied predictions
                })
            elif key == "model2":
                return pd.DataFrame({
                    "N": [1, 2, 3, 4, 5, 6],
                    "Z": [10, 20, 30, 40, 50, 60],
                    "BE": [95, 210, 305, 405, 510, 600]  # Different variation
                })
            elif key == "model3":
                return pd.DataFrame({
                    "N": [1, 2, 3, 4, 5, 6],
                    "Z": [10, 20, 30, 40, 50, 60],
                    "BE": [105, 195, 310, 395, 505, 610]  # Another pattern
                })
            elif key == "truth":
                # Truth data has only 4 points
                return pd.DataFrame({
                    "N": [1, 2, 3, 4],
                    "Z": [10, 20, 30, 40],
                    "BE": [102, 203, 301, 404]
                })
        
        mock_read_hdf.side_effect = mock_hdf_reader
        
        # Step 1: Load data with truth_column_name
        dataset = Dataset(data_source="fake_path.h5")
        data_dict = dataset.load_data(
            models=["model1", "model2", "model3", "truth"],
            keys=["BE"],
            domain_keys=["N", "Z"],
            truth_column_name="truth"
        )
        
        # Verify loaded data
        self.assertIn("BE", data_dict)
        df = data_dict["BE"]
        self.assertEqual(len(df), 6, "Should have 6 domain points from models")
        self.assertTrue(all(col in df.columns for col in ["N", "Z", "model1", "model2", "model3", "truth"]))
        
        # Truth should have NaN for last 2 points
        self.assertEqual(df["truth"].isna().sum(), 2)
        
        # Step 2: Create training set (first 4 points where truth is available)
        train_df = df[df["truth"].notna()].copy()
        self.assertEqual(len(train_df), 4, "Training set should have 4 points where truth is available")
        
        # Step 3: Initialize BMC
        models_list = ["model1", "model2", "model3"]
        bmc = BayesianModelCombination(
            models_list=models_list,
            data_dict=data_dict,
            truth_column_name="truth"
        )
        
        # Step 4: Train BMC on the available truth data
        bmc.orthogonalize(property="BE", train_df=train_df, components_kept=2)
        bmc.train()
        
        # Verify training completed
        self.assertIsNotNone(bmc.samples)
        self.assertIsNotNone(bmc.Vt_hat)
        
        # Step 5: Predict on all domain points (including those without truth)
        rndm_m, lower_df, median_df, upper_df, weights= bmc.predict("BE")
        
        # Verify predictions
        self.assertEqual(rndm_m.shape[1], 6, "Should have predictions for all 6 domain points")
        self.assertEqual(len(lower_df), 6, "Lower bounds for all 6 points")
        self.assertEqual(len(median_df), 6, "Median predictions for all 6 points")
        self.assertEqual(len(upper_df), 6, "Upper bounds for all 6 points")
        
        # Check that domain keys are preserved in output
        self.assertTrue(all(col in lower_df.columns for col in ["N", "Z"]))
        self.assertTrue(all(col in median_df.columns for col in ["N", "Z"]))
        self.assertTrue(all(col in upper_df.columns for col in ["N", "Z"]))
        
        # Predictions should exist for all points (no NaN)
        self.assertTrue(lower_df["Predicted_Lower"].notna().all())
        self.assertTrue(median_df["Predicted_Median"].notna().all())
        self.assertTrue(upper_df["Predicted_Upper"].notna().all())
        
        # Step 6: Evaluate on training data (points with truth)
        coverage_results = bmc.evaluate()
        
        # Verify evaluation results
        self.assertIsNotNone(coverage_results)
        self.assertIsInstance(coverage_results, list)
        self.assertEqual(len(coverage_results), 21)  # Coverage at percentiles 0, 5, 10, ..., 100
        self.assertTrue(all(isinstance(c, float) for c in coverage_results))
    
    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_csv")
    def test_bmc_workflow_with_smaller_truth_domain_csv(self, mock_read_csv, mock_exists):
        """Test BMC workflow with CSV format when truth data has smaller domain."""
        
        # Create CSV data with models having 6 points and truth having 4 points
        # Use realistic variation in model predictions
        csv_data = []
        model_predictions = {
            "model1": [100, 205, 295, 410, 495, 605],
            "model2": [95, 210, 305, 405, 510, 600],
            "model3": [105, 195, 310, 395, 505, 610]
        }
        
        for i, (n, z) in enumerate([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]):
            for model, predictions in model_predictions.items():
                csv_data.append({"N": n, "Z": z, "BE": predictions[i], "model": model})
        
        # Truth data only for first 4 points
        for n, z, be in [(1, 10, 102), (2, 20, 203), (3, 30, 301), (4, 40, 404)]:
            csv_data.append({"N": n, "Z": z, "BE": be, "model": "truth"})
        
        mock_read_csv.return_value = pd.DataFrame(csv_data)
        
        # Load data with truth_column_name
        dataset = Dataset(data_source="fake_path.csv")
        data_dict = dataset.load_data(
            models=["model1", "model2", "model3", "truth"],
            keys=["BE"],
            domain_keys=["N", "Z"],
            model_column="model",
            truth_column_name="truth"
        )
        
        # Verify data structure
        df = data_dict["BE"]
        self.assertEqual(len(df), 6)
        self.assertEqual(df["truth"].isna().sum(), 2)
        
        # Train on points with truth
        train_df = df[df["truth"].notna()].copy()
        
        # Initialize and train BMC
        bmc = BayesianModelCombination(
            models_list=["model1", "model2", "model3"],
            data_dict=data_dict,
            truth_column_name="truth"
        )
        bmc.orthogonalize(property="BE", train_df=train_df, components_kept=2)
        bmc.train()
        
        # Predict on all points
        rndm_m, lower_df, median_df, upper_df, weights = bmc.predict("BE")
        
        # Verify predictions cover all domain points
        self.assertEqual(len(lower_df), 6)
        self.assertTrue(lower_df["Predicted_Lower"].notna().all())


if __name__ == "__main__":
    unittest.main()
