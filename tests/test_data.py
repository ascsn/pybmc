import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch
from pybmc.data import Dataset
import logging
from io import StringIO

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [1, 2, 3, 4],
                "target": [10, 20, 30, 40],
                "modelA": [9, 19, 29, 39],
                "modelB": [11, 21, 31, 41],
            }
        )
        self.dataset = Dataset(data_source="fake_path.h5")
        self.dataset.data = {"target": self.sample_df}
        self.dataset.domain_keys = ["x", "y"]

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_csv")
    def test_load_data_csv(self, mock_read_csv, mock_exists):
        mock_read_csv.return_value = pd.DataFrame(
            {
                "x": [1, 2],
                "y": [1, 2],
                "target": [10, 20],
                "model": ["modelA", "modelB"],
            }
        )

        dataset = Dataset(data_source="fake_path.csv")
        result = dataset.load_data(
            models=["modelA", "modelB"],
            keys=["target"],
            domain_keys=["x", "y"],
            model_column="model",
        )

        self.assertIn("target", result)
        self.assertTrue(
            all(
                col in result["target"].columns
                for col in ["x", "y", "modelA", "modelB"]
            )
        )

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_hdf")
    def test_load_data_h5(self, mock_read_hdf, mock_exists):
        model_data = pd.DataFrame(
            {"x": [1, 2], "y": [1, 2], "target": [10, 20]}
        )
        mock_read_hdf.side_effect = lambda file, key: model_data

        dataset = Dataset(data_source="fake_path.h5")
        result = dataset.load_data(
            models=["modelA", "modelB"],
            keys=["target"],
            domain_keys=["x", "y"],
        )

        self.assertIn("target", result)
        self.assertIsInstance(result["target"], pd.DataFrame)
        self.assertTrue(
            all(
                col in result["target"].columns
                for col in ["x", "y", "modelA", "modelB"]
            )
        )

    @patch("pybmc.data.os.path.exists", return_value=True)
    def test_load_data_unsupported_format(self, mock_exists):
        dataset = Dataset(data_source="fake_path.txt")
        with self.assertRaises(ValueError) as context:
            dataset.load_data(models=["modelA"], keys=["target"], domain_keys=["x", "y"])
        self.assertIn("Unsupported file format", str(context.exception))

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_csv")
    def test_load_data_missing_columns_csv(self, mock_read_csv, mock_exists):
        mock_read_csv.return_value = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        dataset = Dataset(data_source="fake_path.csv")
        with self.assertRaises(ValueError) as context:
            dataset.load_data(models=["modelA"], keys=["target"], domain_keys=["x", "y"], model_column="model")
        self.assertIn("Expected column 'model' not found in CSV", str(context.exception))

    def test_split_data_random(self):
        data_dict = {"target": self.sample_df}
        train, val, test = self.dataset.split_data(
            data_dict=data_dict,
            property_name="target",
            splitting_algorithm="random",
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
        )
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, len(self.sample_df))

    def test_split_data_inside_to_outside(self):
        coords_only = self.sample_df[["x", "y"]].copy()
        self.dataset.data = {"target": coords_only}

        train, val, test = self.dataset.split_data(
            data_dict={"target": coords_only},
            property_name="target",
            splitting_algorithm="inside_to_outside",
            stable_points=[(1, 1)],
            distance1=0.1,
            distance2=100,
        )
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, len(coords_only))
        self.assertTrue(
            any(
                (row["x"] == 1 and row["y"] == 1)
                for _, row in train.iterrows()
            )
        )

    def test_get_subset_basic_filter(self):
        filters = {"x": lambda x: x > 2}
        result = self.dataset.get_subset(
            property_name="target",
            filters=filters,
            models_to_include=["modelA", "modelB"],
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            (result["modelA"].notna()).all()
        )  # Check filtered rows exist
        self.assertTrue(
            all(col in result.columns for col in ["modelA", "modelB"])
        )

    def test_view_data_available_properties_and_models(self):
        result = self.dataset.view_data()
        self.assertIn("available_properties", result)
        self.assertIn("available_models", result)

    def test_view_data_specific_property(self):
        result = self.dataset.view_data(property_name="target")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue("modelA" in result.columns)

    def test_separate_points_distance_allSets_edge_cases(self):
        coords_only = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        self.dataset.data = {"target": coords_only}

        # Adjusted test case with clearer distances
        list1 = [(1, 1), (2, 2)]
        list2 = [(1.1, 1.1), (3, 3)]
        distance1 = 0.2
        distance2 = 1.5

        train, val, test = self.dataset.separate_points_distance_allSets(
            list1=list1, list2=list2, distance1=distance1, distance2=distance2
        )

        # Validate the results
        self.assertEqual(len(train), 1)  # Only (1, 1) should be in train
        self.assertEqual(len(val), 1)    # Only (2, 2) should be in validation
        self.assertEqual(len(test), 0)   # No points should be in test

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_hdf")
    def test_verbose_enabled(self, mock_read_hdf, mock_exists):
        """Test that warnings are logged when verbose=True (default)."""
        # Mock missing columns scenario
        model_data_incomplete = pd.DataFrame(
            {"x": [1, 2], "y": [1, 2]}  # Missing 'target' column
        )
        mock_read_hdf.side_effect = lambda file, key: model_data_incomplete

        dataset = Dataset(data_source="fake_path.h5", verbose=True)
        
        # Capture logging output
        with self.assertLogs('pybmc.data', level='INFO') as cm:
            result = dataset.load_data(
                models=["modelA"],
                keys=["target"],
                domain_keys=["x", "y"],
            )
        
        # Verify that the warning was logged
        self.assertTrue(any("[Skipped]" in msg or "[Warning]" in msg for msg in cm.output))

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_hdf")
    def test_verbose_disabled(self, mock_read_hdf, mock_exists):
        """Test that warnings are suppressed when verbose=False."""
        # Mock missing columns scenario
        model_data_incomplete = pd.DataFrame(
            {"x": [1, 2], "y": [1, 2]}  # Missing 'target' column
        )
        mock_read_hdf.side_effect = lambda file, key: model_data_incomplete

        dataset = Dataset(data_source="fake_path.h5", verbose=False)
        
        # Since verbose=False sets logger to WARNING level, INFO messages should not appear
        # We'll verify by checking that the logger level is set correctly
        self.assertEqual(dataset.logger.level, logging.WARNING)
        
        result = dataset.load_data(
            models=["modelA"],
            keys=["target"],
            domain_keys=["x", "y"],
        )
        
        # Result should still be returned properly (empty dataframe for missing property)
        self.assertIn("target", result)
        self.assertIsInstance(result["target"], pd.DataFrame)

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_hdf")
    def test_load_data_with_smaller_truth_domain_h5(self, mock_read_hdf, mock_exists):
        """Test that truth data can have a smaller domain than models when using truth_column_name."""
        # Model data has domain points: (1,1), (2,2), (3,3), (4,4)
        def mock_hdf_reader(file, key):
            if key == "modelA":
                return pd.DataFrame({
                    "x": [1, 2, 3, 4],
                    "y": [1, 2, 3, 4],
                    "target": [10, 20, 30, 40]
                })
            elif key == "modelB":
                return pd.DataFrame({
                    "x": [1, 2, 3, 4],
                    "y": [1, 2, 3, 4],
                    "target": [11, 21, 31, 41]
                })
            elif key == "truth":
                # Truth data has only 2 points: (1,1), (2,2)
                return pd.DataFrame({
                    "x": [1, 2],
                    "y": [1, 2],
                    "target": [10.5, 20.5]
                })
        
        mock_read_hdf.side_effect = mock_hdf_reader
        
        dataset = Dataset(data_source="fake_path.h5")
        result = dataset.load_data(
            models=["modelA", "modelB", "truth"],
            keys=["target"],
            domain_keys=["x", "y"],
            truth_column_name="truth"
        )
        
        # Result should have all 4 domain points from the models
        self.assertIn("target", result)
        df = result["target"]
        self.assertEqual(len(df), 4, "Should have all 4 domain points from models")
        
        # All columns should be present
        self.assertTrue(all(col in df.columns for col in ["x", "y", "modelA", "modelB", "truth"]))
        
        # Model data should be complete (no NaN)
        self.assertTrue(df["modelA"].notna().all(), "modelA should have no NaN values")
        self.assertTrue(df["modelB"].notna().all(), "modelB should have no NaN values")
        
        # Truth data should have NaN for points (3,3) and (4,4)
        self.assertEqual(df["truth"].isna().sum(), 2, "truth should have 2 NaN values for missing domain points")
        self.assertTrue(df.loc[(df["x"] == 1) & (df["y"] == 1), "truth"].notna().all(), "truth should have value at (1,1)")
        self.assertTrue(df.loc[(df["x"] == 2) & (df["y"] == 2), "truth"].notna().all(), "truth should have value at (2,2)")
        self.assertTrue(df.loc[(df["x"] == 3) & (df["y"] == 3), "truth"].isna().all(), "truth should be NaN at (3,3)")
        self.assertTrue(df.loc[(df["x"] == 4) & (df["y"] == 4), "truth"].isna().all(), "truth should be NaN at (4,4)")

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_csv")
    def test_load_data_with_smaller_truth_domain_csv(self, mock_read_csv, mock_exists):
        """Test that truth data can have a smaller domain than models in CSV format."""
        # CSV with models having 4 points and truth having 2 points
        mock_read_csv.return_value = pd.DataFrame({
            "x": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            "y": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            "target": [10, 20, 30, 40, 11, 21, 31, 41, 10.5, 20.5],
            "model": ["modelA", "modelA", "modelA", "modelA", 
                     "modelB", "modelB", "modelB", "modelB",
                     "truth", "truth"]
        })
        
        dataset = Dataset(data_source="fake_path.csv")
        result = dataset.load_data(
            models=["modelA", "modelB", "truth"],
            keys=["target"],
            domain_keys=["x", "y"],
            model_column="model",
            truth_column_name="truth"
        )
        
        # Result should have all 4 domain points from the models
        self.assertIn("target", result)
        df = result["target"]
        self.assertEqual(len(df), 4, "Should have all 4 domain points from models")
        
        # All columns should be present
        self.assertTrue(all(col in df.columns for col in ["x", "y", "modelA", "modelB", "truth"]))
        
        # Model data should be complete (no NaN)
        self.assertTrue(df["modelA"].notna().all(), "modelA should have no NaN values")
        self.assertTrue(df["modelB"].notna().all(), "modelB should have no NaN values")
        
        # Truth data should have NaN for points (3,3) and (4,4)
        self.assertEqual(df["truth"].isna().sum(), 2, "truth should have 2 NaN values for missing domain points")

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_hdf")
    def test_load_data_without_truth_column_name_backward_compat(self, mock_read_hdf, mock_exists):
        """Test backward compatibility: without truth_column_name, all models should be inner-joined."""
        def mock_hdf_reader(file, key):
            if key == "modelA":
                return pd.DataFrame({
                    "x": [1, 2, 3, 4],
                    "y": [1, 2, 3, 4],
                    "target": [10, 20, 30, 40]
                })
            elif key == "modelB":
                return pd.DataFrame({
                    "x": [1, 2, 3, 4],
                    "y": [1, 2, 3, 4],
                    "target": [11, 21, 31, 41]
                })
            elif key == "truth":
                # Truth data has only 2 points
                return pd.DataFrame({
                    "x": [1, 2],
                    "y": [1, 2],
                    "target": [10.5, 20.5]
                })
        
        mock_read_hdf.side_effect = mock_hdf_reader
        
        dataset = Dataset(data_source="fake_path.h5")
        # Without truth_column_name, should use inner join (old behavior)
        result = dataset.load_data(
            models=["modelA", "modelB", "truth"],
            keys=["target"],
            domain_keys=["x", "y"]
            # NOTE: truth_column_name not provided
        )
        
        # Result should have only 2 domain points (intersection of all)
        self.assertIn("target", result)
        df = result["target"]
        self.assertEqual(len(df), 2, "Should have only 2 domain points (intersection)")
        
        # All columns should be present with no NaN values
        self.assertTrue(all(col in df.columns for col in ["x", "y", "modelA", "modelB", "truth"]))
        self.assertTrue(df["modelA"].notna().all())
        self.assertTrue(df["modelB"].notna().all())
        self.assertTrue(df["truth"].notna().all())


if __name__ == "__main__":
    unittest.main()
