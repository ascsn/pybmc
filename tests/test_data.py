import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch
from pybmc.data import Dataset

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


if __name__ == "__main__":
    unittest.main()
