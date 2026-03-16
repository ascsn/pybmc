import unittest
import pandas as pd
import numpy as np
from pybmc.bmc import BayesianModelCombination


class TestBayesianModelCombination(unittest.TestCase):
    def setUp(self):
        # Expanded mock dataset (6 rows)
        self.df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "y": [10, 11, 12, 13, 14, 15],
                "truth": [11, 21, 31, 41, 51, 61],
                "model1": [10, 20, 30, 40, 50, 60],
                "model2": [15, 25, 35, 45, 55, 65],
                "model3": [12, 30, 32, 43, 58, 67],
            }
        )

        self.property = "target"
        self.data_dict = {self.property: self.df}
        self.models = ["model1", "model2", "model3", "truth"]

        self.bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
        )

        # Manually designate training data (first 4 rows)
        self.train_df = self.df.iloc[:4]

    def test_bmc_init(self):
        # Valid inputs
        models_list = ["model1", "model2"]
        data_dict = {"property": pd.DataFrame({"model1": [1, 2], "model2": [3, 4]})}
        truth_column_name = "truth"

        bmc = BayesianModelCombination(models_list, data_dict, truth_column_name)

        self.assertEqual(bmc.models_list, models_list)
        self.assertEqual(bmc.data_dict, data_dict)
        self.assertEqual(bmc.truth_column_name, truth_column_name)

        # Invalid models_list
        with self.assertRaises(ValueError):
            BayesianModelCombination("not_a_list", data_dict, truth_column_name)

        # Invalid data_dict
        with self.assertRaises(ValueError):
            BayesianModelCombination(models_list, "not_a_dict", truth_column_name)

    def test_orthogonalize(self):
        self.bmc.orthogonalize(
            property=self.property, train_df=self.train_df, components_kept=2
        )
        self.assertTrue(hasattr(self.bmc, "centered_experiment_train"))
        self.assertEqual(self.bmc.U_hat.shape[0], self.train_df.shape[0])
        self.assertEqual(self.bmc.Vt_hat.shape[1], len(self.bmc.models))

        models_list = ["model1", "model2"]
        data_dict = {
            "property": pd.DataFrame(
                {"model1": [1, 2], "model2": [3, 4], "truth": [5, 6]}
            )
        }
        truth_column_name = "truth"
        bmc = BayesianModelCombination(models_list, data_dict, truth_column_name)

        train_df = pd.DataFrame({"model1": [1, 2], "model2": [3, 4], "truth": [5, 6]})
        components_kept = 1

        bmc.orthogonalize("property", train_df, components_kept)

        self.assertEqual(bmc.current_property, "property")
        self.assertIsNotNone(bmc.centered_experiment_train)
        self.assertIsNotNone(bmc.U_hat)
        self.assertIsNotNone(bmc.Vt_hat)
        self.assertIsNotNone(bmc.S_hat)
        self.assertIsNotNone(bmc._predictions_mean_train)

    def test_train(self):
        self.bmc.orthogonalize(
            property=self.property, train_df=self.train_df, components_kept=2
        )
        self.bmc.train()
        self.assertEqual(
            self.bmc.samples.shape[1], 3
        )  # 2 components + 1 sigma

        models_list = ["model1", "model2"]
        data_dict = {"property": pd.DataFrame({"model1": [1, 2], "model2": [3, 4], "truth": [5, 6]})}
        truth_column_name = "truth"
        bmc = BayesianModelCombination(models_list, data_dict, truth_column_name)

        train_df = pd.DataFrame({"model1": [1, 2], "model2": [3, 4], "truth": [5, 6]})
        components_kept = 1

        # Perform orthogonalization first
        bmc.orthogonalize("property", train_df, components_kept)

        # Default training options
        bmc.train()
        self.assertIsNotNone(bmc.samples)

        # Custom training options
        training_options = {
            "iterations": 100,
            "sampler": "simplex",
            "burn": 10,
            "stepsize": 0.01,
            "b_mean_prior": np.zeros(components_kept),
            "b_mean_cov": np.eye(components_kept),
            "nu0_chosen": 1.0,
            "sigma20_chosen": 0.02,
        }
        bmc.train(training_options=training_options)
        self.assertIsNotNone(bmc.samples)
        self.assertEqual(bmc.samples.shape[0], training_options["iterations"])

    def test_predict(self):
        self.bmc.orthogonalize(
            property=self.property, train_df=self.train_df, components_kept=2
        )
        self.bmc.train()
        # Use all rows for prediction input
        X = self.df[["x", "y", "model1", "model2", "model3"]].copy()
        rndm_m, lower_df, median_df, upper_df, weights = self.bmc.predict(self.property)

        self.assertEqual(rndm_m.shape[1], len(X))
        self.assertIn("Predicted_Lower", lower_df.columns)
        self.assertIn("Predicted_Median", median_df.columns)
        self.assertIn("Predicted_Upper", upper_df.columns)

    def test_evaluate(self):
        # Run orthogonalization and training first
        self.bmc.orthogonalize(
            property=self.property, train_df=self.train_df, components_kept=2
        )
        self.bmc.train()
        eval_results = self.bmc.evaluate()

        # Assertions
        self.assertIsInstance(eval_results, list)
        self.assertEqual(len(eval_results), 21)
        self.assertTrue(all(isinstance(x, (int, float)) for x in eval_results))

    def test_bmc_predict(self):
        models_list = ["model1", "model2"]
        data_dict = {"property": pd.DataFrame({"N": [1, 2], "Z": [10, 20], "model1": [1, 2], "model2": [3, 4], "truth": [5, 6]})}
        truth_column_name = "truth"
        bmc = BayesianModelCombination(models_list, data_dict, truth_column_name)

        train_df = pd.DataFrame({"N": [1, 2], "Z": [10, 20], "model1": [1, 2], "model2": [3, 4], "truth": [5, 6]})
        components_kept = 1

        # Perform orthogonalization and training
        bmc.orthogonalize("property", train_df, components_kept)
        bmc.train()

        # Perform prediction using property name
        rndm_m, lower_df, median_df, upper_df, weights = bmc.predict("property")

        self.assertIsNotNone(rndm_m)
        self.assertIsInstance(lower_df, pd.DataFrame)
        self.assertIsInstance(median_df, pd.DataFrame)
        self.assertIsInstance(upper_df, pd.DataFrame)
        self.assertFalse(lower_df.empty)
        self.assertFalse(median_df.empty)
        self.assertFalse(upper_df.empty)
        self.assertIsNotNone(weights)
        # Check domain columns are present
        for col in ["N", "Z"]:
            self.assertIn(col, lower_df.columns)
            self.assertIn(col, median_df.columns)
            self.assertIn(col, upper_df.columns)

    def test_bmc_evaluate(self):
        models_list = ["model1", "model2"]
        data_dict = {"property": pd.DataFrame({"model1": [1, 2], "model2": [3, 4], "truth": [5, 6]})}
        truth_column_name = "truth"
        bmc = BayesianModelCombination(models_list, data_dict, truth_column_name)

        train_df = pd.DataFrame({"model1": [1, 2], "model2": [3, 4], "truth": [5, 6]})
        components_kept = 1

        # Perform orthogonalization and training
        bmc.orthogonalize("property", train_df, components_kept)
        bmc.train()

        # Perform evaluation
        coverage_results = bmc.evaluate()

        assert coverage_results is not None
        assert isinstance(coverage_results, list)
        assert all(isinstance(c, float) for c in coverage_results)


if __name__ == "__main__":
    unittest.main()
