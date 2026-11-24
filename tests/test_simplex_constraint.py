"""Tests for the simplex constraint mode in BayesianModelCombination."""
import unittest
import numpy as np
import pandas as pd
from pybmc.bmc import BayesianModelCombination


class TestSimplexConstraintInit(unittest.TestCase):
    """Test initialization with constraint parameter."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "truth": [11, 21, 31, 41, 51, 61],
                "model1": [10, 20, 30, 40, 50, 60],
                "model2": [15, 25, 35, 45, 55, 65],
                "model3": [12, 30, 32, 43, 58, 67],
            }
        )
        self.data_dict = {"target": self.df}
        self.models = ["model1", "model2", "model3"]

    def test_default_constraint_is_unconstrained(self):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
        )
        self.assertEqual(bmc.constraint, "unconstrained")

    def test_explicit_unconstrained(self):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
            constraint="unconstrained",
        )
        self.assertEqual(bmc.constraint, "unconstrained")

    def test_simplex_constraint(self):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
            constraint="simplex",
        )
        self.assertEqual(bmc.constraint, "simplex")

    def test_invalid_constraint_raises(self):
        with self.assertRaises(ValueError) as ctx:
            BayesianModelCombination(
                models_list=self.models,
                data_dict=self.data_dict,
                truth_column_name="truth",
                constraint="invalid",
            )
        self.assertIn("Invalid constraint", str(ctx.exception))

    def test_valid_constraints_tuple(self):
        self.assertEqual(
            BayesianModelCombination.VALID_CONSTRAINTS,
            ("unconstrained", "simplex"),
        )


class TestSimplexConstraintTraining(unittest.TestCase):
    """Test training with simplex constraint mode."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "truth": [11, 21, 31, 41, 51, 61],
                "model1": [10, 20, 30, 40, 50, 60],
                "model2": [15, 25, 35, 45, 55, 65],
                "model3": [12, 30, 32, 43, 58, 67],
            }
        )
        self.data_dict = {"target": self.df}
        self.models = ["model1", "model2", "model3"]
        self.train_df = self.df.iloc[:4]

    def _make_bmc(self, constraint="unconstrained"):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
            constraint=constraint,
        )
        bmc.orthogonalize(
            property="target",
            train_df=self.train_df,
            components_kept=2,
        )
        return bmc

    def test_train_unconstrained_default(self):
        bmc = self._make_bmc("unconstrained")
        bmc.train(training_options={"iterations": 200})
        self.assertIsNotNone(bmc.samples)
        # 2 components + 1 sigma
        self.assertEqual(bmc.samples.shape, (200, 3))

    def test_train_simplex_via_init(self):
        bmc = self._make_bmc("simplex")
        bmc.train(
            training_options={
                "iterations": 200,
                "burn": 50,
                "stepsize": 0.01,
            }
        )
        self.assertIsNotNone(bmc.samples)
        self.assertEqual(bmc.samples.shape[0], 200)
        # 2 components + 1 sigma
        self.assertEqual(bmc.samples.shape[1], 3)

    def test_train_simplex_via_training_options_override(self):
        """An unconstrained BMC can be overridden to simplex per train call."""
        bmc = self._make_bmc("unconstrained")
        bmc.train(
            training_options={
                "iterations": 200,
                "sampler": "simplex",
                "burn": 50,
                "stepsize": 0.01,
            }
        )
        self.assertIsNotNone(bmc.samples)
        self.assertEqual(bmc.samples.shape[0], 200)

    def test_train_unconstrained_via_training_options_override(self):
        """A simplex BMC can be overridden to unconstrained per train call."""
        bmc = self._make_bmc("simplex")
        bmc.train(
            training_options={
                "iterations": 200,
                "sampler": "unconstrained",
            }
        )
        self.assertIsNotNone(bmc.samples)
        self.assertEqual(bmc.samples.shape[0], 200)

    def test_invalid_sampler_in_training_options_raises(self):
        bmc = self._make_bmc("unconstrained")
        with self.assertRaises(ValueError) as ctx:
            bmc.train(training_options={"sampler": "bogus"})
        self.assertIn("Invalid sampler", str(ctx.exception))


class TestSimplexWeights(unittest.TestCase):
    """Test that simplex-constrained weights satisfy the simplex property."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "truth": [11, 21, 31, 41, 51, 61],
                "model1": [10, 20, 30, 40, 50, 60],
                "model2": [15, 25, 35, 45, 55, 65],
                "model3": [12, 30, 32, 43, 58, 67],
            }
        )
        self.data_dict = {"target": self.df}
        self.models = ["model1", "model2", "model3"]
        self.train_df = self.df.iloc[:4]

    def _trained_bmc(self, constraint, iterations=500, burn=100):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
            constraint=constraint,
        )
        bmc.orthogonalize("target", self.train_df, components_kept=2)
        opts = {"iterations": iterations}
        if constraint == "simplex":
            opts["burn"] = burn
            opts["stepsize"] = 0.01
        bmc.train(training_options=opts)
        return bmc

    def test_simplex_weights_nonnegative(self):
        bmc = self._trained_bmc("simplex", iterations=500, burn=200)
        weight_matrix = bmc.get_weights(summary=False)
        self.assertTrue(
            np.all(weight_matrix >= -1e-10),
            "Simplex weights should be non-negative",
        )

    def test_simplex_weights_sum_to_one(self):
        bmc = self._trained_bmc("simplex", iterations=500, burn=200)
        weight_matrix = bmc.get_weights(summary=False)
        row_sums = np.sum(weight_matrix, axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(len(row_sums)),
            atol=1e-10,
            err_msg="Simplex weights should sum to 1",
        )

    def test_get_weights_summary(self):
        bmc = self._trained_bmc("unconstrained", iterations=500)
        summary = bmc.get_weights(summary=True)
        self.assertIn("mean", summary)
        self.assertIn("std", summary)
        self.assertIn("median", summary)
        self.assertIn("models", summary)
        self.assertEqual(len(summary["mean"]), 3)
        self.assertEqual(len(summary["models"]), 3)
        self.assertEqual(summary["models"], self.models)

    def test_get_weights_full_matrix(self):
        bmc = self._trained_bmc("unconstrained", iterations=500)
        weight_matrix = bmc.get_weights(summary=False)
        self.assertEqual(weight_matrix.shape, (500, 3))

    def test_get_weights_before_training_raises(self):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
        )
        with self.assertRaises(ValueError):
            bmc.get_weights()


class TestSimplexPredictAndEvaluate(unittest.TestCase):
    """Test predict and evaluate work correctly in simplex mode."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "truth": [11, 21, 31, 41, 51, 61],
                "model1": [10, 20, 30, 40, 50, 60],
                "model2": [15, 25, 35, 45, 55, 65],
                "model3": [12, 30, 32, 43, 58, 67],
            }
        )
        self.data_dict = {"target": self.df}
        self.models = ["model1", "model2", "model3"]
        self.train_df = self.df.iloc[:4]

    def _trained_bmc(self, constraint):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
            constraint=constraint,
        )
        bmc.orthogonalize("target", self.train_df, components_kept=2)
        opts = {"iterations": 500}
        if constraint == "simplex":
            opts["burn"] = 100
            opts["stepsize"] = 0.01
        bmc.train(training_options=opts)
        return bmc

    def test_simplex_predict(self):
        bmc = self._trained_bmc("simplex")
        rndm_m, lower_df, median_df, upper_df = bmc.predict("target")
        self.assertEqual(rndm_m.shape[1], 6)
        self.assertIn("Predicted_Lower", lower_df.columns)
        self.assertIn("Predicted_Median", median_df.columns)
        self.assertIn("Predicted_Upper", upper_df.columns)

    def test_simplex_evaluate(self):
        bmc = self._trained_bmc("simplex")
        coverage_results = bmc.evaluate()
        self.assertIsInstance(coverage_results, list)
        self.assertEqual(len(coverage_results), 21)
        self.assertTrue(all(isinstance(c, float) for c in coverage_results))

    def test_unconstrained_predict(self):
        bmc = self._trained_bmc("unconstrained")
        rndm_m, lower_df, median_df, upper_df = bmc.predict("target")
        self.assertEqual(rndm_m.shape[1], 6)

    def test_unconstrained_evaluate(self):
        bmc = self._trained_bmc("unconstrained")
        coverage_results = bmc.evaluate()
        self.assertIsInstance(coverage_results, list)
        self.assertEqual(len(coverage_results), 21)


class TestSimplexIntegration(unittest.TestCase):
    """Integration test: full pipeline with simplex constraint."""

    def test_full_simplex_pipeline(self):
        """End-to-end: init → orthogonalize → train(simplex) → predict → evaluate → get_weights."""
        df = pd.DataFrame(
            {
                "N": [1, 2, 3, 4, 5, 6],
                "Z": [10, 20, 30, 40, 50, 60],
                "truth": [102, 203, 301, 404, 500, 610],
                "model1": [100, 205, 295, 410, 495, 605],
                "model2": [95, 210, 305, 405, 510, 600],
                "model3": [105, 195, 310, 395, 505, 615],
            }
        )
        data_dict = {"BE": df}
        train_df = df.iloc[:4]

        bmc = BayesianModelCombination(
            models_list=["model1", "model2", "model3"],
            data_dict=data_dict,
            truth_column_name="truth",
            constraint="simplex",
        )

        bmc.orthogonalize("BE", train_df, components_kept=2)
        bmc.train(
            training_options={
                "iterations": 500,
                "burn": 100,
                "stepsize": 0.01,
            }
        )

        # Predict
        rndm_m, lower_df, median_df, upper_df = bmc.predict("BE")
        self.assertEqual(rndm_m.shape[1], 6)
        self.assertIn("N", lower_df.columns)
        self.assertIn("Z", lower_df.columns)

        # Evaluate
        coverage_results = bmc.evaluate()
        self.assertEqual(len(coverage_results), 21)

        # Weights should be on simplex
        weight_matrix = bmc.get_weights(summary=False)
        self.assertTrue(np.all(weight_matrix >= -1e-10))
        row_sums = np.sum(weight_matrix, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

        # Summary
        summary = bmc.get_weights(summary=True)
        self.assertEqual(summary["models"], ["model1", "model2", "model3"])
        np.testing.assert_allclose(
            np.sum(summary["mean"]), 1.0, atol=0.1,
            err_msg="Mean weights should approximately sum to 1",
        )

    def test_switching_between_modes(self):
        """Test switching between unconstrained and simplex across train calls."""
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "truth": [11, 21, 31, 41, 51, 61],
                "model1": [10, 20, 30, 40, 50, 60],
                "model2": [15, 25, 35, 45, 55, 65],
            }
        )
        data_dict = {"target": df}
        train_df = df.iloc[:4]

        bmc = BayesianModelCombination(
            models_list=["model1", "model2"],
            data_dict=data_dict,
            truth_column_name="truth",
            constraint="unconstrained",
        )
        bmc.orthogonalize("target", train_df, components_kept=1)

        # Train unconstrained first
        bmc.train(training_options={"iterations": 200})
        w_unconstrained = bmc.get_weights(summary=False)
        self.assertEqual(w_unconstrained.shape[0], 200)

        # Re-train with simplex override
        bmc.train(
            training_options={
                "iterations": 200,
                "sampler": "simplex",
                "burn": 50,
                "stepsize": 0.01,
            }
        )
        w_simplex = bmc.get_weights(summary=False)
        self.assertEqual(w_simplex.shape[0], 200)
        self.assertTrue(np.all(w_simplex >= -1e-10))
        row_sums = np.sum(w_simplex, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
