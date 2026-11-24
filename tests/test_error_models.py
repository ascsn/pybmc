"""Tests for the heteroscedastic error models."""
import unittest
import numpy as np
import pandas as pd

from pybmc.bmc import BayesianModelCombination
from pybmc.error_models import (
    VARIANCE_MODELS,
    HeteroscedasticMetrics,
    required_metrics,
    variance_basis,
    variance_parameter_names,
)
from pybmc.inference_utils import gibbs_sampler_heteroscedastic
from pybmc.sampling_utils import (
    coverage_quality,
    diagnose_coverage_shape,
    mace,
    reduced_chi_square,
)

HETERO_MODELS = [m for m in VARIANCE_MODELS if m != "homoscedastic"]


def make_heteroscedastic_data(n_points=60, n_models=4, seed=7):
    """Synthetic dataset whose noise grows with the model spread."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n_points)
    base = 5.0 + 2.0 * x

    data = {"x": x}
    preds = []
    for j in range(n_models):
        # Models disagree more at large x.
        offset = rng.normal(0, 0.2) + rng.normal(0, 0.05) * x
        preds.append(base + offset)
        data[f"model{j + 1}"] = preds[j]

    spread = np.var(np.column_stack(preds), axis=1)
    noise = rng.normal(0, np.sqrt(0.05 + 2.0 * spread))
    data["truth"] = base + noise
    return pd.DataFrame(data)


class TestErrorModelRegistry(unittest.TestCase):
    def test_registry_contents(self):
        self.assertIn("homoscedastic", VARIANCE_MODELS)
        self.assertEqual(len(VARIANCE_MODELS), 7)
        self.assertEqual(VARIANCE_MODELS["homoscedastic"], [])
        self.assertEqual(
            VARIANCE_MODELS["hetero_combined_quadratic"],
            [("pc_dist", 1), ("pc_dist", 2), ("model_var", 1), ("model_var", 2)],
        )

    def test_required_metrics(self):
        self.assertEqual(required_metrics("homoscedastic"), [])
        self.assertEqual(required_metrics("hetero_pc_dist"), ["pc_dist"])
        self.assertEqual(
            required_metrics("hetero_combined_linear"),
            ["model_var", "pc_dist"],
        )
        with self.assertRaises(ValueError):
            required_metrics("bogus")

    def test_variance_parameter_names(self):
        self.assertEqual(variance_parameter_names("homoscedastic"), ["sigma^2"])
        self.assertEqual(
            variance_parameter_names("hetero_model_var_quad"),
            ["alpha", "beta_model_var^1", "beta_model_var^2"],
        )
        with self.assertRaises(ValueError):
            variance_parameter_names("bogus")


class TestHeteroscedasticMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.train_preds = rng.normal(size=(30, 4)) + 10 * rng.normal(size=(30, 1))
        centered = self.train_preds - self.train_preds.mean(axis=1, keepdims=True)
        _, S, Vt = np.linalg.svd(centered)
        self.Vt_hat = np.array([Vt[i] / S[i] for i in range(2)])

    def test_fit_and_compute_on_training_data(self):
        calc = HeteroscedasticMetrics(["pc_dist", "model_var"]).fit(
            self.train_preds, self.Vt_hat
        )
        metrics = calc.compute(self.train_preds)
        for name in ("pc_dist", "model_var"):
            self.assertEqual(metrics[name].shape, (30,))
            # Training metrics are min-max scaled to [0, 1].
            self.assertAlmostEqual(float(np.min(metrics[name])), 0.0)
            self.assertAlmostEqual(float(np.max(metrics[name])), 1.0)

    def test_extrapolation_can_exceed_one(self):
        calc = HeteroscedasticMetrics(["model_var"]).fit(
            self.train_preds, self.Vt_hat
        )
        far = self.train_preds * 10  # much larger spread among models
        metrics = calc.compute(far)
        self.assertGreater(float(np.max(metrics["model_var"])), 1.0)

    def test_compute_before_fit_raises(self):
        calc = HeteroscedasticMetrics(["pc_dist"])
        with self.assertRaises(ValueError):
            calc.compute(self.train_preds)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            HeteroscedasticMetrics(["bogus"])

    def test_variance_basis_shape(self):
        calc = HeteroscedasticMetrics(["pc_dist", "model_var"]).fit(
            self.train_preds, self.Vt_hat
        )
        metrics = calc.compute(self.train_preds)
        basis = variance_basis(
            metrics, VARIANCE_MODELS["hetero_combined_quadratic"]
        )
        self.assertEqual(basis.shape, (30, 5))
        np.testing.assert_allclose(basis[:, 0], 1.0)
        np.testing.assert_allclose(basis[:, 2], basis[:, 1] ** 2)


class TestHeteroscedasticSampler(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 50
        self.X = rng.normal(size=(n, 2))
        metric = np.linspace(0, 1, n)
        true_sigma2 = 0.1 + 0.5 * metric
        self.y = self.X @ np.array([1.5, -0.5]) + rng.normal(
            0, np.sqrt(true_sigma2)
        )
        self.basis = np.column_stack([np.ones(n), metric])

    def test_sampler_shapes_and_finiteness(self):
        samples, acceptance = gibbs_sampler_heteroscedastic(
            self.y, self.X, self.basis, iterations=200, burn=100
        )
        self.assertEqual(samples.shape, (200, 4))  # 2 betas + alpha + beta_1
        self.assertFalse(np.any(np.isnan(samples)))
        self.assertGreaterEqual(acceptance, 0.0)
        self.assertLessEqual(acceptance, 1.0)
        # Variance parameters stay positive.
        self.assertTrue(np.all(samples[:, 2:] > 0))

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            gibbs_sampler_heteroscedastic(
                self.y, self.X, self.basis, iterations=10, burn=-1
            )
        no_ones = np.column_stack([self.basis[:, 1], self.basis[:, 1]])
        with self.assertRaises(ValueError):
            gibbs_sampler_heteroscedastic(
                self.y, self.X, no_ones, iterations=10, burn=0
            )
        with self.assertRaises(ValueError):
            gibbs_sampler_heteroscedastic(
                self.y, self.X, self.basis, iterations=10, burn=0,
                proposal_scales=[0.1],  # wrong length
            )
        negative_basis = self.basis.copy()
        negative_basis[:, 1] -= 1.0  # negative entries break positivity
        with self.assertRaises(ValueError):
            gibbs_sampler_heteroscedastic(
                self.y, self.X, negative_basis, iterations=10, burn=0
            )

    def test_sampler_seed_reproducible(self):
        s1, _ = gibbs_sampler_heteroscedastic(
            self.y, self.X, self.basis, iterations=50, burn=20, seed=11
        )
        s2, _ = gibbs_sampler_heteroscedastic(
            self.y, self.X, self.basis, iterations=50, burn=20, seed=11
        )
        np.testing.assert_array_equal(s1, s2)


class TestBMCErrorModelInit(unittest.TestCase):
    def setUp(self):
        self.df = make_heteroscedastic_data()
        self.data_dict = {"target": self.df}
        self.models = ["model1", "model2", "model3", "model4"]

    def test_default_error_model_is_homoscedastic(self):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
        )
        self.assertEqual(bmc.error_model, "homoscedastic")

    def test_valid_error_models_tuple(self):
        self.assertEqual(
            set(BayesianModelCombination.VALID_ERROR_MODELS),
            set(VARIANCE_MODELS),
        )

    def test_invalid_error_model_raises(self):
        with self.assertRaises(ValueError) as ctx:
            BayesianModelCombination(
                models_list=self.models,
                data_dict=self.data_dict,
                truth_column_name="truth",
                error_model="bogus",
            )
        self.assertIn("Invalid error model", str(ctx.exception))

    def test_simplex_with_hetero_raises(self):
        with self.assertRaises(ValueError):
            BayesianModelCombination(
                models_list=self.models,
                data_dict=self.data_dict,
                truth_column_name="truth",
                constraint="simplex",
                error_model="hetero_pc_dist",
            )

    def test_simplex_override_with_hetero_raises_in_train(self):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
            error_model="hetero_pc_dist",
        )
        bmc.orthogonalize("target", self.df.iloc[:40], components_kept=2)
        with self.assertRaises(ValueError):
            bmc.train(training_options={"sampler": "simplex"})

    def test_invalid_error_model_in_training_options_raises(self):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
        )
        bmc.orthogonalize("target", self.df.iloc[:40], components_kept=2)
        with self.assertRaises(ValueError):
            bmc.train(training_options={"error_model": "bogus"})


class TestBMCHeteroscedasticTraining(unittest.TestCase):
    def setUp(self):
        self.df = make_heteroscedastic_data()
        self.data_dict = {"target": self.df}
        self.models = ["model1", "model2", "model3", "model4"]
        self.train_df = self.df.iloc[:40]
        self.components = 2

    def _trained_bmc(self, error_model, iterations=300, burn=100):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
            error_model=error_model,
        )
        bmc.orthogonalize("target", self.train_df, components_kept=self.components)
        bmc.train(training_options={"iterations": iterations, "burn": burn})
        return bmc

    def test_all_hetero_models_train_predict_evaluate(self):
        for error_model in HETERO_MODELS:
            with self.subTest(error_model=error_model):
                bmc = self._trained_bmc(error_model)

                n_variance_params = 1 + len(VARIANCE_MODELS[error_model])
                self.assertEqual(
                    bmc.samples.shape,
                    (300, self.components + n_variance_params),
                )
                self.assertIsNotNone(bmc.mh_acceptance_rate_)

                rndm_m, lower_df, median_df, upper_df = bmc.predict("target")
                self.assertEqual(rndm_m.shape[1], len(self.df))
                self.assertTrue(
                    np.all(
                        upper_df["Predicted_Upper"].values
                        >= lower_df["Predicted_Lower"].values
                    )
                )
                self.assertFalse(np.any(np.isnan(rndm_m)))

                coverage_results = bmc.evaluate()
                self.assertEqual(len(coverage_results), 21)

                weights = bmc.get_weights(summary=False)
                self.assertEqual(weights.shape, (300, len(self.models)))
                # Weights always sum to 1 (Vt rows are orthogonal to ones).
                np.testing.assert_allclose(
                    weights.sum(axis=1), 1.0, atol=1e-8
                )

    def test_hetero_predict_default_seed_is_reproducible(self):
        bmc = self._trained_bmc("hetero_combined_linear")
        rndm_m1, *_ = bmc.predict("target")
        rndm_m2, *_ = bmc.predict("target")
        np.testing.assert_array_equal(rndm_m1, rndm_m2)

    def test_hetero_predict_different_seeds_give_different_draws(self):
        bmc = self._trained_bmc("hetero_combined_linear")
        rndm_m1, *_ = bmc.predict("target", seed=1)
        rndm_m2, *_ = bmc.predict("target", seed=2)
        self.assertFalse(np.array_equal(rndm_m1, rndm_m2))

    def test_error_model_override_in_training_options(self):
        bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
        )
        bmc.orthogonalize("target", self.train_df, components_kept=2)
        bmc.train(
            training_options={
                "iterations": 200,
                "burn": 50,
                "error_model": "hetero_model_var",
            }
        )
        # 2 betas + alpha + beta_1
        self.assertEqual(bmc.samples.shape, (200, 4))
        self.assertEqual(bmc._trained_error_model, "hetero_model_var")

    def test_retrain_homoscedastic_after_hetero(self):
        bmc = self._trained_bmc("hetero_model_var")
        bmc.train(
            training_options={"iterations": 200, "error_model": "homoscedastic"}
        )
        self.assertEqual(bmc.samples.shape, (200, 3))  # 2 betas + sigma^2
        rndm_m, *_ = bmc.predict("target")
        self.assertEqual(rndm_m.shape[1], len(self.df))

    def test_hetero_widens_intervals_at_extrapolation(self):
        """Interval width should grow with the metric for a linear model."""
        bmc = self._trained_bmc("hetero_model_var", iterations=500, burn=200)
        _, lower_df, _, upper_df = bmc.predict("target")
        width = (
            upper_df["Predicted_Upper"].values
            - lower_df["Predicted_Lower"].values
        )
        spread = np.var(self.df[self.models].values, axis=1)
        # The widest-interval points should be among the high-spread points.
        self.assertGreater(
            np.mean(width[spread > np.median(spread)]),
            np.mean(width[spread <= np.median(spread)]),
        )


class TestCalibrationDiagnostics(unittest.TestCase):
    def test_coverage_quality_perfect(self):
        percentiles = np.arange(0, 101, 5)
        self.assertEqual(coverage_quality(percentiles, percentiles), 0.0)

    def test_coverage_quality_offset(self):
        percentiles = np.arange(0, 101, 5)
        self.assertAlmostEqual(
            coverage_quality(percentiles, percentiles + 3.0), 3.0
        )

    def test_diagnose_well_calibrated(self):
        percentiles = np.arange(0, 101, 5)
        result = diagnose_coverage_shape(percentiles, percentiles + 1.0)
        self.assertEqual(result["diagnosis"], "well_calibrated")

    def test_diagnose_underdispersed(self):
        percentiles = np.arange(0, 101, 5)
        coverage_results = np.maximum(percentiles - 20.0, 0.0)
        result = diagnose_coverage_shape(percentiles, coverage_results)
        self.assertEqual(result["diagnosis"], "underdispersed")
        self.assertLess(result["mean_bias"], 0)

    def test_diagnose_overdispersed(self):
        percentiles = np.arange(0, 101, 5)
        coverage_results = np.minimum(percentiles + 20.0, 100.0)
        result = diagnose_coverage_shape(percentiles, coverage_results)
        self.assertEqual(result["diagnosis"], "overdispersed")
        self.assertGreater(result["mean_bias"], 0)

    def test_diagnostics_on_bmc_evaluate_output(self):
        df = make_heteroscedastic_data()
        bmc = BayesianModelCombination(
            models_list=["model1", "model2", "model3", "model4"],
            data_dict={"target": df},
            truth_column_name="truth",
            error_model="hetero_model_var",
        )
        bmc.orthogonalize("target", df.iloc[:40], components_kept=2)
        bmc.train(training_options={"iterations": 300, "burn": 100})
        coverage_results = bmc.evaluate()
        score = coverage_quality(np.arange(0, 101, 5), coverage_results)
        self.assertGreaterEqual(score, 0.0)
        diag = diagnose_coverage_shape(np.arange(0, 101, 5), coverage_results)
        self.assertIn(
            diag["diagnosis"],
            {"well_calibrated", "underdispersed", "overdispersed", "mixed"},
        )


class TestMACEAndReducedChiSquare(unittest.TestCase):
    def test_mace_all_points_at_predicted_value(self):
        # rndm_m constant at 0, y_true == 0 everywhere: every point is
        # "at or below" every predicted quantile, so empirical coverage
        # is 100% regardless of the nominal quantile level.
        rndm_m = np.zeros((10, 4))
        y_true = np.zeros(4)
        self.assertAlmostEqual(mace(rndm_m, y_true), 50.0)

    def test_mace_custom_quantile_levels(self):
        rndm_m = np.zeros((10, 4))
        y_true = np.zeros(4)
        score = mace(rndm_m, y_true, quantile_levels=[50])
        self.assertAlmostEqual(score, 50.0)

    def test_mace_well_calibrated_is_small(self):
        rng = np.random.default_rng(0)
        n_points = 2000
        rndm_m = rng.normal(size=(5000, n_points))
        y_true = rng.normal(size=n_points)
        self.assertLess(mace(rndm_m, y_true), 3.0)

    def test_mace_overconfident_is_large(self):
        rng = np.random.default_rng(0)
        n_points = 2000
        rndm_m = rng.normal(scale=0.1, size=(5000, n_points))
        y_true = rng.normal(scale=1.0, size=n_points)
        self.assertGreater(mace(rndm_m, y_true), 20.0)

    def test_reduced_chi_square_well_calibrated(self):
        rng = np.random.default_rng(1)
        n_points = 5000
        pred_mean = rng.normal(size=n_points)
        rndm_m = pred_mean[None, :] + rng.normal(size=(2000, n_points))
        y_true = pred_mean + rng.normal(size=n_points)
        self.assertAlmostEqual(reduced_chi_square(rndm_m, y_true), 1.0, delta=0.15)

    def test_reduced_chi_square_overconfident_exceeds_one(self):
        rng = np.random.default_rng(2)
        n_points = 2000
        rndm_m = rng.normal(scale=0.1, size=(2000, n_points))
        y_true = rng.normal(scale=1.0, size=n_points)
        self.assertGreater(reduced_chi_square(rndm_m, y_true), 5.0)

    def test_reduced_chi_square_zero_variance_raises(self):
        rndm_m = np.ones((10, 3))
        y_true = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            reduced_chi_square(rndm_m, y_true)


if __name__ == "__main__":
    unittest.main()
