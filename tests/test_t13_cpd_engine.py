from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.cpd.fit_window import (  # noqa: E402
    WINDOW_STANDARDIZATION_DDOF,
    compute_severity_score,
    compute_gamma_from_location,
    fit_cpd_window,
    standardize_return_window,
)
from lstm_cpd.cpd.gp_kernels import (  # noqa: E402
    BASELINE_INITIAL_PARAMETER_VALUE,
    CHANGEPOINT_INITIAL_STEEPNESS,
    build_baseline_model,
    build_changepoint_model,
    build_local_time_index,
    extract_changepoint_location,
    extract_changepoint_steepness,
    extract_baseline_hyperparameters,
)
from lstm_cpd.cpd.precompute_contract import (  # noqa: E402
    CPDPreviousOutputs,
    CPDWindowInput,
    STATUS_BASELINE_FAILURE,
    STATUS_CHANGEPOINT_FAILURE,
    STATUS_FALLBACK_PREVIOUS,
    STATUS_INVALID_WINDOW,
    STATUS_RETRY_SUCCESS,
    STATUS_SUCCESS,
)


def make_window_returns() -> tuple[float, ...]:
    return (
        0.02,
        -0.01,
        0.03,
        0.01,
        -0.02,
        0.015,
        0.45,
        0.52,
        0.49,
        0.51,
        0.48,
    )


class T13CpdEngineTests(unittest.TestCase):
    def test_standardize_return_window_uses_population_variance(self) -> None:
        standardized_window = standardize_return_window((1.0, 3.0, 5.0))

        self.assertAlmostEqual(standardized_window.mean, 3.0)
        self.assertAlmostEqual(standardized_window.variance, np.var([1.0, 3.0, 5.0], ddof=0))
        self.assertAlmostEqual(float(np.mean(standardized_window.values)), 0.0, places=12)
        self.assertAlmostEqual(
            float(np.var(standardized_window.values, ddof=WINDOW_STANDARDIZATION_DDOF)),
            1.0,
            places=12,
        )

    def test_compute_gamma_from_location_uses_local_coordinate_ratio(self) -> None:
        self.assertAlmostEqual(compute_gamma_from_location(5.0, 10), 0.5)
        self.assertAlmostEqual(compute_gamma_from_location(2.5, 10), 0.25)

    def test_compute_severity_score_is_high_when_changepoint_fit_is_better(self) -> None:
        improved = compute_severity_score(12.0, 8.0)
        worse = compute_severity_score(8.0, 12.0)

        self.assertGreater(improved, 0.5)
        self.assertLess(worse, 0.5)
        self.assertGreater(improved, worse)

    def test_build_baseline_model_initializes_all_parameters_to_one(self) -> None:
        x_values = build_local_time_index(10)
        model = build_baseline_model(x_values, np.asarray(make_window_returns(), dtype=np.float64))

        self.assertAlmostEqual(float(model.kernel.lengthscales.numpy()), BASELINE_INITIAL_PARAMETER_VALUE)
        self.assertAlmostEqual(float(model.kernel.variance.numpy()), BASELINE_INITIAL_PARAMETER_VALUE)
        self.assertAlmostEqual(float(model.likelihood.variance.numpy()), BASELINE_INITIAL_PARAMETER_VALUE)

    def test_build_changepoint_model_clones_baseline_hyperparameters(self) -> None:
        x_values = build_local_time_index(10)
        baseline_model = build_baseline_model(
            x_values,
            np.asarray(make_window_returns(), dtype=np.float64),
        )
        baseline_model.kernel.lengthscales.assign(2.5)
        baseline_model.kernel.variance.assign(3.5)
        baseline_model.likelihood.variance.assign(4.5)
        hyperparameters = extract_baseline_hyperparameters(baseline_model)

        changepoint_model = build_changepoint_model(
            x_values,
            np.asarray(make_window_returns(), dtype=np.float64),
            lbw=10,
            baseline_hyperparameters=hyperparameters,
        )

        self.assertAlmostEqual(float(changepoint_model.kernel.kernels[0].lengthscales.numpy()), 2.5)
        self.assertAlmostEqual(float(changepoint_model.kernel.kernels[1].lengthscales.numpy()), 2.5)
        self.assertAlmostEqual(float(changepoint_model.kernel.kernels[0].variance.numpy()), 3.5)
        self.assertAlmostEqual(float(changepoint_model.kernel.kernels[1].variance.numpy()), 3.5)
        self.assertAlmostEqual(float(changepoint_model.likelihood.variance.numpy()), 4.5)
        self.assertAlmostEqual(extract_changepoint_location(changepoint_model), 5.0)
        self.assertAlmostEqual(extract_changepoint_steepness(changepoint_model), CHANGEPOINT_INITIAL_STEEPNESS)

    def test_invalid_window_rejects_zero_variance_without_fit(self) -> None:
        window_input = CPDWindowInput(
            lbw=10,
            window_returns=(0.5,) * 11,
        )

        with mock.patch("lstm_cpd.cpd.fit_window.optimize_model") as optimize_model:
            result = fit_cpd_window(window_input)

        self.assertEqual(result.status, STATUS_INVALID_WINDOW)
        self.assertEqual(result.failure_stage, "window_standardization")
        self.assertIsNone(result.nu)
        self.assertIsNone(result.gamma)
        optimize_model.assert_not_called()

    def test_baseline_failure_returns_explicit_status(self) -> None:
        window_input = CPDWindowInput(
            lbw=10,
            window_returns=make_window_returns(),
        )

        with mock.patch(
            "lstm_cpd.cpd.fit_window.optimize_model",
            side_effect=RuntimeError("baseline blew up"),
        ):
            result = fit_cpd_window(window_input)

        self.assertEqual(result.status, STATUS_BASELINE_FAILURE)
        self.assertEqual(result.failure_stage, "baseline_fit")
        self.assertFalse(result.has_outputs)

    def test_retry_success_uses_exactly_one_retry(self) -> None:
        window_input = CPDWindowInput(
            lbw=10,
            window_returns=make_window_returns(),
        )
        side_effects = [None, RuntimeError("first cp fit failed"), None]

        with mock.patch(
            "lstm_cpd.cpd.fit_window.optimize_model",
            side_effect=side_effects,
        ) as optimize_model:
            result = fit_cpd_window(window_input)

        self.assertEqual(result.status, STATUS_RETRY_SUCCESS)
        self.assertTrue(result.retry_used)
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.has_outputs)
        self.assertEqual(optimize_model.call_count, 3)

    def test_fallback_previous_reuses_prior_outputs_after_two_failures(self) -> None:
        previous_outputs = CPDPreviousOutputs(nu=0.25, gamma=0.75)
        window_input = CPDWindowInput(
            lbw=10,
            window_returns=make_window_returns(),
            previous_outputs=previous_outputs,
        )
        side_effects = [None, RuntimeError("first cp fit failed"), RuntimeError("second cp fit failed")]

        with mock.patch(
            "lstm_cpd.cpd.fit_window.optimize_model",
            side_effect=side_effects,
        ):
            result = fit_cpd_window(window_input)

        self.assertEqual(result.status, STATUS_FALLBACK_PREVIOUS)
        self.assertTrue(result.retry_used)
        self.assertTrue(result.fallback_used)
        self.assertAlmostEqual(result.nu, previous_outputs.nu)
        self.assertAlmostEqual(result.gamma, previous_outputs.gamma)
        self.assertIsNotNone(result.nlml_baseline)
        self.assertIsNone(result.nlml_changepoint)

    def test_double_changepoint_failure_without_previous_outputs_is_explicit(self) -> None:
        window_input = CPDWindowInput(
            lbw=10,
            window_returns=make_window_returns(),
        )
        side_effects = [None, RuntimeError("first cp fit failed"), RuntimeError("second cp fit failed")]

        with mock.patch(
            "lstm_cpd.cpd.fit_window.optimize_model",
            side_effect=side_effects,
        ):
            result = fit_cpd_window(window_input)

        self.assertEqual(result.status, STATUS_CHANGEPOINT_FAILURE)
        self.assertEqual(result.failure_stage, "fallback_unavailable")
        self.assertTrue(result.retry_used)
        self.assertFalse(result.fallback_used)
        self.assertFalse(result.has_outputs)

    def test_real_gpflow_smoke_fit_produces_finite_outputs(self) -> None:
        window_input = CPDWindowInput(
            lbw=10,
            window_returns=make_window_returns(),
        )

        result = fit_cpd_window(window_input)

        self.assertIn(result.status, {STATUS_SUCCESS, STATUS_RETRY_SUCCESS})
        self.assertTrue(result.has_outputs)
        self.assertTrue(math.isfinite(result.nu))
        self.assertTrue(math.isfinite(result.gamma))
        self.assertTrue(math.isfinite(result.nlml_baseline))
        self.assertTrue(math.isfinite(result.nlml_changepoint))
        self.assertGreaterEqual(result.gamma, 0.0)
        self.assertLessEqual(result.gamma, 1.0)
        self.assertGreaterEqual(result.nu, 0.0)
        self.assertLessEqual(result.nu, 1.0)


if __name__ == "__main__":
    unittest.main()
