from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import gpflow
import numpy as np
from scipy.special import expit

from lstm_cpd.cpd.gp_kernels import (
    build_baseline_model,
    build_changepoint_model,
    build_local_time_index,
    extract_baseline_hyperparameters,
    extract_changepoint_location,
    extract_changepoint_steepness,
    reset_changepoint_model_for_retry,
)
from lstm_cpd.cpd.precompute_contract import (
    CPDWindowInput,
    CPDWindowResult,
    STATUS_BASELINE_FAILURE,
    STATUS_CHANGEPOINT_FAILURE,
    STATUS_FALLBACK_PREVIOUS,
    STATUS_INVALID_WINDOW,
    STATUS_RETRY_SUCCESS,
    STATUS_SUCCESS,
    is_allowed_lbw,
    validate_previous_outputs,
)


WINDOW_STANDARDIZATION_DDOF = 0


@dataclass(frozen=True)
class StandardizedReturnWindow:
    values: np.ndarray
    mean: float
    variance: float


def standardize_return_window(
    window_returns: Sequence[float],
    *,
    ddof: int = WINDOW_STANDARDIZATION_DDOF,
) -> StandardizedReturnWindow:
    values = np.asarray(window_returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("window returns must be one-dimensional")
    if values.size == 0:
        raise ValueError("window returns must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("window returns must be finite")

    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=ddof))
    if not math.isfinite(mean):
        raise ValueError("window mean must be finite")
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("window variance must be finite and positive")

    standardized = (values - mean) / math.sqrt(variance)
    return StandardizedReturnWindow(
        values=standardized.astype(np.float64),
        mean=mean,
        variance=variance,
    )


def optimize_model(model: gpflow.models.GPR) -> None:
    optimizer = gpflow.optimizers.Scipy()
    result = optimizer.minimize(
        model.training_loss,
        model.trainable_variables,
        method="L-BFGS-B",
    )
    if not bool(result.success):
        raise RuntimeError(str(result.message))


def compute_nlml(model: gpflow.models.GPR) -> float:
    nlml = float(model.training_loss().numpy())
    if not math.isfinite(nlml):
        raise RuntimeError("negative log marginal likelihood is not finite")
    return nlml


def compute_severity_score(
    nlml_baseline: float,
    nlml_changepoint: float,
) -> float:
    severity = float(expit(nlml_changepoint - nlml_baseline))
    if not math.isfinite(severity):
        raise RuntimeError("severity score is not finite")
    return severity


def compute_gamma_from_location(
    location_c: float,
    lbw: int,
) -> float:
    if lbw <= 0:
        raise ValueError(f"lbw must be positive: {lbw}")
    gamma = float(location_c / float(lbw))
    if not math.isfinite(gamma):
        raise RuntimeError("gamma is not finite")
    return gamma


def _invalid_window_result(
    window_input: CPDWindowInput,
    *,
    failure_stage: str,
    failure_message: str,
) -> CPDWindowResult:
    return CPDWindowResult(
        status=STATUS_INVALID_WINDOW,
        lbw=window_input.lbw,
        window_size=len(window_input.window_returns),
        nu=None,
        gamma=None,
        nlml_baseline=None,
        nlml_changepoint=None,
        retry_used=False,
        fallback_used=False,
        location_c=None,
        steepness_s=None,
        failure_stage=failure_stage,
        failure_message=failure_message,
    )


def _baseline_failure_result(
    window_input: CPDWindowInput,
    *,
    failure_message: str,
) -> CPDWindowResult:
    return CPDWindowResult(
        status=STATUS_BASELINE_FAILURE,
        lbw=window_input.lbw,
        window_size=len(window_input.window_returns),
        nu=None,
        gamma=None,
        nlml_baseline=None,
        nlml_changepoint=None,
        retry_used=False,
        fallback_used=False,
        location_c=None,
        steepness_s=None,
        failure_stage="baseline_fit",
        failure_message=failure_message,
    )


def _changepoint_failure_result(
    window_input: CPDWindowInput,
    *,
    nlml_baseline: float,
    retry_used: bool,
    failure_stage: str,
    failure_message: str,
) -> CPDWindowResult:
    return CPDWindowResult(
        status=STATUS_CHANGEPOINT_FAILURE,
        lbw=window_input.lbw,
        window_size=len(window_input.window_returns),
        nu=None,
        gamma=None,
        nlml_baseline=nlml_baseline,
        nlml_changepoint=None,
        retry_used=retry_used,
        fallback_used=False,
        location_c=None,
        steepness_s=None,
        failure_stage=failure_stage,
        failure_message=failure_message,
    )


def _success_result(
    window_input: CPDWindowInput,
    *,
    status: str,
    retry_used: bool,
    nlml_baseline: float,
    nlml_changepoint: float,
    location_c: float,
    steepness_s: float,
    nu: float,
    gamma: float,
) -> CPDWindowResult:
    return CPDWindowResult(
        status=status,
        lbw=window_input.lbw,
        window_size=len(window_input.window_returns),
        nu=nu,
        gamma=gamma,
        nlml_baseline=nlml_baseline,
        nlml_changepoint=nlml_changepoint,
        retry_used=retry_used,
        fallback_used=False,
        location_c=location_c,
        steepness_s=steepness_s,
        failure_stage=None,
        failure_message=None,
    )


def _fallback_result(
    window_input: CPDWindowInput,
    *,
    nlml_baseline: float,
    failure_message: str,
) -> CPDWindowResult:
    previous_outputs = window_input.previous_outputs
    if previous_outputs is None:
        raise ValueError("previous outputs required for fallback result")
    return CPDWindowResult(
        status=STATUS_FALLBACK_PREVIOUS,
        lbw=window_input.lbw,
        window_size=len(window_input.window_returns),
        nu=previous_outputs.nu,
        gamma=previous_outputs.gamma,
        nlml_baseline=nlml_baseline,
        nlml_changepoint=None,
        retry_used=True,
        fallback_used=True,
        location_c=None,
        steepness_s=None,
        failure_stage="changepoint_fit",
        failure_message=failure_message,
    )


def _extract_success_metrics(
    changepoint_model: gpflow.models.GPR,
    *,
    lbw: int,
    nlml_baseline: float,
) -> tuple[float, float, float, float]:
    nlml_changepoint = compute_nlml(changepoint_model)
    location_c = extract_changepoint_location(changepoint_model)
    steepness_s = extract_changepoint_steepness(changepoint_model)
    gamma = compute_gamma_from_location(location_c, lbw)
    if not 0.0 <= gamma <= 1.0:
        raise RuntimeError(f"gamma must lie in [0, 1], got {gamma}")
    if not math.isfinite(steepness_s) or steepness_s <= 0.0:
        raise RuntimeError(f"steepness must be finite and positive, got {steepness_s}")
    nu = compute_severity_score(nlml_baseline, nlml_changepoint)
    return nlml_changepoint, location_c, steepness_s, nu


def fit_cpd_window(window_input: CPDWindowInput) -> CPDWindowResult:
    if not is_allowed_lbw(window_input.lbw):
        return _invalid_window_result(
            window_input,
            failure_stage="lbw",
            failure_message=f"Unsupported lbw: {window_input.lbw}",
        )

    try:
        validate_previous_outputs(window_input.previous_outputs)
    except ValueError as exc:
        return _invalid_window_result(
            window_input,
            failure_stage="previous_outputs",
            failure_message=str(exc),
        )

    expected_window_size = window_input.lbw + 1
    if len(window_input.window_returns) != expected_window_size:
        return _invalid_window_result(
            window_input,
            failure_stage="window_length",
            failure_message=(
                f"Expected {expected_window_size} returns for lbw={window_input.lbw}, "
                f"got {len(window_input.window_returns)}"
            ),
        )

    try:
        standardized_window = standardize_return_window(window_input.window_returns)
    except ValueError as exc:
        return _invalid_window_result(
            window_input,
            failure_stage="window_standardization",
            failure_message=str(exc),
        )

    x_values = build_local_time_index(window_input.lbw)
    try:
        baseline_model = build_baseline_model(x_values, standardized_window.values)
        optimize_model(baseline_model)
        nlml_baseline = compute_nlml(baseline_model)
        baseline_hyperparameters = extract_baseline_hyperparameters(baseline_model)
    except Exception as exc:
        return _baseline_failure_result(
            window_input,
            failure_message=str(exc),
        )

    changepoint_model = build_changepoint_model(
        x_values,
        standardized_window.values,
        lbw=window_input.lbw,
        baseline_hyperparameters=baseline_hyperparameters,
    )

    try:
        optimize_model(changepoint_model)
        nlml_changepoint, location_c, steepness_s, nu = _extract_success_metrics(
            changepoint_model,
            lbw=window_input.lbw,
            nlml_baseline=nlml_baseline,
        )
        gamma = compute_gamma_from_location(location_c, window_input.lbw)
        return _success_result(
            window_input,
            status=STATUS_SUCCESS,
            retry_used=False,
            nlml_baseline=nlml_baseline,
            nlml_changepoint=nlml_changepoint,
            location_c=location_c,
            steepness_s=steepness_s,
            nu=nu,
            gamma=gamma,
        )
    except Exception as first_exc:
        reset_changepoint_model_for_retry(
            changepoint_model,
            lbw=window_input.lbw,
        )
        try:
            optimize_model(changepoint_model)
            nlml_changepoint, location_c, steepness_s, nu = _extract_success_metrics(
                changepoint_model,
                lbw=window_input.lbw,
                nlml_baseline=nlml_baseline,
            )
            gamma = compute_gamma_from_location(location_c, window_input.lbw)
            return _success_result(
                window_input,
                status=STATUS_RETRY_SUCCESS,
                retry_used=True,
                nlml_baseline=nlml_baseline,
                nlml_changepoint=nlml_changepoint,
                location_c=location_c,
                steepness_s=steepness_s,
                nu=nu,
                gamma=gamma,
            )
        except Exception as second_exc:
            failure_message = (
                f"First changepoint fit failed: {first_exc}; "
                f"second changepoint fit failed: {second_exc}"
            )
            if window_input.previous_outputs is not None:
                return _fallback_result(
                    window_input,
                    nlml_baseline=nlml_baseline,
                    failure_message=failure_message,
                )
            return _changepoint_failure_result(
                window_input,
                nlml_baseline=nlml_baseline,
                retry_used=True,
                failure_stage="fallback_unavailable",
                failure_message=failure_message,
            )
