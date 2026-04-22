from __future__ import annotations

from dataclasses import dataclass

import gpflow
import numpy as np
import tensorflow_probability as tfp


BASELINE_INITIAL_PARAMETER_VALUE = 1.0
CHANGEPOINT_INITIAL_STEEPNESS = 1.0
CHANGEPOINT_LOCATION_EPSILON = 1e-6


@dataclass(frozen=True)
class BaselineHyperparameters:
    lengthscale: float
    variance: float
    noise_variance: float


def build_local_time_index(lbw: int) -> np.ndarray:
    if lbw <= 0:
        raise ValueError(f"lbw must be positive: {lbw}")
    return np.arange(lbw + 1, dtype=np.float64).reshape(-1, 1)


def build_training_targets(standardized_returns: np.ndarray) -> np.ndarray:
    targets = np.asarray(standardized_returns, dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("standardized_returns must be one-dimensional")
    return targets.reshape(-1, 1)


def build_baseline_model(
    x_values: np.ndarray,
    standardized_returns: np.ndarray,
) -> gpflow.models.GPR:
    kernel = gpflow.kernels.Matern32(
        variance=BASELINE_INITIAL_PARAMETER_VALUE,
        lengthscales=BASELINE_INITIAL_PARAMETER_VALUE,
    )
    return gpflow.models.GPR(
        data=(x_values, build_training_targets(standardized_returns)),
        kernel=kernel,
        noise_variance=BASELINE_INITIAL_PARAMETER_VALUE,
    )


def extract_baseline_hyperparameters(model: gpflow.models.GPR) -> BaselineHyperparameters:
    return BaselineHyperparameters(
        lengthscale=float(model.kernel.lengthscales.numpy()),
        variance=float(model.kernel.variance.numpy()),
        noise_variance=float(model.likelihood.variance.numpy()),
    )


def make_changepoint_location_parameter(
    lbw: int,
    *,
    location: float | None = None,
    epsilon: float = CHANGEPOINT_LOCATION_EPSILON,
) -> gpflow.Parameter:
    if lbw <= 0:
        raise ValueError(f"lbw must be positive: {lbw}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive: {epsilon}")
    upper_bound = float(lbw) - epsilon
    if upper_bound <= epsilon:
        raise ValueError(f"epsilon too large for lbw={lbw}: {epsilon}")
    midpoint = float(lbw) / 2.0 if location is None else float(location)
    low = np.array([epsilon], dtype=np.float64)
    high = np.array([upper_bound], dtype=np.float64)
    bijector = tfp.bijectors.Sigmoid(low=low, high=high)
    return gpflow.Parameter(
        np.array([midpoint], dtype=np.float64),
        transform=bijector,
    )


def build_changepoint_model(
    x_values: np.ndarray,
    standardized_returns: np.ndarray,
    *,
    lbw: int,
    baseline_hyperparameters: BaselineHyperparameters,
) -> gpflow.models.GPR:
    midpoint = np.array([float(lbw) / 2.0], dtype=np.float64)
    pre_kernel = gpflow.kernels.Matern32(
        variance=baseline_hyperparameters.variance,
        lengthscales=baseline_hyperparameters.lengthscale,
    )
    post_kernel = gpflow.kernels.Matern32(
        variance=baseline_hyperparameters.variance,
        lengthscales=baseline_hyperparameters.lengthscale,
    )
    kernel = gpflow.kernels.ChangePoints(
        kernels=[pre_kernel, post_kernel],
        locations=midpoint,
        steepness=gpflow.Parameter(
            CHANGEPOINT_INITIAL_STEEPNESS,
            transform=gpflow.utilities.positive(),
        ),
    )
    kernel.locations = make_changepoint_location_parameter(
        lbw,
        location=float(midpoint[0]),
    )
    return gpflow.models.GPR(
        data=(x_values, build_training_targets(standardized_returns)),
        kernel=kernel,
        noise_variance=baseline_hyperparameters.noise_variance,
    )


def reset_changepoint_model_for_retry(
    model: gpflow.models.GPR,
    *,
    lbw: int,
) -> None:
    for kernel in model.kernel.kernels:
        kernel.lengthscales.assign(BASELINE_INITIAL_PARAMETER_VALUE)
        kernel.variance.assign(BASELINE_INITIAL_PARAMETER_VALUE)
    model.likelihood.variance.assign(BASELINE_INITIAL_PARAMETER_VALUE)
    model.kernel.locations.assign(np.array([float(lbw) / 2.0], dtype=np.float64))
    model.kernel.steepness.assign(CHANGEPOINT_INITIAL_STEEPNESS)


def extract_changepoint_location(model: gpflow.models.GPR) -> float:
    return float(model.kernel.locations.numpy()[0])


def extract_changepoint_steepness(model: gpflow.models.GPR) -> float:
    return float(model.kernel.steepness.numpy())
