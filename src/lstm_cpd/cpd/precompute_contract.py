from __future__ import annotations

import math
from dataclasses import dataclass


ALLOWED_CPD_LBWS = (10, 21, 63, 126, 252)

STATUS_SUCCESS = "success"
STATUS_RETRY_SUCCESS = "retry_success"
STATUS_FALLBACK_PREVIOUS = "fallback_previous"
STATUS_BASELINE_FAILURE = "baseline_failure"
STATUS_CHANGEPOINT_FAILURE = "changepoint_failure"
STATUS_INVALID_WINDOW = "invalid_window"

CPD_RESULT_STATUSES = (
    STATUS_SUCCESS,
    STATUS_RETRY_SUCCESS,
    STATUS_FALLBACK_PREVIOUS,
    STATUS_BASELINE_FAILURE,
    STATUS_CHANGEPOINT_FAILURE,
    STATUS_INVALID_WINDOW,
)


@dataclass(frozen=True)
class CPDPreviousOutputs:
    nu: float
    gamma: float


@dataclass(frozen=True)
class CPDWindowInput:
    lbw: int
    window_returns: tuple[float, ...]
    window_end_timestamp: str | None = None
    previous_outputs: CPDPreviousOutputs | None = None


@dataclass(frozen=True)
class CPDWindowResult:
    status: str
    lbw: int
    window_size: int
    nu: float | None
    gamma: float | None
    nlml_baseline: float | None
    nlml_changepoint: float | None
    retry_used: bool
    fallback_used: bool
    location_c: float | None
    steepness_s: float | None
    failure_stage: str | None
    failure_message: str | None

    @property
    def has_outputs(self) -> bool:
        return self.nu is not None and self.gamma is not None


def is_allowed_lbw(lbw: int) -> bool:
    return lbw in ALLOWED_CPD_LBWS


def validate_previous_outputs(previous_outputs: CPDPreviousOutputs | None) -> None:
    if previous_outputs is None:
        return

    if not math.isfinite(previous_outputs.nu):
        raise ValueError("previous nu must be finite")
    if not math.isfinite(previous_outputs.gamma):
        raise ValueError("previous gamma must be finite")
    if not 0.0 <= previous_outputs.nu <= 1.0:
        raise ValueError("previous nu must lie in [0, 1]")
    if not 0.0 <= previous_outputs.gamma <= 1.0:
        raise ValueError("previous gamma must lie in [0, 1]")
