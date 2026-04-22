# CPD Engine Contract

This document is the binding output contract for `T-13`.

## Scope

- `T-13` provides the reusable single-window CPD fitting engine only.
- It does not materialize per-asset or per-LBW CPD CSVs. That remains `T-14`.
- GP fits remain upstream precomputation only and are not part of the differentiable graph.

## Window and Coordinates

- Allowed LBWs are exactly `{10, 21, 63, 126, 252}`.
- Each fit consumes exactly `lbw + 1` arithmetic returns from the rolling window `{r_t}_{t=T-l}^{T}`.
- The engine uses local coordinates `x = [0, 1, ..., lbw]`, which is the shifted equivalent of the paper window `[T-lbw, ..., T]`.
- The normalized changepoint location is therefore computed as `gamma = c / lbw`.

## Standardization

- Returns are standardized within each window to local zero mean and unit variance.
- The operational variance convention is the **population variance** (`ddof=0`).
- If a window contains non-finite values, the wrong length, or zero/non-finite variance, the engine returns `invalid_window` and does not fit a GP.

## GP Runtime

- Baseline model: GPflow `GPR` with Matérn 3/2 kernel and baseline initialization `lengthscale=1`, `variance=1`, `noise_variance=1`.
- Baseline hyperparameters are reinitialized from scratch for every window.
- Changepoint model: GPflow `ChangePoints` over two Matérn 3/2 kernels.
- First changepoint initialization:
  - `c = lbw / 2`
  - `s = 1`
  - both pre/post kernels clone the fitted baseline kernel parameters
  - observation noise clones the fitted baseline noise parameter
- The changepoint location is constrained to the open interval `(0, lbw)` using an explicit bounded parameter transform.
- Optimization uses GPflow SciPy `L-BFGS-B` with no extra generic restarts, bounds overrides, or multistart search.

## Retry and Failure Policy

- If the first changepoint fit fails, retry exactly once after reinitializing all changepoint parameters to `1`, except `c`, which stays at the window midpoint.
- If the second changepoint fit fails and previous outputs exist, reuse the previous `nu` and `gamma` exactly and return `fallback_previous`.
- If the second changepoint fit fails and previous outputs do not exist, return `changepoint_failure` with no synthetic outputs.
- If the baseline fit fails, return `baseline_failure` with no synthetic outputs.

## Output Semantics

- Severity is computed exactly as:

  `nu = 1 - 1 / (1 + exp(-(nlml_changepoint - nlml_baseline)))`

- This is numerically equivalent to `sigmoid(nlml_changepoint - nlml_baseline)`.
- The paper's `nlmn`/`nlml` notation inconsistency is treated as a paper typo only; the engine uses the negative log marginal likelihood defined earlier in the spec.
- The in-memory engine result exposes at least:
  - `status`
  - `nu`
  - `gamma`
  - `nlml_baseline`
  - `nlml_changepoint`
  - `retry_used`
  - `fallback_used`
  - `location_c`
  - `steepness_s`
  - `failure_stage`
  - `failure_message`

## Status Meanings

- `success`: first changepoint fit succeeded.
- `retry_success`: first changepoint fit failed, retry succeeded.
- `fallback_previous`: both changepoint fits failed, previous outputs were reused.
- `baseline_failure`: baseline fit failed, so no changepoint fit was trusted.
- `changepoint_failure`: both changepoint fits failed and no previous outputs were available.
- `invalid_window`: the input window itself was not fit-eligible.
