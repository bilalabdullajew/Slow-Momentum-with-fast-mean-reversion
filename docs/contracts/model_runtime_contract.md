# Model Runtime Contract

## Purpose

This contract freezes the `T-19` model runtime used by every downstream candidate run.

## Runtime shape contract

- Input tensor shape: `[batch, 63, 10]`
- Output tensor shape: `[batch, 63, 1]`
- Per-timestep feature order:
  - `normalized_return_1`
  - `normalized_return_21`
  - `normalized_return_63`
  - `normalized_return_126`
  - `normalized_return_256`
  - `macd_8_24`
  - `macd_16_28`
  - `macd_32_96`
  - `nu`
  - `gamma`

## Architecture

- Exactly one shared stateless unidirectional `LSTM` layer is used.
- The LSTM returns the full sequence.
- The final head is a direct `TimeDistributed(Dense(1, activation="tanh"))`.
- No extra dense hidden layer is allowed between the LSTM and the final output head.
- No second recurrent layer is allowed.
- No bidirectional wrapping is allowed.
- No stateful recurrent runtime is allowed.

## Dropout policy

- Dropout is applied at exactly two sites during training only:
  - on the inputs to the LSTM
  - on the LSTM sequence outputs immediately before the final dense head
- The input-site dropout uses `noise_shape=(None, 1, 10)`.
- The output-site dropout uses `noise_shape=(None, 1, H)`.
- These `noise_shape` settings enforce one sampled mask per sequence and reuse that mask across all timesteps.
- Validation and inference call the model with `training=False`.
- Recurrent dropout is fixed to `0.0`.

## Loss wiring

- The runtime does not consume class labels.
- The Sharpe loss consumes model position outputs together with the `target_scale` tensor from Phase 05.
- Realized return wiring is:

  `realized_return = position_output * target_scale`

- The scalar loss is:

  `-sqrt(252) * mean(realized_return) / sqrt(var(realized_return) + eps)`

- The loss is computed over all asset-time pairs present in the active tensor span.
