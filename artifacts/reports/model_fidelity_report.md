# Model Fidelity Report

Contract reference: `/Users/bilalabdullajew/Arbeit/tradinglab/quantitive/Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/model_runtime_contract.md`
Dataset registry: `/Users/bilalabdullajew/Arbeit/tradinglab/quantitive/Dev/Research/Slow Momentum with fast mean reversion/artifacts/manifests/dataset_registry.json`
Best model checkpoint: `/Users/bilalabdullajew/Arbeit/tradinglab/quantitive/Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/smoke_run/smoke_best_model.keras`

## Smoke Candidate

- candidate_id: `SMOKE`
- candidate_index: `0`
- lbw: `63`
- dropout: `0.1`
- hidden_size: `20`
- minibatch_size: `64`
- learning_rate: `0.001`
- max_grad_norm: `1.0`

## Architecture Verification

- PASS: input tensor shape is `(None, 63, 10)` and maps to `[batch, 63, 10]`.
- PASS: output tensor shape is `(None, 63, 1)` and maps to `[batch, 63, 1]`.
- PASS: exactly one LSTM layer is present: `shared_lstm`.
- PASS: LSTM is stateless=`False`, go_backwards=`False`, recurrent_dropout=`0.0`.
- PASS: dropout layers are `['input_dropout', 'output_dropout']` with noise_shape `{'input_dropout': (None, 1, 10), 'output_dropout': (None, 1, 20)}`.
- PASS: no extra dense hidden layer exists between the LSTM output and the final time-distributed tanh head.
- PASS: no extra recurrent layer exists.

## Loss Wiring Verification

- PASS: Sharpe loss is computed from model positions multiplied by the dataset target-scale tensor.
- PASS: dataset feature columns are `['normalized_return_1', 'normalized_return_21', 'normalized_return_63', 'normalized_return_126', 'normalized_return_256', 'macd_8_24', 'macd_16_28', 'macd_32_96', 'nu', 'gamma']`.

## Smoke Run Outcome

- initial_validation_loss: `-0.012358653359`
- best_validation_loss: `-0.451431155205`
- best_epoch_index: `14`
- epochs_completed: `40`
- PASS: validation loss decreased during the smoke run.
