# Training Runner Contract

## Purpose

This contract freezes the `T-20` single-candidate training runner.

## Entrypoint

- Module: `src/lstm_cpd/training/train_candidate.py`
- CLI inputs:
  - `--dataset-registry`
  - `--candidate-config`
  - `--output-dir`
  - `--project-root`
- Default dataset registry path remains the official project path:
  - `artifacts/manifests/dataset_registry.json`
- The dataset registry path is overrideable so pre-`G-05` smoke runs can use:
  - `artifacts/interim/manifests/dataset_registry.json`

## Candidate config contract

The candidate config JSON must contain:

- `candidate_id`
- `candidate_index`
- `dropout`
- `hidden_size`
- `minibatch_size`
- `learning_rate`
- `max_grad_norm`
- `lbw`

## Dataset loading contract

- One registry entry is selected by `lbw`.
- The runner loads:
  - `train_inputs.npy`
  - `train_target_scale.npy`
  - `val_inputs.npy`
  - `val_target_scale.npy`
- Registry feature order must match the frozen 10-feature contract.
- The runner must not infer schema from array contents.

## Deterministic training policy

- Base seed: `20260421`
- Python `random`, NumPy, and TensorFlow seeds are set from that base seed.
- Per-epoch train-sequence shuffle seed is `20260421 + epoch_index`.
- Model-specific deterministic seeds are derived from `20260421 + candidate_index`.
- Shuffling occurs only at sequence level.
- The final smaller batch is kept.
- No asset balancing or stratification is allowed.

## Optimization policy

- Optimizer: Adam
- Maximum epochs: `300`
- Early stopping patience: `25`
- Gradient clipping: global norm clipping only
- Validation loss is evaluated every epoch and written to log artifacts.

## Persisted artifacts

For a chosen artifact stem `<stem>`, the runner writes:

- `<stem>_config.json`
- `<stem>_best_model.keras`
- `<stem>_epoch_log.csv`
- `<stem>_validation_history.csv`

The persisted checkpoint must load back cleanly with `tf.keras.models.load_model(..., compile=False)`.
