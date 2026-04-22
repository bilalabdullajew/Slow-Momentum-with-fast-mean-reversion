# Config Blueprint

## Purpose

This document is the stable configuration blueprint produced by `T-04`.

It defines the configuration groupings that later implementation tasks may rely on. The blueprint freezes grouping boundaries, not final config-file contents.

## Path semantics

- Paths under the project root are project-relative.
- FTMO source-document paths are repo-relative because they live outside the project root:
  - `data/FTMO Data/ftmo_assets_nach_kategorie.md`
  - `data/FTMO Data/FTMO Data_struktur.md`
- Execution-policy choices frozen by `docs/contracts/execution_policy_rules.md` are not to be duplicated as free-form tunables.

## Stable configuration groupings

### 1. `data_contract`

Required concerns:

- project root
- FTMO raw-data root
- FTMO asset-list document path
- FTMO structure document path
- canonical timeframe (`D`)
- canonical price field (`close`)
- accepted timestamp-column candidates
- minimum raw-history threshold

Suggested keys:

```yaml
data_contract:
  project_root: Dev/Research/Slow Momentum with fast mean reversion
  raw_ftmo_root: data/FTMO Data
  allowed_asset_document: data/FTMO Data/ftmo_assets_nach_kategorie.md
  raw_structure_document: data/FTMO Data/FTMO Data_struktur.md
  canonical_timeframe: D
  canonical_price_column: close
  accepted_timestamp_columns: [timestamp, datetime, date, time]
  minimum_raw_history_observations: 318
```

### 2. `feature_parameters`

Required concerns:

- arithmetic-return horizons
- volatility settings
- MACD pair definitions
- MACD normalization windows
- winsorization scope
- winsorization half-life and cap width

Suggested keys:

```yaml
feature_parameters:
  return_horizons: [1, 21, 63, 126, 256]
  volatility:
    family: ewm_std
    span: 60
  macd:
    pairs: [[8, 24], [16, 28], [32, 96]]
    price_std_window: 63
    signal_std_window: 252
    ewma_alpha_mode: reciprocal_period
  winsorization:
    apply_to: [normalized_returns, macd_features]
    half_life: 252
    sigma_multiple: 5.0
```

### 3. `cpd_parameters`

Required concerns:

- allowed LBWs
- rolling-window standardization rule
- baseline kernel family
- baseline initialization
- changepoint initialization
- changepoint-location constraint
- optimizer family
- retry limit
- fallback policy

Suggested keys:

```yaml
cpd_parameters:
  allowed_lbws: [10, 21, 63, 126, 252]
  window_standardization: local_zero_mean_unit_variance
  baseline_kernel: matern32
  baseline_initialization:
    lambda: 1.0
    sigma_h: 1.0
    sigma_n: 1.0
  changepoint_initialization:
    c_mode: window_midpoint
    s: 1.0
    clone_baseline_kernel_to_pre_post: true
  changepoint_location_constraint: open_interval
  optimizer: gpflow_scipy_lbfgsb
  retry_limit_after_first_failure: 1
  fallback_policy: previous_nu_gamma
```

### 4. `training_search_parameters`

Required concerns:

- sequence length
- target volatility
- optimizer family
- epoch budget
- patience
- search-iteration budget
- search grid
- selection metric
- gradient-clipping mode
- reference to the frozen execution-policy document

Suggested keys:

```yaml
training_search_parameters:
  sequence_length: 63
  target_volatility: 0.15
  optimizer: adam
  max_epochs: 300
  early_stopping_patience: 25
  search_iterations: 50
  search_grid:
    dropout: [0.1, 0.2, 0.3, 0.4, 0.5]
    hidden_size: [5, 10, 20, 40, 80, 160]
    minibatch_size: [64, 128, 256]
    learning_rate: [1e-4, 1e-3, 1e-2, 1e-1]
    max_grad_norm: [1e-2, 1e0, 1e2]
    lbw: [10, 21, 63, 126, 252]
  selection_metric: minimum_validation_loss
  gradient_clipping: global_norm
  execution_policy_document: docs/contracts/execution_policy_rules.md
```

### 5. `runtime_paths`

Required concerns:

- manifests directory
- reports directory
- canonical store directory
- feature directories
- dataset directory
- training directory
- inference directory
- evaluation directory
- notebook directory
- logs directory

Suggested keys:

```yaml
runtime_paths:
  contracts_dir: docs/contracts
  docs_reports_dir: docs/reports
  manifests_dir: artifacts/manifests
  artifact_reports_dir: artifacts/reports
  canonical_daily_close_dir: artifacts/canonical_daily_close
  base_feature_dir: artifacts/features/base
  cpd_feature_dir: artifacts/features/cpd
  datasets_dir: artifacts/datasets
  training_dir: artifacts/training
  inference_dir: artifacts/inference
  evaluation_dir: artifacts/evaluation
  source_root: src/lstm_cpd
  notebooks_dir: notebooks
  tests_dir: tests
  config_dir: config
  logs_dir: logs
```

## Blueprint rules

- Later tasks may add files under `config/`, but they must preserve these five top-level grouping boundaries.
- Operational seeds, sampling mode, tie-breaking, and batching behavior are governed by `execution_policy_rules.md`, not by ad hoc config knobs.
- Path keys must remain stable across runs so manifests can be compared without schema inference.
