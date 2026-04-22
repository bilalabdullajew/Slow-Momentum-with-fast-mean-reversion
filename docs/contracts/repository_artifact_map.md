# Repository Artifact Map

## Purpose

This document is the binding artifact-location map produced by `T-04`.

It assigns a concrete path pattern, owner task, and upstream dependency boundary to every downstream artifact currently defined by the live Asana task set.

Project root:

`Dev/Research/Slow Momentum with fast mean reversion/`

Repo-level upstream source documents outside the project root:

- `data/FTMO Data/ftmo_assets_nach_kategorie.md`
- `data/FTMO Data/FTMO Data_struktur.md`

## Structural directories created in T-04

These directories are part of the frozen repository skeleton:

- `docs/contracts/`
- `docs/reports/`
- `artifacts/manifests/`
- `artifacts/reports/`
- `artifacts/canonical_daily_close/`
- `artifacts/features/base/`
- `artifacts/features/cpd/`
- `artifacts/datasets/`
- `artifacts/training/`
- `artifacts/inference/`
- `artifacts/evaluation/`
- `src/lstm_cpd/`
- `notebooks/`
- `tests/`
- `config/`
- `logs/`

Note:

- `artifacts/reports/` is created as a derived container because later live-Asana tasks publish report artifacts there even though the literal `T-04` skeleton list omits it.

## Artifact ownership map

| Path pattern | Owner task | Upstream dependency | Notes |
| --- | --- | --- | --- |
| `docs/contracts/invariant_ledger.md` and `docs/contracts/exclusions_ledger.md` | T-01 | None | Phase-01 authority ledgers |
| `docs/contracts/unresolved_execution_items.md` | T-02 | T-01 | Reports unresolved execution-policy items only |
| `docs/contracts/execution_policy_rules.md` | T-03 | T-02 | Freezes sampling, tie-breaking, seed, and batching rules |
| `docs/contracts/repository_artifact_map.md`, `docs/contracts/config_blueprint.md`, `docs/contracts/run_manifest_schema.md`, `requirements.txt` | T-04 | T-03 | Structure-only outputs; no implementation logic |
| `artifacts/manifests/ftmo_asset_universe.json` and `artifacts/manifests/ftmo_asset_universe.csv` | T-05 | G-01 | Structured FTMO asset universe |
| `artifacts/manifests/d_timeframe_path_manifest.json`, `docs/contracts/daily_close_schema_contract.md`, and `artifacts/reports/schema_inspection_report.csv` | T-06 | T-05 | Canonical D/close resolution contract |
| `artifacts/reports/asset_eligibility_report.csv`, `artifacts/reports/asset_exclusion_report.csv`, and `artifacts/reports/minimum_history_screening_report.csv` | T-07 | T-06 | Raw availability and 318-observation sufficiency screen |
| `artifacts/canonical_daily_close/<asset_id>.csv` and `artifacts/manifests/canonical_daily_close_manifest.json` | T-08 | T-07 | Canonical daily-close store |
| `src/lstm_cpd/features/returns.py`, `src/lstm_cpd/features/volatility.py`, and `artifacts/features/base/<asset_id>_returns_volatility.csv` | T-09 | G-02 | Primitive return and sigma series |
| `src/lstm_cpd/features/normalized_returns.py` and `artifacts/features/base/<asset_id>_normalized_returns.csv` | T-10 | T-09 | Five normalized return features |
| `src/lstm_cpd/features/macd.py` and `artifacts/features/base/<asset_id>_macd_features.csv` | T-11 | T-10 | Three normalized MACD features |
| `src/lstm_cpd/features/winsorize.py`, `artifacts/features/base/<asset_id>_base_features.csv`, and `artifacts/reports/feature_provenance_report.md` | T-12 | T-10 and T-11 | Final eight non-CPD features plus provenance report |
| `src/lstm_cpd/cpd/gp_kernels.py`, `src/lstm_cpd/cpd/fit_window.py`, `src/lstm_cpd/cpd/precompute_contract.py`, and `docs/contracts/cpd_engine_contract.md` | T-13 | G-03 | Spec-faithful CPD engine |
| `artifacts/features/cpd/lbw_10/<asset_id>_cpd.csv`, `artifacts/features/cpd/lbw_21/<asset_id>_cpd.csv`, `artifacts/features/cpd/lbw_63/<asset_id>_cpd.csv`, `artifacts/features/cpd/lbw_126/<asset_id>_cpd.csv`, and `artifacts/features/cpd/lbw_252/<asset_id>_cpd.csv` | T-14 | T-13 | Per-LBW CPD feature stores |
| `artifacts/reports/cpd_fit_telemetry.csv`, `artifacts/reports/cpd_failure_ledger.csv`, `artifacts/reports/cpd_fallback_ledger.csv`, and `artifacts/manifests/cpd_feature_store_manifest.json` | T-15 | T-14 | CPD audit and join manifest |
| `artifacts/datasets/lbw_<lbw>/joined_features/<asset_id>.csv` and `artifacts/datasets/lbw_<lbw>/split_manifest.csv` | T-16 | G-04 | Asset-local chronological joins and splits |
| `artifacts/datasets/lbw_<lbw>/sequence_manifest.csv`, `artifacts/datasets/lbw_<lbw>/target_alignment_registry.csv`, `artifacts/datasets/lbw_<lbw>/discarded_fragments_report.csv`, and `artifacts/datasets/lbw_<lbw>/gap_exclusion_report.csv` | T-17 | T-16 | Sequence construction and target alignment |
| `artifacts/datasets/lbw_<lbw>/train_inputs.npy`, `artifacts/datasets/lbw_<lbw>/train_target_scale.npy`, `artifacts/datasets/lbw_<lbw>/val_inputs.npy`, `artifacts/datasets/lbw_<lbw>/val_target_scale.npy`, `artifacts/datasets/lbw_<lbw>/train_sequence_index.csv`, `artifacts/datasets/lbw_<lbw>/val_sequence_index.csv`, and `artifacts/manifests/dataset_registry.json` | T-18 | T-17 | Batch-ready dataset contract |
| `src/lstm_cpd/model/network.py`, `src/lstm_cpd/training/losses.py`, and `docs/contracts/model_runtime_contract.md` | T-19 | G-05 | Shared LSTM DMN runtime |
| `src/lstm_cpd/training/train_candidate.py` and `artifacts/training/training_runner_contract.md` | T-20 | T-19 | Deterministic single-candidate trainer |
| `artifacts/training/smoke_run/smoke_config.json`, `artifacts/training/smoke_run/smoke_best_model.keras`, `artifacts/training/smoke_run/smoke_epoch_log.csv`, `artifacts/training/smoke_run/smoke_validation_history.csv`, and `artifacts/reports/model_fidelity_report.md` | T-21 | T-20 | Smoke-train fidelity validation |

## Path policy

- All project-owned outputs are expressed relative to the project root above.
- Placeholder segments such as `<asset_id>` and `<lbw>` are part of the stable path contract and must not be renamed downstream without updating this map.
- Contracts belong under `docs/contracts/`.
- Generated manifests belong under `artifacts/manifests/`.
- Generated reports belong under `artifacts/reports/` unless a later task explicitly says otherwise.
- Runtime code belongs under `src/lstm_cpd/`.

## Ownership rule

No downstream task may write an artifact outside its declared path family unless a later human-approved contract update changes this map first.
