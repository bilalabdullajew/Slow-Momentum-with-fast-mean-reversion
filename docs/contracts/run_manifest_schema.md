# Run Manifest Schema

## Purpose

This document is the stable run-manifest schema produced by `T-04`.

Its role is to standardize how later runs record run identifiers, seeds, selected LBW, candidate IDs, and artifact locations.

## Serialization format

- Preferred on-disk format: JSON
- Encoding: UTF-8
- Paths: stored as project-relative paths unless the artifact lives outside the project root

## Required top-level fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `run_id` | string | Yes | Stable unique identifier for the run |
| `run_type` | string | Yes | One of `candidate`, `smoke`, `search_summary`, `inference`, or `evaluation` |
| `created_at_utc` | string | Yes | ISO-8601 UTC timestamp |
| `project_root` | string | Yes | Project root path |
| `authority_documents` | object | Yes | Contract and spec documents the run is bound to |
| `seed_policy` | object | Yes | Effective seeds and seed formulas used by the run |
| `selected_lbw` | integer or null | Yes | The LBW used by this run, if applicable |
| `candidate_ids` | array[string] | Yes | Candidate IDs covered by the run |
| `selected_candidate_id` | string or null | Yes | Winning candidate ID when a selection exists |
| `artifact_locations` | object | Yes | Concrete artifact paths produced or consumed by the run |
| `source_artifacts` | object | Yes | Input manifests and datasets used by the run |
| `status` | string | Yes | One of `planned`, `running`, `completed`, `failed`, or `aborted` |

## Field details

### `authority_documents`

Minimum keys:

- `spec`
- `implementation_plan`
- `project_overview`
- `invariant_ledger`
- `exclusions_ledger`
- `execution_policy_rules`

### `seed_policy`

Minimum keys:

- `global_seed`
- `epoch_seed_formula`
- `candidate_seed_formula`
- `candidate_index_basis`

Recommended additional keys:

- `sampled_schedule_seed`
- `sampled_schedule_hash`

### `artifact_locations`

This object records concrete paths, not abstract categories.

Recommended keys:

- `dataset_registry`
- `joined_feature_manifest`
- `split_manifest`
- `sequence_manifest`
- `target_alignment_registry`
- `checkpoint`
- `epoch_log`
- `validation_history`
- `report_paths`

### `source_artifacts`

Recommended keys:

- `ftmo_asset_universe_manifest`
- `d_timeframe_path_manifest`
- `canonical_daily_close_manifest`
- `cpd_feature_store_manifest`
- `dataset_registry`

## Example schema instance

```json
{
  "run_id": "smoke_lbw63_20260421_0001",
  "run_type": "smoke",
  "created_at_utc": "2026-04-21T18:00:00Z",
  "project_root": "Dev/Research/Slow Momentum with fast mean reversion",
  "authority_documents": {
    "spec": "spec_lstm_cpd_model_revised_sole_authority.md",
    "implementation_plan": "lstm_cpd_implementation_plan.md",
    "project_overview": "project_overview_slow_momentum_with_fast_reversion.md",
    "invariant_ledger": "docs/contracts/invariant_ledger.md",
    "exclusions_ledger": "docs/contracts/exclusions_ledger.md",
    "execution_policy_rules": "docs/contracts/execution_policy_rules.md"
  },
  "seed_policy": {
    "global_seed": 20260421,
    "epoch_seed_formula": "20260421 + epoch_index",
    "candidate_seed_formula": "20260421 + candidate_index",
    "candidate_index_basis": "zero_based_immutable_sample_schedule",
    "sampled_schedule_seed": 20260421,
    "sampled_schedule_hash": "sha256:<full_grid_sample_hash>"
  },
  "selected_lbw": 63,
  "candidate_ids": ["candidate_0007"],
  "selected_candidate_id": "candidate_0007",
  "artifact_locations": {
    "dataset_registry": "artifacts/manifests/dataset_registry.json",
    "checkpoint": "artifacts/training/smoke_run/smoke_best_model.keras",
    "epoch_log": "artifacts/training/smoke_run/smoke_epoch_log.csv",
    "validation_history": "artifacts/training/smoke_run/smoke_validation_history.csv",
    "report_paths": [
      "artifacts/reports/model_fidelity_report.md"
    ]
  },
  "source_artifacts": {
    "ftmo_asset_universe_manifest": "artifacts/manifests/ftmo_asset_universe.json",
    "canonical_daily_close_manifest": "artifacts/manifests/canonical_daily_close_manifest.json",
    "cpd_feature_store_manifest": "artifacts/manifests/cpd_feature_store_manifest.json",
    "dataset_registry": "artifacts/manifests/dataset_registry.json"
  },
  "status": "completed"
}
```

## Schema rules

- `run_id` must be unique within the project.
- `candidate_ids` must be non-empty for `candidate`, `smoke`, and `search_summary` runs.
- `selected_lbw` must be one of `{10, 21, 63, 126, 252}` when present.
- `artifact_locations` must include every artifact produced by the run that a downstream task would otherwise have to rediscover.
- A failed run still writes a manifest with `status = failed` and records every artifact that was successfully materialized before failure.
