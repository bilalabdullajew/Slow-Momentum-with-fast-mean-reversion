## Project level

**Project name:** `LSTM CPD — Replication`

**Description:** Replicate the paper’s LSTM+CPD pipeline on the project-defined FTMO universe in Python, using only daily `D`/`close` data, exact paper-faithful feature/CPD/training logic, causal online inference, and a final notebook assembled only after all implementation artifacts are frozen. fileciteturn7file10 fileciteturn7file18

**Default view:** `List`

---

## 01 — Implementation Contract Freeze

### T-01 — Compile invariant and exclusions ledgers
---
**Purpose:**
Convert the authoritative Spec and approved Plan into a concrete list of non-negotiable implementation invariants and explicit exclusions.

**Inputs:**
- `spec_lstm_cpd_model_revised_sole_authority.md`
- `lstm_cpd_implementation_plan.md`
- `project_overview_slow_momentum_with_fast_reversion.md`

**Dependencies:**
- —

**Exact action:**
Read the authoritative Spec first, then the Implementation Plan, then the Project Overview. Extract every binding requirement that affects data, features, CPD, sequence construction, model architecture, training, inference, evaluation, scope, and notebook timing. Write one implementation invariant per item in an invariant ledger. Write one excluded design per item in an exclusions ledger. Include, at minimum, the following as explicit exclusions: intraday or OHLC adaptation, external or synthetic data, hard regime labels, changepoint-threshold trading rules, multiple CPD modules in parallel, transaction-cost-adjusted loss as default, extra recurrent layers, extra dense hidden layers, bidirectionality, statefulness, and notebook-first development.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/invariant_ledger.md`
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/exclusions_ledger.md`

**Acceptance criteria:**
- Every major methodological rule from the Spec and Plan appears exactly once in either the invariant ledger or the exclusions ledger.
- The invariant ledger explicitly includes daily close-based arithmetic returns, 60-day EWM volatility, 5 normalized return features, 3 MACD features, 2 CPD features, non-overlapping 63-step sequences, one stateless unidirectional LSTM, time-distributed `tanh` output, and Sharpe-ratio training.
- The exclusions ledger explicitly blocks all out-of-scope designs listed above.
- No item in either ledger contradicts the authoritative Spec.

**Human review required:** No

**Notes for Codex CLI:**
Treat the Spec as sole methodological authority. Do not reopen closed design decisions while writing these ledgers.
---

### T-02 — Register unresolved execution-policy items
---
**Purpose:**
Surface any remaining execution-policy choices that would otherwise force Codex CLI to improvise during implementation.

**Inputs:**
- `docs/contracts/invariant_ledger.md`
- `docs/contracts/exclusions_ledger.md`
- `lstm_cpd_implementation_plan.md`

**Dependencies:**
- T-01

**Exact action:**
Create a register of all execution-policy items that are not fully fixed by the Spec but must be fixed before coding tasks begin. Record for each item: the decision topic, why it is unresolved at spec level, the downstream components it affects, and which later tasks must consume the decision. The register must contain exactly these four items unless an additional unresolved item is discovered directly from the approved documents: random-search sampling mode, tie-breaking rule for equal validation losses, random seed policy, and batching convention across assets.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/unresolved_execution_items.md`

**Acceptance criteria:**
- The register contains the four required unresolved execution-policy items.
- Each item states which later implementation tasks depend on it.
- No already-closed Spec decision is incorrectly listed as unresolved.
- The register is complete enough that T-03 can freeze every remaining execution-policy choice.

**Human review required:** No

**Notes for Codex CLI:**
This task reports unresolved items only. Do not decide them here.
---

### T-03 — Freeze execution-policy rules for Codex-safe implementation
---
**Purpose:**
Eliminate the final silent choices so every later implementation task is fully executable without methodological interpretation.

**Inputs:**
- `docs/contracts/unresolved_execution_items.md`

**Dependencies:**
- T-02

**Exact action:**
Write the project’s binding execution-policy rules as follows.  
Rule 1 — Random-search sampling mode: enumerate the full discrete Cartesian product in fixed nested-loop order `dropout -> hidden_size -> minibatch_size -> learning_rate -> max_grad_norm -> lbw`, then draw exactly 50 unique candidates without replacement using `random.Random(20260421).sample(full_grid, 50)`.  
Rule 2 — Tie-breaking: select the candidate with minimum validation loss; if multiple candidates share the exact same serialized minimum value, select the candidate with the smallest candidate index in the immutable sampled schedule.  
Rule 3 — Seed policy: set Python `random`, NumPy, and TensorFlow seeds to `20260421`; use per-epoch shuffle seed `20260421 + epoch_index`; use candidate-specific deterministic seed `20260421 + candidate_index` only where a separate deterministic stream is needed.  
Rule 4 — Batching across assets: construct sequences independently per asset and per split, concatenate them into one shared dataset per LBW while preserving `asset_id` and sequence index metadata, shuffle only at sequence level once per epoch, keep the final smaller batch, and do not perform asset balancing or stratification.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/execution_policy_rules.md`

**Acceptance criteria:**
- All four execution-policy items are explicitly frozen with deterministic rules.
- The rules are complete enough that no later training or search task needs to choose sampling, tie-breaking, seeding, or batching behavior.
- The rules do not alter any Spec-closed methodological decision.
- The document is written as binding project authority for all downstream tasks.

**Human review required:** No

**Notes for Codex CLI:**
These rules are task-stage project closures required by the approved Plan. Once written, downstream tasks must treat them as fixed.
---

### T-04 — Scaffold repository and artifact blueprint
---
**Purpose:**
Assign a concrete path and ownership boundary to every downstream artifact before implementation begins.

**Inputs:**
- `docs/contracts/invariant_ledger.md`
- `docs/contracts/exclusions_ledger.md`
- `docs/contracts/execution_policy_rules.md`

**Dependencies:**
- T-03

**Exact action:**
Create the directory skeleton under the project root for contracts, manifests, canonical data, features, CPD, datasets, training, inference, evaluation, reproducibility, notebook outputs, source modules, and tests. Write a repository artifact map that lists every downstream artifact path expected by later tasks. Write a config blueprint that defines stable config groupings for data contract, feature parameters, CPD parameters, training/search parameters, and runtime paths. Write a run-manifest schema that records run identifiers, seeds, selected LBW, candidate IDs, and artifact locations.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/repository_artifact_map.md`
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/config_blueprint.md`
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/run_manifest_schema.md`
- Created directory tree under `Dev/Research/Slow Momentum with fast mean reversion/`

**Acceptance criteria:**
- Every later task artifact in this Asana structure has a predeclared destination path.
- The repository tree exists under the required project root.
- The config blueprint and run-manifest schema cover all later phases.
- No notebook-only logic or implementation code is introduced here.

**Human review required:** No

**Notes for Codex CLI:**
This task defines structure only. Do not implement feature, CPD, training, or notebook logic yet.
---

### G-01 — Review Gate: Source fidelity and execution rule freeze
---
**Review scope:**
Invariant ledger, exclusions ledger, unresolved execution register, execution-policy rules, and repository/artifact blueprint.

**Pass conditions:**
- The Spec remains the sole methodological authority.
- All out-of-scope designs are explicitly excluded.
- The four execution-policy items are fully frozen with explicit rules.
- Every downstream artifact path is defined before coding begins.

**Blocked downstream tasks:**
- T-05
- T-06
- T-07
- T-08

**Reviewer:** Human
---

### M-01 — Implementation Contract Freeze Complete

---

## 02 — FTMO Data Contract & Canonical Daily-Close Layer

### T-05 — Build allowed FTMO asset-universe manifest
---
**Purpose:**
Convert the allowed FTMO asset list into a machine-usable universe manifest.

**Inputs:**
- `ftmo_assets_nach_kategorie.md`
- `docs/contracts/repository_artifact_map.md`

**Dependencies:**
- G-01

**Exact action:**
Parse `ftmo_assets_nach_kategorie.md` and create one record per allowed asset. Preserve the source category, source symbol, stable `asset_id`, and source ordering. Do not add, rename, normalize, merge, or remove assets beyond what is directly required to store them as structured records. Persist both JSON and CSV manifests.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/manifests/ftmo_asset_universe.json`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/manifests/ftmo_asset_universe.csv`

**Acceptance criteria:**
- Every asset from the allowed FTMO asset document appears exactly once.
- Each manifest row contains `asset_id`, `symbol`, and `category`.
- No asset absent from the source document appears in the manifest.
- JSON and CSV manifests reconcile exactly.

**Human review required:** No

**Notes for Codex CLI:**
Do not create any mapping to the paper’s original 50-futures universe here.
---

### T-06 — Resolve D-timeframe paths and daily-close schema contract
---
**Purpose:**
Define exactly how permitted FTMO raw files become admissible daily-close sources.

**Inputs:**
- `artifacts/manifests/ftmo_asset_universe.json`
- `FTMO Data_struktur.md`
- `docs/contracts/repository_artifact_map.md`

**Dependencies:**
- T-05

**Exact action:**
For each allowed asset, resolve the `D` timeframe file path implied by `FTMO Data_struktur.md` and store it in a path manifest. Then define a canonical schema contract for reading those files. Accept exactly one source timestamp column chosen case-insensitively from `{timestamp, datetime, date, time}` and map it to canonical column `timestamp`. Accept exactly one source `close` column case-insensitively and map it to canonical column `close`. Sort rows ascending by timestamp. If duplicate timestamps have identical `close` values, keep the last occurrence and log it. If duplicate timestamps disagree on `close`, exclude the asset with reason code `DUPLICATE_TIMESTAMP_CONFLICT`. Exclude any file missing a timestamp column, missing a close column, containing unparsable timestamps, or containing nonnumeric closes.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/manifests/d_timeframe_path_manifest.json`
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/daily_close_schema_contract.md`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/reports/schema_inspection_report.csv`

**Acceptance criteria:**
- Every allowed asset has either a resolved `D` path or an explicit resolution failure reason.
- The schema contract fully defines accepted columns and exclusion conditions.
- The contract refers only to `D` timeframe and `close`.
- No non-`D` timeframe or non-`close` field is admitted into the canonical layer.

**Human review required:** No

**Notes for Codex CLI:**
Ignore open, high, low, volume, and every non-`D` timeframe even if present in source files.
---

### T-07 — Screen raw availability and minimum-history sufficiency
---
**Purpose:**
Identify which assets are eligible to continue into canonical daily-close processing.

**Inputs:**
- `artifacts/manifests/d_timeframe_path_manifest.json`
- `docs/contracts/daily_close_schema_contract.md`

**Dependencies:**
- T-06

**Exact action:**
Load each resolved `D` file using the schema contract and compute raw row counts, date coverage, and raw eligibility status. Apply a minimum raw-history screening threshold of at least 318 daily observations. Produce reason-coded eligibility and exclusion reports. Use reason codes such as `MISSING_FILE`, `UNREADABLE_FILE`, `SCHEMA_FAILURE`, `EMPTY_SERIES`, `DUPLICATE_TIMESTAMP_CONFLICT`, and `INSUFFICIENT_RAW_HISTORY`.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/reports/asset_eligibility_report.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/reports/asset_exclusion_report.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/reports/minimum_history_screening_report.csv`

**Acceptance criteria:**
- Every allowed asset is assigned either eligible or excluded status.
- Every excluded asset has exactly one explicit reason code.
- Every eligible asset has at least 318 raw daily observations after schema cleaning.
- Reports reconcile with the path manifest.

**Human review required:** No

**Notes for Codex CLI:**
This is raw-history screening only. Later missing-data filtering may exclude additional assets.
---

### T-08 — Materialize canonical daily-close store
---
**Purpose:**
Persist a clean, deterministic daily-close store for all admitted FTMO assets.

**Inputs:**
- `artifacts/reports/asset_eligibility_report.csv`
- `artifacts/manifests/d_timeframe_path_manifest.json`
- `docs/contracts/daily_close_schema_contract.md`

**Dependencies:**
- T-07

**Exact action:**
For every eligible asset, load the resolved `D` file under the schema contract, retain only canonical columns `timestamp`, `asset_id`, and `close`, enforce ascending order, enforce one row per timestamp under the schema rules, and persist one canonical CSV per asset. Create a manifest containing file path, row count, first timestamp, last timestamp, and file hash for each canonical series.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/canonical_daily_close/<asset_id>.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/manifests/canonical_daily_close_manifest.json`

**Acceptance criteria:**
- Every eligible asset has exactly one canonical daily-close CSV.
- Canonical files contain only `timestamp`, `asset_id`, and `close`.
- The canonical store manifest reconciles with actual files and counts.
- No excluded asset receives a canonical CSV.

**Human review required:** No

**Notes for Codex CLI:**
Do not resample, impute, or infer missing dates here.
---

### G-02 — Review Gate: FTMO daily-close data contract
---
**Review scope:**
Allowed asset manifest, `D`-timeframe path manifest, schema contract, eligibility/exclusion reports, and canonical daily-close store.

**Pass conditions:**
- Every admitted asset resolves to FTMO `D` data and `close` only.
- Every exclusion has an explicit reason code.
- The canonical store contains only daily close series.
- No intraday or OHLC leakage remains in the pipeline definition.

**Blocked downstream tasks:**
- T-09
- T-10
- T-11
- T-12

**Reviewer:** Human
---

### M-02 — FTMO Data Contract & Canonical Daily-Close Layer Complete

---

## 03 — Deterministic Feature Engineering

### T-09 — Implement arithmetic returns and 60-day EWM volatility engine
---
**Purpose:**
Create the primitive return and volatility series required by the model.

**Inputs:**
- `artifacts/manifests/canonical_daily_close_manifest.json`
- Canonical daily-close CSVs from Section 02

**Dependencies:**
- G-02

**Exact action:**
Implement Python modules `src/lstm_cpd/features/returns.py` and `src/lstm_cpd/features/volatility.py`. For each asset, compute one-day arithmetic return as `(p_t - p_{t-1}) / p_{t-1}`. Then compute the ex-ante volatility estimate `sigma_t` as the exponentially weighted moving standard deviation of daily arithmetic returns using deterministic implementation settings `span=60`, `adjust=False`, `min_periods=60`, and `bias=False`. Persist one per-asset series file containing canonical timestamp, asset_id, close, arithmetic return, and `sigma_t`.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/features/returns.py`
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/features/volatility.py`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/features/base/<asset_id>_returns_volatility.csv`

**Acceptance criteria:**
- Daily return matches arithmetic return, not log return.
- `sigma_t` is produced with a single deterministic EWM implementation across all assets.
- Every output row is keyed by canonical timestamp and asset_id.
- Files are generated for every asset admitted by G-02.

**Human review required:** No

**Notes for Codex CLI:**
Record the chosen EWM implementation in feature provenance later and do not change it downstream.
---

### T-10 — Construct normalized multi-horizon return features
---
**Purpose:**
Generate the five paper-required normalized return inputs.

**Inputs:**
- `artifacts/features/base/<asset_id>_returns_volatility.csv`

**Dependencies:**
- T-09

**Exact action:**
For each asset and for horizons `{1, 21, 63, 126, 256}`, compute the close-to-close arithmetic interval return `p_t / p_{t-t'} - 1`. Normalize each interval return as `z_{t,t'} = r_{t-t',t} / (sigma_t * sqrt(t'))`. Persist the five normalized return features alongside canonical keys. Preserve the paper’s 256-day annual horizon exactly; do not replace it with 252.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/features/normalized_returns.py`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/features/base/<asset_id>_normalized_returns.csv`

**Acceptance criteria:**
- Exactly five normalized return features are created per asset-date.
- Horizons are exactly `1, 21, 63, 126, 256`.
- The annual 256-day horizon is preserved.
- All features use `sigma_t` from T-09 and canonical daily close data only.

**Human review required:** No

**Notes for Codex CLI:**
Use close-to-close arithmetic interval returns as canonical implementation form.
---

### T-11 — Construct normalized MACD feature set
---
**Purpose:**
Generate the three paper-required MACD inputs.

**Inputs:**
- `artifacts/manifests/canonical_daily_close_manifest.json`
- Canonical daily-close CSVs from Section 02

**Dependencies:**
- T-10

**Exact action:**
Implement `src/lstm_cpd/features/macd.py`. For each pair `(8,24)`, `(16,28)`, `(32,96)`, compute the exponentially weighted moving averages of daily close using `alpha = 1/x`, which matches the Spec’s half-life identity. Compute `MACD_t(S,L) = m_t(S) - m_t(L)`. Compute `q_t(S,L) = MACD_t(S,L) / std(p_{t-63:t})` using trailing rolling standard deviation with `min_periods=63`. Then compute `Y_t(S,L) = q_t(S,L) / std(q_{t-252:t}(S,L))` using trailing rolling standard deviation with `min_periods=252`. Persist the three final MACD features by asset-date.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/features/macd.py`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/features/base/<asset_id>_macd_features.csv`

**Acceptance criteria:**
- Exactly three MACD features are produced using pairs `(8,24)`, `(16,28)`, `(32,96)`.
- MACD features are computed from daily close only.
- The two-step normalization uses trailing windows 63 and 252 exactly.
- Output files are keyed by canonical timestamp and asset_id.

**Human review required:** No

**Notes for Codex CLI:**
Do not substitute any benchmark MACD implementation from other projects; use the Spec-defined formulation only.
---

### T-12 — Apply feature-level winsorization and materialize base feature store
---
**Purpose:**
Freeze the eight non-CPD model features into their final causal base-feature form.

**Inputs:**
- `artifacts/features/base/<asset_id>_normalized_returns.csv`
- `artifacts/features/base/<asset_id>_macd_features.csv`
- `docs/contracts/invariant_ledger.md`

**Dependencies:**
- T-10
- T-11

**Exact action:**
Join the five normalized return features and three MACD features by asset-date. For each of the eight feature series independently, apply causal feature-level winsorization using trailing EWM mean ± 5 trailing EWM standard deviations with half-life 252. Do not winsorize raw prices. Do not winsorize CPD features. Persist one final base-feature file per asset and a provenance report documenting formulas, horizons, normalization steps, and winsorization settings.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/features/winsorize.py`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/features/base/<asset_id>_base_features.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/reports/feature_provenance_report.md`

**Acceptance criteria:**
- Final base-feature files contain exactly eight non-CPD features per asset-date.
- Winsorization is applied only to those eight features.
- Winsorization is causal and uses half-life 252.
- The provenance report explicitly records arithmetic returns, 60-day EWM volatility, horizons `{1,21,63,126,256}`, MACD pairs `{(8,24),(16,28),(32,96)}`, and winsorization scope.

**Human review required:** No

**Notes for Codex CLI:**
This task freezes the non-CPD feature layer. Do not append CPD features here.
---

### G-03 — Review Gate: Deterministic feature correctness
---
**Review scope:**
Arithmetic returns, 60-day EWM volatility, normalized return features, MACD features, winsorized base-feature store, and feature provenance report.

**Pass conditions:**
- Arithmetic returns are used instead of log returns.
- Volatility is the 60-day EWM estimate used consistently.
- Return horizons are exactly `{1,21,63,126,256}`.
- MACD pairs are exactly `{(8,24),(16,28),(32,96)}`.
- Winsorization applies only to the eight non-CPD features.

**Blocked downstream tasks:**
- T-13
- T-14
- T-15

**Reviewer:** Human
---

### M-03 — Deterministic Feature Engineering Complete

---

## 04 — CPD Precomputation

### T-13 — Implement spec-faithful CPD GP fitting engine
---
**Purpose:**
Create the exact CPD fitting runtime required by the authoritative Spec.

**Inputs:**
- `artifacts/features/base/<asset_id>_returns_volatility.csv`
- `docs/contracts/invariant_ledger.md`
- `docs/contracts/exclusions_ledger.md`

**Dependencies:**
- G-03

**Exact action:**
Implement the CPD engine in Python modules under `src/lstm_cpd/cpd/`. For each asset-time pair and chosen LBW, construct the rolling return window `{r_t}_{t=T-l}^{T}`, standardize returns within the window to zero mean and unit variance, fit a baseline GP with Matérn 3/2 kernel and all baseline hyperparameters initialized to 1 at every timestep, then initialize the changepoint GP with `c = t - l/2`, `s = 1`, and all remaining changepoint-kernel parameters cloned from the fitted baseline GP into both pre- and post-changepoint kernels. Enforce changepoint location constraint `c in (t-l, t)`. Use GPflow’s SciPy L-BFGS-B optimizer with no extra generic restarts. If the first changepoint fit fails, reinitialize all changepoint parameters to 1 except `c = t - l/2` and retry exactly once. If the second fit fails, set `nu_t = nu_{t-1}` and `gamma_t = gamma_{t-1}`. Compute severity and location exactly as specified and log every fit, retry, and fallback event.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/cpd/gp_kernels.py`
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/cpd/fit_window.py`
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/cpd/precompute_contract.py`
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/cpd_engine_contract.md`

**Acceptance criteria:**
- Baseline GP uses Matérn 3/2 and resets initialization at every timestep.
- Changepoint GP uses the constrained changepoint kernel and exactly one retry after failure.
- No extra optimizer restarts or multistart search is performed.
- The engine computes `nu_t` and `gamma_t` and exposes telemetry needed for downstream audit.

**Human review required:** No

**Notes for Codex CLI:**
CPD is an upstream precomputation stage only. Do not backpropagate through GP fits or integrate CPD into the differentiable graph.
---

### T-14 — Precompute CPD features for all allowed LBWs
---
**Purpose:**
Materialize the two CPD features for every allowed changepoint lookback window.

**Inputs:**
- CPD engine modules from T-13
- `artifacts/features/base/<asset_id>_returns_volatility.csv`
- `artifacts/manifests/canonical_daily_close_manifest.json`

**Dependencies:**
- T-13

**Exact action:**
For each admitted asset and each LBW in `{10, 21, 63, 126, 252}`, run the CPD engine across all eligible timesteps. Persist per-asset CPD feature files containing at least `timestamp`, `asset_id`, `lbw`, `nu`, `gamma`, `nlml_baseline`, `nlml_changepoint`, `retry_used`, and `fallback_used`.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/features/cpd/lbw_10/<asset_id>_cpd.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/features/cpd/lbw_21/<asset_id>_cpd.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/features/cpd/lbw_63/<asset_id>_cpd.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/features/cpd/lbw_126/<asset_id>_cpd.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/features/cpd/lbw_252/<asset_id>_cpd.csv`

**Acceptance criteria:**
- CPD features are produced only for LBWs `{10,21,63,126,252}`.
- Each per-LBW output file contains both `nu` and `gamma`.
- Per-file rows are aligned to canonical timestamps.
- Missing timesteps are handled by the Spec-defined fallback or omitted according to upstream availability, never by imputation.

**Human review required:** No

**Notes for Codex CLI:**
Do not create a multi-LBW CPD input tensor for the model. Each LBW remains a separate upstream feature store.
---

### T-15 — Consolidate CPD telemetry and fallback reports
---
**Purpose:**
Make CPD behavior fully auditable before dataset assembly begins.

**Inputs:**
- Per-LBW CPD feature files from T-14

**Dependencies:**
- T-14

**Exact action:**
Aggregate all per-window CPD telemetry into global reports covering fit status, retry counts, fallback counts, affected timesteps, and asset/LBW coverage. Produce a manifest that lists every CPD feature file and the row counts needed for deterministic joins with base features later.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/reports/cpd_fit_telemetry.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/reports/cpd_failure_ledger.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/reports/cpd_fallback_ledger.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/manifests/cpd_feature_store_manifest.json`

**Acceptance criteria:**
- Every CPD file created in T-14 appears in the manifest.
- Retry and fallback events are explicitly logged with asset, timestamp, and LBW.
- Telemetry is sufficient to verify one-retry-only and previous-output fallback behavior.
- No CPD feature store is missing coverage metadata.

**Human review required:** No

**Notes for Codex CLI:**
This report is part of the fidelity contract. Do not suppress failed fits or fallback rows.
---

### G-04 — Review Gate: CPD fidelity
---
**Review scope:**
CPD engine implementation, per-LBW CPD feature stores, fit telemetry, failure ledger, and fallback ledger.

**Pass conditions:**
- Rolling window standardization is implemented exactly.
- Baseline GP uses Matérn 3/2 and per-timestep reinitialization.
- Changepoint kernel enforces location constraint and one retry only.
- Fallback behavior matches the frozen Spec closure.
- CPD artifacts are audit-ready by asset, timestamp, and LBW.

**Blocked downstream tasks:**
- T-16
- T-17
- T-18

**Reviewer:** Human
---

### M-04 — CPD Precomputation Complete

---

## 05 — Split, Sequence, and Dataset Assembly

### T-16 — Generate chronological train-validation split manifests
---
**Purpose:**
Create deterministic, per-asset chronological 90/10 splits over usable joined features.

**Inputs:**
- `artifacts/features/base/<asset_id>_base_features.csv`
- `artifacts/features/cpd/lbw_<lbw>/<asset_id>_cpd.csv`
- `artifacts/manifests/cpd_feature_store_manifest.json`

**Dependencies:**
- G-04

**Exact action:**
For each LBW and each admitted asset, join the eight-feature base store with the matching two CPD features on asset-date keys. Retain only rows where all 10 timestep features are present. Then create chronological per-asset splits where the first 90% of usable rows form training and the final 10% form validation. Persist split manifests and joined feature tables by LBW.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/joined_features/<asset_id>.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/split_manifest.csv`

**Acceptance criteria:**
- Every split is chronological and per asset.
- Only rows with all 10 features present remain in joined feature tables.
- Train and validation counts reconcile with the 90/10 rule.
- No random shuffling occurs during splitting.

**Human review required:** No

**Notes for Codex CLI:**
Do not mix assets before splitting. Splits are asset-local first, shared dataset later.
---

### T-17 — Build non-overlapping 63-step sequences and target-alignment registry
---
**Purpose:**
Transform joined feature tables into exact model-ready sequences with correctly aligned next-step targets.

**Inputs:**
- `artifacts/datasets/lbw_<lbw>/split_manifest.csv`
- Joined feature tables from T-16

**Dependencies:**
- T-16

**Exact action:**
Within each asset, split, and LBW, partition rows into contiguous non-overlapping sequences of exactly 63 timesteps. Discard any terminal fragment shorter than 63. Do not pad, overlap, wrap, stitch, or create variable-length sequences. Detect missing-feature gaps and exclude any sequence containing a gap. For each retained timestep, compute the aligned target-scale value `0.15 / sigma_t * r_{t+1}` for use in Sharpe-loss training. Persist a sequence manifest and a target-alignment registry that records sequence ID, asset ID, split, LBW, start timestamp, end timestamp, and target alignment.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/sequence_manifest.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/target_alignment_registry.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/discarded_fragments_report.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/gap_exclusion_report.csv`

**Acceptance criteria:**
- Every retained sequence has length exactly 63.
- No retained sequence overlaps another sequence for the same asset and split.
- Target-scale values align to next-step returns, not same-step returns.
- Discard and gap-exclusion reports account for every removed fragment or sequence.

**Human review required:** No

**Notes for Codex CLI:**
Do not create classification labels. The target object here is the volatility-scaled future-return factor used by the Sharpe loss.
---

### T-18 — Materialize batch-ready dataset registry by LBW
---
**Purpose:**
Freeze the exact arrays and metadata that the training runtime will consume.

**Inputs:**
- Sequence manifests from T-17
- Target-alignment registries from T-17

**Dependencies:**
- T-17

**Exact action:**
For each LBW, materialize NumPy arrays for training and validation inputs with shape `[num_sequences, 63, 10]` and corresponding target-scale arrays with shape `[num_sequences, 63]`. Also persist sequence index tables linking every array row to asset ID, split, sequence ID, and timestamp boundaries. Build a global dataset registry that records every LBW dataset artifact path, row count, tensor shape, and source manifests.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/train_inputs.npy`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/train_target_scale.npy`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/val_inputs.npy`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/val_target_scale.npy`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/train_sequence_index.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/datasets/lbw_<lbw>/val_sequence_index.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/manifests/dataset_registry.json`

**Acceptance criteria:**
- Input arrays have final timestep dimension 10 and sequence length 63.
- Target-scale arrays reconcile with the target-alignment registry.
- Dataset registry contains one entry per LBW dataset.
- Training runtime can load every dataset artifact without inferring schema.

**Human review required:** No

**Notes for Codex CLI:**
This task finalizes the dataset contract. Do not modify sequence layout downstream.
---

### G-05 — Review Gate: Split, sequence, and dataset assembly
---
**Review scope:**
Joined feature tables, split manifests, sequence manifests, target-alignment registries, discarded-fragment reports, gap-exclusion reports, and dataset registry.

**Pass conditions:**
- Splits are per-asset chronological 90/10.
- Sequences are contiguous, non-overlapping, and exactly length 63.
- No missing-data imputation or gap-bridging is present.
- Terminal fragments shorter than 63 are discarded.
- Target alignment is correct for next-step volatility-scaled returns.

**Blocked downstream tasks:**
- T-19
- T-20
- T-21

**Reviewer:** Human
---

### M-05 — Split, Sequence, and Dataset Assembly Complete

---

## 06 — LSTM Training Core

### T-19 — Implement shared LSTM DMN and Sharpe-loss runtime
---
**Purpose:**
Create the exact paper-faithful model and loss runtime that all candidate trainings must use.

**Inputs:**
- `artifacts/manifests/dataset_registry.json`
- `docs/contracts/invariant_ledger.md`
- `docs/contracts/execution_policy_rules.md`

**Dependencies:**
- G-05

**Exact action:**
Implement `src/lstm_cpd/model/network.py` and `src/lstm_cpd/training/losses.py`. Define a single shared stateless unidirectional LSTM with hidden size `H`, followed directly by one time-distributed dense output layer of size 1 with `tanh` activation. Implement dropout at two sites only: inputs to the single LSTM layer and sequence outputs of that LSTM immediately before the final dense head. Reuse one dropout mask per sequence across timesteps. Disable dropout at validation and inference time. Do not implement recurrent dropout. Implement the Sharpe loss over all asset-time pairs using model outputs multiplied by the target-scale tensor from Section 05.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/model/network.py`
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/training/losses.py`
- `Dev/Research/Slow Momentum with fast mean reversion/docs/contracts/model_runtime_contract.md`

**Acceptance criteria:**
- The model contains exactly one stateless unidirectional LSTM layer.
- No extra dense hidden layer or extra recurrent layer exists.
- Output shape is compatible with `[batch, 63, 1]`.
- Sharpe loss consumes target-scale tensors and does not rely on supervised class labels.

**Human review required:** No

**Notes for Codex CLI:**
Any architectural enrichment beyond the direct LSTM -> time-distributed `tanh` head is forbidden.
---

### T-20 — Implement deterministic training runner
---
**Purpose:**
Create the reusable candidate-training entrypoint for search and validation.

**Inputs:**
- `src/lstm_cpd/model/network.py`
- `src/lstm_cpd/training/losses.py`
- `artifacts/manifests/dataset_registry.json`
- `docs/contracts/execution_policy_rules.md`

**Dependencies:**
- T-19

**Exact action:**
Implement `src/lstm_cpd/training/train_candidate.py`. The runner must load one LBW dataset entry and one candidate configuration, create the model runtime from T-19, train with Adam, apply global gradient-norm clipping using the candidate’s `max_grad_norm`, run for up to 300 epochs, stop early after patience 25 on validation loss, and persist best-checkpoint artifacts and epoch-level logs. Use deterministic sequence-level shuffling and batch construction according to the frozen execution-policy rules.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/training/train_candidate.py`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/training_runner_contract.md`

**Acceptance criteria:**
- The runner can execute one candidate end-to-end from dataset loading to checkpoint persistence.
- Adam, max-300 epochs, patience 25, and global gradient clipping are implemented exactly.
- Validation loss is logged every epoch.
- Training behavior is deterministic under the frozen seed and batching rules.

**Human review required:** No

**Notes for Codex CLI:**
Do not embed outer-loop search logic here. This is a single-candidate runner only.
---

### T-21 — Execute smoke-train fidelity run
---
**Purpose:**
Validate that the entire model-training stack works end-to-end before running the full search.

**Inputs:**
- `src/lstm_cpd/training/train_candidate.py`
- `artifacts/manifests/dataset_registry.json`

**Dependencies:**
- T-20

**Exact action:**
Run one fixed smoke candidate with configuration `dropout=0.1`, `hidden_size=20`, `minibatch_size=64`, `learning_rate=1e-3`, `max_grad_norm=1.0`, and `lbw=63`. Persist the config snapshot, best checkpoint, epoch log, and validation history under a dedicated smoke-run directory. Do not treat this run as part of the 50-candidate search.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/smoke_run/smoke_config.json`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/smoke_run/smoke_best_model.keras`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/smoke_run/smoke_epoch_log.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/smoke_run/smoke_validation_history.csv`

**Acceptance criteria:**
- The smoke run completes without changing the runtime contract.
- All four smoke artifacts are written.
- Validation loss is logged across epochs.
- The produced checkpoint can be loaded back by the runtime.

**Human review required:** No

**Notes for Codex CLI:**
This run is purely for fidelity validation. Do not use it for model selection.
---

### G-06 — Review Gate: Model fidelity
---
**Review scope:**
Model runtime contract, training runner, smoke-run artifacts, and training configuration handling.

**Pass conditions:**
- The runtime uses one shared stateless unidirectional LSTM only.
- The final head is direct time-distributed dense `tanh`.
- Dropout, Sharpe loss, Adam, gradient clipping, max-300 epochs, and patience-25 stopping are implemented exactly.
- No extra hidden or recurrent layer exists.
- Smoke-run artifacts prove the runtime works end-to-end.

**Blocked downstream tasks:**
- T-22
- T-23
- T-24

**Reviewer:** Human
---

### M-06 — LSTM Training Core Complete

---

## 07 — Hyperparameter Search & Model Selection

### T-22 — Generate immutable 50-candidate search schedule
---
**Purpose:**
Create the exact search schedule that the outer loop must execute.

**Inputs:**
- `docs/contracts/execution_policy_rules.md`
- `docs/contracts/invariant_ledger.md`

**Dependencies:**
- G-06

**Exact action:**
Enumerate the full discrete hyperparameter grid using the frozen nested-loop order and allowed values: dropout `{0.1,0.2,0.3,0.4,0.5}`, hidden size `{5,10,20,40,80,160}`, minibatch size `{64,128,256}`, learning rate `{1e-4,1e-3,1e-2,1e-1}`, max gradient norm `{1e-2,1e0,1e2}`, and LBW `{10,21,63,126,252}`. Sample exactly 50 unique candidates without replacement using the frozen seed rule. Assign immutable IDs `C-001` through `C-050` and persist the full grid and sampled schedule.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/search/full_search_grid.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/search/candidate_schedule.csv`

**Acceptance criteria:**
- The sampled schedule contains exactly 50 unique candidates.
- Every candidate lies inside the allowed search grid.
- Candidate IDs are stable and ordered.
- The schedule is reproducible from the frozen sampling rule.

**Human review required:** No

**Notes for Codex CLI:**
Once persisted, this schedule is immutable. Do not redraw or reorder candidates.
---

### T-23 — Run 50 candidate trainings and persist search artifacts
---
**Purpose:**
Execute the full outer-loop search exactly once against the frozen schedule.

**Inputs:**
- `artifacts/training/search/candidate_schedule.csv`
- `artifacts/manifests/dataset_registry.json`
- `src/lstm_cpd/training/train_candidate.py`

**Dependencies:**
- T-22

**Exact action:**
For each candidate in candidate-ID order, run the deterministic training runner against the dataset registry entry matching that candidate’s LBW. Persist per-candidate config snapshots, best checkpoints, epoch logs, validation histories, status flags, and artifact paths under candidate-specific directories. Update a master registry row after each candidate completes.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/search/candidate_registry.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/search/candidate_<candidate_id>/candidate_config.json`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/search/candidate_<candidate_id>/best_model.keras`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/search/candidate_<candidate_id>/epoch_log.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/search/candidate_<candidate_id>/validation_history.csv`

**Acceptance criteria:**
- The master registry contains 50 candidate rows.
- Every successful candidate row includes checkpoint and validation-history paths.
- Any failed candidate is explicitly logged with failure status and reason.
- No candidate outside the frozen schedule is executed.

**Human review required:** No

**Notes for Codex CLI:**
Run candidates strictly in schedule order. Do not silently skip or replace failed candidates.
---

### T-24 — Select winning model and freeze selection metadata
---
**Purpose:**
Bind the winning configuration and checkpoint to a single frozen selected-model artifact.

**Inputs:**
- `artifacts/training/search/candidate_registry.csv`
- Candidate validation histories from T-23
- `docs/contracts/execution_policy_rules.md`

**Dependencies:**
- T-23

**Exact action:**
Read the candidate registry, identify the candidate with minimum validation loss, apply the frozen tie-breaking rule if needed, copy that candidate’s best checkpoint into the selected-model location, and write a selection report containing selected candidate ID, selected LBW, full hyperparameter vector, minimum validation loss, and tie-resolution details if applicable.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/selected/selected_model.keras`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/selected/selected_model_metadata.json`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/selected/model_selection_report.md`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/training/selected/selected_validation_history.csv`

**Acceptance criteria:**
- The selected candidate exists in the 50-candidate registry.
- The selected metadata contains candidate ID, LBW, and full hyperparameters.
- The selection report records the exact validation loss used for selection.
- No post-selection retraining is performed.

**Human review required:** No

**Notes for Codex CLI:**
Selection is based on minimum validation loss only, with the frozen tie rule as sole fallback.
---

### G-07 — Review Gate: Search and selection
---
**Review scope:**
Full search grid, sampled 50-candidate schedule, candidate registry, candidate artifacts, and selected-model metadata/report.

**Pass conditions:**
- Exactly 50 search iterations were executed against the allowed grid.
- The selected candidate corresponds to minimum validation loss.
- Any tie or randomness policy used is explicitly recorded.
- The selected checkpoint and metadata are frozen and traceable.

**Blocked downstream tasks:**
- T-25
- T-26
- T-27

**Reviewer:** Human
---

### M-07 — Hyperparameter Search & Model Selection Complete

---

## 08 — Causal Inference & Evaluation

### T-25 — Implement causal online inference path
---
**Purpose:**
Freeze the production-style inference path that uses only the latest causal sequence and final output position.

**Inputs:**
- `artifacts/training/selected/selected_model.keras`
- `artifacts/training/selected/selected_model_metadata.json`
- Canonical daily-close store from Section 02
- Feature and CPD generation modules from Sections 03 and 04

**Dependencies:**
- G-07

**Exact action:**
Implement `src/lstm_cpd/inference/online_inference.py`. For the selected LBW only, recompute the most recent required base features and CPD outputs from the newest available canonical daily-close data, assemble the latest 63-step feature sequence for each admitted asset, run the selected model, and emit only the final position value for each asset as the next-day holding instruction. Persist both the positions output and a manifest describing the exact latest sequence used per asset.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/src/lstm_cpd/inference/online_inference.py`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/inference/latest_positions.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/inference/latest_sequence_manifest.csv`

**Acceptance criteria:**
- Exactly one final position per admitted asset is produced.
- Only the latest causal 63-step sequence is used.
- No future data, bidirectional context, or multi-LBW input is used.
- The sequence manifest is sufficient to reproduce each inference output.

**Human review required:** No

**Notes for Codex CLI:**
This path must mirror training-time feature definitions exactly and use the selected LBW only.
---

### T-26 — Run validation-universe evaluation and persist raw/rescaled metrics
---
**Purpose:**
Produce artifact-backed evaluation outputs for the frozen selected model.

**Inputs:**
- `artifacts/training/selected/selected_model.keras`
- `artifacts/training/selected/selected_model_metadata.json`
- `artifacts/manifests/dataset_registry.json`
- `docs/contracts/invariant_ledger.md`

**Dependencies:**
- T-25

**Exact action:**
Using the selected LBW validation dataset, run the selected model across validation sequences and compute per-timestep realized strategy returns. Aggregate equal-weight portfolio daily returns across available assets. Compute and persist raw validation metrics at minimum for annualized return, annualized volatility, annualized downside deviation, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, and percentage of positive daily strategy returns. Then apply the Spec-closed constant test-window volatility rescaling formula and persist the corresponding rescaled returns and metrics in separately labeled artifacts. Write an evaluation report that explicitly states that FTMO evaluation is model-faithful but not a literal reproduction of the paper’s 50-futures universe.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/evaluation/raw_validation_returns.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/evaluation/raw_validation_metrics.json`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/evaluation/rescaled_validation_returns.csv`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/evaluation/rescaled_validation_metrics.json`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/evaluation/evaluation_report.md`

**Acceptance criteria:**
- Raw and rescaled evaluation outputs are both persisted and clearly labeled.
- Metrics are reproducible from frozen selected-model and dataset artifacts.
- The report explicitly discloses the FTMO universe mismatch versus the paper’s original 50 futures.
- No benchmark strategy replication is introduced.

**Human review required:** No

**Notes for Codex CLI:**
This section evaluates the selected LSTM+CPD model only. Do not expand scope into full-paper benchmark replication.
---

### T-27 — Build reproducibility manifest
---
**Purpose:**
Create a single machine-readable map of the final pipeline state and all artifacts needed to rerun it.

**Inputs:**
- `artifacts/training/selected/selected_model_metadata.json`
- `artifacts/training/search/candidate_schedule.csv`
- `artifacts/manifests/ftmo_asset_universe.json`
- `artifacts/manifests/canonical_daily_close_manifest.json`
- `artifacts/manifests/dataset_registry.json`
- Evaluation artifacts from T-26

**Dependencies:**
- T-26

**Exact action:**
Collect the selected candidate ID, selected LBW, sampled search schedule hash, seed policy, asset-universe manifest hash, canonical-store manifest hash, dataset registry hash, source module entrypoints, and final artifact paths into one reproducibility manifest. Include enough metadata for another deterministic run to identify the exact selected configuration and all required inputs.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/reproducibility/reproducibility_manifest.json`

**Acceptance criteria:**
- The manifest resolves every final artifact path produced through T-26.
- It records the selected candidate ID and selected LBW explicitly.
- It records all seeds and frozen execution-policy rules needed for rerun.
- It references the exact code entrypoints used for feature generation, CPD, training, inference, and evaluation.

**Human review required:** No

**Notes for Codex CLI:**
Keep this manifest machine-readable and stable. Do not embed human-only narrative here.
---

### G-08 — Review Gate: Causal inference and evaluation
---
**Review scope:**
Online inference path, latest positions output, raw/rescaled evaluation artifacts, and reproducibility manifest.

**Pass conditions:**
- Inference uses the latest causal sequence and final position only.
- Evaluation artifacts are reproducible from frozen selected-model and dataset artifacts.
- The FTMO universe mismatch is explicitly disclosed.
- Reproducibility metadata is complete enough for deterministic rerun.

**Blocked downstream tasks:**
- T-28
- T-29

**Reviewer:** Human
---

### M-08 — Causal Inference & Evaluation Complete

---

## 09 — Final Notebook Assembly

### T-28 — Assemble final reproducible notebook from frozen artifacts
---
**Purpose:**
Create the final research notebook as a presentation and reproducibility wrapper around frozen implementation outputs.

**Inputs:**
- `artifacts/reproducibility/reproducibility_manifest.json`
- All frozen artifacts from Sections 01–08
- Source modules from Sections 02–08

**Dependencies:**
- G-08

**Exact action:**
Create `notebooks/lstm_cpd_replication.ipynb`. Structure the notebook to load and display frozen artifacts and call existing source modules only. Include sections for implementation contract, FTMO data contract, canonical daily-close layer, base features, CPD outputs, dataset assembly, model/training setup, search results, selected model, causal inference, validation evaluation, and reproducibility manifest. Ensure that all core logic exists in source modules or persisted artifacts, not only in notebook cells.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/notebooks/lstm_cpd_replication.ipynb`

**Acceptance criteria:**
- The notebook is created only after all core artifacts are frozen.
- Every notebook section references existing artifacts or source modules.
- No new methodology is implemented solely inside notebook cells.
- The notebook covers the full frozen pipeline end-to-end.

**Human review required:** No

**Notes for Codex CLI:**
This is the final packaging milestone. Do not move any core implementation logic into the notebook.
---

### T-29 — Execute notebook end-to-end and persist notebook mapping outputs
---
**Purpose:**
Verify that the final notebook is reproducible and fully tied to frozen artifacts.

**Inputs:**
- `notebooks/lstm_cpd_replication.ipynb`
- All frozen artifacts from Sections 01–08

**Dependencies:**
- T-28

**Exact action:**
Run the notebook from a clean kernel top-to-bottom without manual intervention. Save the executed notebook, write a notebook execution report recording execution success and section order, and write a notebook-to-artifact mapping table that lists which artifact each notebook section consumes or visualizes.

**Output artifacts:**
- `Dev/Research/Slow Momentum with fast mean reversion/notebooks/lstm_cpd_replication.executed.ipynb`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/notebook/notebook_execution_report.md`
- `Dev/Research/Slow Momentum with fast mean reversion/artifacts/notebook/notebook_artifact_map.csv`

**Acceptance criteria:**
- The executed notebook completes successfully from a fresh kernel.
- The execution report confirms no manual edits were required during execution.
- The notebook-artifact map covers every notebook section.
- All displayed results trace back to frozen upstream artifacts only.

**Human review required:** No

**Notes for Codex CLI:**
Do not patch notebook cells during execution except to fix path references that were already defined upstream.
---

### G-09 — Review Gate: Final notebook assembly and endgame
---
**Review scope:**
Final notebook, executed notebook, notebook execution report, and notebook-to-artifact mapping.

**Pass conditions:**
- The notebook was assembled only after frozen implementation artifacts existed.
- The executed notebook runs end-to-end from a fresh kernel.
- The notebook contains no notebook-only core logic.
- Every notebook result is traceable to frozen artifacts or source modules.

**Blocked downstream tasks:**
- —

**Reviewer:** Human
---

### M-09 — Final Notebook Assembly Complete

---

## Dependency map

| Task ID | Title | Depends On |
|---------|-------|------------|
| T-01 | Compile invariant and exclusions ledgers | — |
| T-02 | Register unresolved execution-policy items | T-01 |
| T-03 | Freeze execution-policy rules for Codex-safe implementation | T-02 |
| T-04 | Scaffold repository and artifact blueprint | T-03 |
| G-01 | Review Gate: Source fidelity and execution rule freeze | T-01, T-02, T-03, T-04 |
| T-05 | Build allowed FTMO asset-universe manifest | G-01 |
| T-06 | Resolve D-timeframe paths and daily-close schema contract | T-05 |
| T-07 | Screen raw availability and minimum-history sufficiency | T-06 |
| T-08 | Materialize canonical daily-close store | T-07 |
| G-02 | Review Gate: FTMO daily-close data contract | T-05, T-06, T-07, T-08 |
| T-09 | Implement arithmetic returns and 60-day EWM volatility engine | G-02 |
| T-10 | Construct normalized multi-horizon return features | T-09 |
| T-11 | Construct normalized MACD feature set | T-10 |
| T-12 | Apply feature-level winsorization and materialize base feature store | T-10, T-11 |
| G-03 | Review Gate: Deterministic feature correctness | T-09, T-10, T-11, T-12 |
| T-13 | Implement spec-faithful CPD GP fitting engine | G-03 |
| T-14 | Precompute CPD features for all allowed LBWs | T-13 |
| T-15 | Consolidate CPD telemetry and fallback reports | T-14 |
| G-04 | Review Gate: CPD fidelity | T-13, T-14, T-15 |
| T-16 | Generate chronological train-validation split manifests | G-04 |
| T-17 | Build non-overlapping 63-step sequences and target-alignment registry | T-16 |
| T-18 | Materialize batch-ready dataset registry by LBW | T-17 |
| G-05 | Review Gate: Split, sequence, and dataset assembly | T-16, T-17, T-18 |
| T-19 | Implement shared LSTM DMN and Sharpe-loss runtime | G-05 |
| T-20 | Implement deterministic training runner | T-19 |
| T-21 | Execute smoke-train fidelity run | T-20 |
| G-06 | Review Gate: Model fidelity | T-19, T-20, T-21 |
| T-22 | Generate immutable 50-candidate search schedule | G-06 |
| T-23 | Run 50 candidate trainings and persist search artifacts | T-22 |
| T-24 | Select winning model and freeze selection metadata | T-23 |
| G-07 | Review Gate: Search and selection | T-22, T-23, T-24 |
| T-25 | Implement causal online inference path | G-07 |
| T-26 | Run validation-universe evaluation and persist raw/rescaled metrics | T-25 |
| T-27 | Build reproducibility manifest | T-26 |
| G-08 | Review Gate: Causal inference and evaluation | T-25, T-26, T-27 |
| T-28 | Assemble final reproducible notebook from frozen artifacts | G-08 |
| T-29 | Execute notebook end-to-end and persist notebook mapping outputs | T-28 |
| G-09 | Review Gate: Final notebook assembly and endgame | T-28, T-29 |

---

## Readiness assessment

Is the task system execution-ready for Codex CLI? **Yes.** It preserves the approved phase order, keeps the notebook strictly last, enforces the FTMO `D`/`close` contract, and turns the four previously unresolved execution-policy items into explicit early artifacts and dependencies so implementation can proceed deterministically. fileciteturn7file17 fileciteturn7file19

Are there any remaining blockers? **No project-structure blockers remain.** The approved Plan identified only four execution-policy gaps that had to be frozen before coding tasks began, and this task system freezes them in Section 01 rather than leaving them to Codex CLI. fileciteturn7file17

Can Codex CLI execute all tasks sequentially from Asana without rethinking structure? **Yes.** Every downstream task consumes only upstream-produced artifacts, every critical handoff has a human review gate, and the dependency chain matches the approved workstream logic from contract freeze through final notebook assembly. fileciteturn7file10 fileciteturn7file19

**"The project is now Asana-ready and execution-ready for Codex CLI."**

