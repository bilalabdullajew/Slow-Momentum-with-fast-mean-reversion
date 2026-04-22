# LSTM CPD Implementation Plan

## 1. Objective

Implement a spec-faithful Python replication of the paper’s **LSTM + CPD** model under `Dev/Research/Slow Momentum with fast mean reversion`, using only the project-defined FTMO sources, with the model logic fixed to: daily close-based arithmetic returns, 60-day EWM volatility, 5 normalized return features, 3 MACD features, 2 CPD features, non-overlapping 63-step sequences, one stateless unidirectional LSTM, time-distributed `tanh` output, Sharpe-ratio training objective, and causal online inference. The implementation target is the model pipeline itself, not the full paper. The final Jupyter Notebook is the last packaging milestone, not a development surface. Source basis: [authoritative spec](spec_lstm_cpd_model_revised_sole_authority.md) and [project overview](project_overview_slow_momentum_with_fast_reversion.md).

## 2. In-Scope / Out-of-Scope

**In scope**

- Python-only implementation of the authoritative LSTM CPD pipeline.
- FTMO-based data resolution using only the allowed asset list and allowed file/timeframe structure.
- Daily `D` timeframe mapping with **close column only**.
- Canonical preprocessing: arithmetic returns, 60-day EWM volatility, 5 normalized return features, 3 MACD features, feature-level winsorization, CPD features.
- Precomputed CPD feature generation for allowed LBW values.
- Chronological 90/10 train/validation splitting per asset.
- Exact 63-step non-overlapping sequence construction.
- Single shared LSTM model across assets.
- Training, hyperparameter search, model selection, causal inference path.
- Final reproducible notebook assembled after implementation artifacts are frozen.

**Out of scope**

- Full-paper replication and all benchmark strategies.
- Literal reproduction of the paper’s original 50-futures asset universe.
- Intraday or OHLC-based adaptation.
- External or synthetic data.
- Hard regime labels, changepoint-threshold trading rules, multiple parallel CPD modules, transaction-cost objective as default, extra recurrent layers, extra dense hidden layers, bidirectional/stateful variants.
- Early notebook-first development.

## 3. Planning Assumptions

- The authoritative spec is the sole implementation authority. Section 14 closures are binding; closed points are not reopened.
- The project implements **paper-faithful model logic on FTMO data**, not a literal recreation of the paper’s asset universe.
- The only faithful FTMO mapping is daily `D` data with `close` as the primitive series; using other timeframes would be a methodology change.
- Core model implementation does not require the exact paper backtest. Paper-style expanding-window evaluation is a separate evaluation layer and only applies if that experiment is explicitly activated.
- The earliest usable endpoint for a full sequence requires roughly **318 daily observations** of usable upstream history; this affects asset eligibility.
- Missing values are not imputed, gaps are not bridged, and incomplete 63-step fragments are discarded.
- CPD is precomputed outside the differentiable graph and must be treated as an upstream feature-generation stage.
- Any still-unspecified execution detail from the paper must be surfaced as an explicit project rule before coding, not handled by silent defaults.

## 4. Execution Phases

### Phase 1 — Implementation Contract Freeze

- **Purpose:** Convert the spec into a non-negotiable execution contract and lock the project control surface.
- **Inputs:** Authoritative spec, project overview, user constraints.
- **Outputs:** Invariant ledger, exclusion ledger, repository artifact map, unresolved-execution-item register, phase/workstream map.
- **Dependency conditions:** None.
- **Completion criteria:** Every normative spec requirement is mapped to a later artifact or validation gate; every excluded design is explicitly listed; every non-spec execution choice that still needs freezing is logged.

### Phase 2 — FTMO Data Contract and Canonical Daily-Close Layer

- **Purpose:** Resolve the permitted FTMO universe into canonical daily-close series that can feed the paper-defined pipeline.
- **Inputs:** Phase 1 contract, `ftmo_assets_nach_kategorie.md`, `FTMO Data_struktur.md`.
- **Outputs:** Asset-resolution manifest, `D`-timeframe path manifest, close-column schema contract, asset eligibility/exclusion report, minimum-history screening report.
- **Dependency conditions:** Phase 1 approved.
- **Completion criteria:** Every allowed asset is either mapped to an admissible daily-close source or excluded with a reason; no non-`D` or non-close input remains in the pipeline definition.

### Phase 3 — Deterministic Feature and CPD Precomputation

- **Purpose:** Build the full causal per-asset feature layer needed before sequence assembly and training.
- **Inputs:** Canonical daily-close layer from Phase 2.
- **Outputs:** Arithmetic return series, 60-day EWM volatility series, 5 normalized return features, 3 MACD features, winsorized feature store, CPD feature stores for each allowed LBW, CPD fit/failure logs.
- **Dependency conditions:** Phase 2 complete.
- **Completion criteria:** The 10-feature timestep contract is realizable; feature formulas match the spec; CPD fits obey the required kernel/init/retry/fallback rules; data gaps are handled per spec.

### Phase 4 — Split, Sequence, and Dataset Assembly

- **Purpose:** Transform per-asset feature tables into exact training/validation datasets with correct targets.
- **Inputs:** Feature/CPD stores from Phase 3.
- **Outputs:** Per-asset chronological split manifests, 63-step sequence manifests, discarded-fragment report, missing-gap exclusion report, batch-ready dataset registry keyed by LBW.
- **Dependency conditions:** Phase 3 validated.
- **Completion criteria:** No overlap, padding, wrapping, stitching, or variable-length sequences remain; next-step targets align correctly with positions and volatility-scaled returns.

### Phase 5 — Model Training, Search, and Selection

- **Purpose:** Train the exact LSTM DMN, run the allowed hyperparameter search, and select the winning configuration by validation loss.
- **Inputs:** Dataset registry from Phase 4, training/search rules from the spec.
- **Outputs:** Candidate-run registry, training logs, validation-loss history, selected hyperparameter record, selected model checkpoint, model-selection report.
- **Dependency conditions:** Phase 4 complete.
- **Completion criteria:** Architecture, dropout, clipping, optimizer, budget, patience, search-space, and selection rules all match the spec; the selected model is traceable to minimum validation loss.

### Phase 6 — Causal Inference, Evaluation, and Notebook Endgame

- **Purpose:** Freeze the production inference path, assemble evaluation outputs, and only then produce the final notebook.
- **Inputs:** Selected model and all frozen artifacts from Phases 1–5.
- **Outputs:** Online inference run path, evaluation report set, reproducibility manifest, final notebook.
- **Dependency conditions:** Phase 5 approved.
- **Completion criteria:** Inference uses the latest causal sequence and final output position only; evaluation is artifact-backed; the notebook is assembled from frozen implementation outputs and introduces no new core logic.

## 5. Workstreams

### Workstream 1 — Source Governance and Invariant Ledger

- **Purpose:** Prevent scope drift and encode the spec as reviewable implementation invariants.
- **Inputs:** Authoritative spec, project overview, user constraints.
- **Outputs:** Invariant ledger, exclusions ledger, unresolved-execution-item register, artifact inventory.
- **Dependency conditions:** None.
- **Completion criteria:** All architectural, feature, data, training, and evaluation invariants are enumerated and each later workstream references them.

### Workstream 2 — Repository and Configuration Blueprint

- **Purpose:** Define where all future artifacts live and how execution is parameterized without changing methodology.
- **Inputs:** Workstream 1 outputs.
- **Outputs:** Directory blueprint under the project root, naming conventions, config schema for assets/timeframe/LBW/search runs, run-manifest schema.
- **Dependency conditions:** Workstream 1 complete.
- **Completion criteria:** Every later deliverable has a destination, identifier, and ownership boundary.

### Workstream 3 — FTMO Asset Resolution and Canonical Series Assembly

- **Purpose:** Convert the permitted FTMO source definitions into admissible daily-close input series.
- **Inputs:** Workstream 2 blueprint, FTMO asset and path documents.
- **Outputs:** Asset/path manifest, canonical close-series schema, per-asset availability report, exclusion report.
- **Dependency conditions:** Workstreams 1–2 complete.
- **Completion criteria:** Each asset is either resolved to `D`/close or formally excluded; no hidden data-source assumptions remain.

### Workstream 4 — Core Feature Engine

- **Purpose:** Generate all non-CPD model features exactly as specified.
- **Inputs:** Canonical close series.
- **Outputs:** Daily arithmetic returns, 60-day EWM volatility, 5 normalized multi-horizon returns, 3 MACD series, feature-level winsorized output store, feature provenance log.
- **Dependency conditions:** Workstream 3 complete.
- **Completion criteria:** Exact horizons and MACD pairs are preserved; arithmetic, not log, returns are used; winsorization touches only the eight specified non-CPD features.

### Workstream 5 — CPD Engine and LBW-Specific Feature Stores

- **Purpose:** Produce severity and location features for each allowed changepoint LBW.
- **Inputs:** Canonical returns from Workstream 4’s upstream series, spec CPD rules.
- **Outputs:** CPD feature stores for LBW values 10/21/63/126/252, CPD fit telemetry, retry/failure ledger, fallback ledger.
- **Dependency conditions:** Workstream 3 complete.
- **Completion criteria:** Baseline and changepoint GP fits use the exact kernels and init/reset policy; one retry only; fallback behavior matches the normative closure; outputs are aligned per asset-date.

### Workstream 6 — Split, Sequence, and Target Builder

- **Purpose:** Assemble model-ready examples from the feature and CPD stores.
- **Inputs:** Workstreams 4–5 outputs.
- **Outputs:** Per-LBW train/validation split manifests, sequence manifests, dropped-fragment report, gap-exclusion report, target-alignment registry.
- **Dependency conditions:** Workstreams 4–5 complete.
- **Completion criteria:** 90/10 chronological splits are per asset; sequences are contiguous, non-overlapping, length 63 only; any gap-contaminated or undersized sequence is removed.

### Workstream 7 — LSTM Training Core

- **Purpose:** Define the exact model/training runtime independent of search orchestration.
- **Inputs:** Workstream 6 datasets, spec architecture/training rules.
- **Outputs:** Single shared LSTM training runtime, checkpoint artifacts, epoch logs, validation-loss logs.
- **Dependency conditions:** Workstream 6 complete.
- **Completion criteria:** Exactly one stateless unidirectional LSTM layer is followed directly by a time-distributed dense `tanh` head; Sharpe loss, dropout, clipping, Adam, max-300-epoch budget, and patience-25 stopping are implemented exactly.

### Workstream 8 — Hyperparameter Search and Model Selection Orchestrator

- **Purpose:** Execute the outer-loop search and bind the winning configuration to a frozen model artifact.
- **Inputs:** Workstream 7 runtime, Workstream 6 datasets, search grid from the spec.
- **Outputs:** 50-run search registry, candidate metadata, selected hyperparameter vector, selected LBW record, winner checkpoint, selection report.
- **Dependency conditions:** Workstreams 6–7 complete.
- **Completion criteria:** Every candidate is auditable; the selected model corresponds to minimum validation loss; any paper-unspecified selection edge case is handled by an explicit project rule, not a silent default.

### Workstream 9 — Causal Inference and Evaluation Harness

- **Purpose:** Freeze the post-training path for daily online use and artifact-backed evaluation.
- **Inputs:** Selected model, feature/CPD generation path, dataset/evaluation manifests.
- **Outputs:** Latest-sequence inference path, evaluation reports, optional Exhibit-4-style rescaled metrics if evaluation beyond raw validation is included.
- **Dependency conditions:** Workstream 8 complete.
- **Completion criteria:** Only the final timestep position is used for trading; all inference is causal; evaluation outputs can be reproduced from frozen artifacts.

### Workstream 10 — Final Notebook Assembly

- **Purpose:** Consolidate the complete research flow into the final notebook only after the implementation is stable.
- **Inputs:** Outputs from Workstreams 1–9.
- **Outputs:** Final reproducible notebook, notebook execution order, notebook-to-artifact mapping.
- **Dependency conditions:** All prior workstreams complete.
- **Completion criteria:** The notebook is a presentation and reproducibility wrapper around frozen pipeline artifacts; no hidden logic exists only inside the notebook.

## 6. Dependency Structure

Primary dependency chain:

`WS1 -> WS2 -> WS3 -> {WS4, WS5} -> WS6 -> WS7 -> WS8 -> WS9 -> WS10`

Operational meaning:

- Nothing proceeds before the invariant ledger and artifact map exist.
- Daily-close asset resolution must finish before any feature or CPD work.
- Non-CPD features and CPD features can be produced in parallel once canonical series exist.
- Sequence assembly depends on both feature branches.
- Training depends on frozen dataset assembly.
- Search depends on a stable training core, not vice versa.
- Inference/evaluation depends on the selected model.
- The notebook depends on all prior frozen artifacts.

Additional dependency conditions:

- Any project rule needed to close a still-paper-unspecified execution detail must be frozen before WS8 candidate execution.
- Any evaluation that claims paper-style behavior must explicitly declare the FTMO universe mismatch and treat it as a project-level adaptation, not a literal paper-universe reproduction.

## 7. Deliverables by Phase

### Phase 1
- Invariant ledger
- Exclusions ledger
- Unresolved-execution-item register
- Repository/config artifact map

### Phase 2
- FTMO asset-resolution manifest
- Daily `D`/close canonical-series schema
- Asset eligibility and exclusion report
- Minimum-history sufficiency report

### Phase 3
- Per-asset base feature store
- Per-LBW CPD feature stores
- Feature provenance report
- CPD fit/retry/failure report

### Phase 4
- Per-asset train/validation split manifests
- Sequence registry by LBW
- Dropped-fragment report
- Gap-exclusion report
- Target-alignment registry

### Phase 5
- Candidate-run registry
- Training and validation logs
- Winner-selection report
- Selected model checkpoint and metadata

### Phase 6
- Causal inference run path
- Evaluation report set
- Reproducibility manifest
- Final Jupyter Notebook

## 8. Validation Gates

### Gate A — Source Fidelity Gate
Human check that the plan and later artifacts use only the authoritative spec for model decisions and the project overview for project workflow/context. Pass only if no excluded design re-enters implicitly.

### Gate B — FTMO Data Contract Gate
Human check that every admitted asset resolves to FTMO `D` data and `close` only, with no intraday or OHLC leakage. Pass only if every exclusion has a reason code.

### Gate C — Feature Correctness Gate
Human check of arithmetic returns, 60-day EWM volatility, horizons `{1,21,63,126,256}`, MACD pairs `{(8,24),(16,28),(32,96)}`, and winsorization scope. Pass only if the feature store is exactly the 8 non-CPD features plus later 2 CPD features.

### Gate D — CPD Fidelity Gate
Human check of rolling window standardization, Matérn 3/2 baseline GP, changepoint kernel, changepoint-location constraint, per-timestep reinitialization, one retry only, and fallback behavior. Pass only if the CPD log proves the exact fit policy.

### Gate E — Dataset Assembly Gate
Human check of per-asset chronological 90/10 split, no imputation, no gap-bridging, 63-step non-overlapping sequences, terminal-fragment discard, and correct target alignment. Pass only if the dataset registry exposes all dropped items and their reasons.

### Gate F — Model Fidelity Gate
Human check of one shared stateless unidirectional LSTM, direct time-distributed dense `tanh` head, exact dropout policy, Sharpe loss, Adam, clipping, max-300 epochs, patience 25. Pass only if no extra hidden or recurrent layer exists.

### Gate G — Search and Selection Gate
Human check that 50 search iterations were executed against the allowed grid and that the selected model corresponds to lowest validation loss. Pass only if any tie/randomness policy used is explicitly recorded.

### Gate H — Endgame Gate
Human check that online inference uses the latest causal sequence and final position only, and that the notebook was assembled after frozen artifacts existed. Pass only if the notebook contains no notebook-only core logic.

## 9. Implementation Risk Controls

- **Risk: timeframe/data-source drift.**  
  **Control:** Canonical `D`/close manifest is created before any downstream work and enforced at Gate B.

- **Risk: formula drift in features.**  
  **Control:** Workstream 4 emits a feature provenance report keyed to every formula-sensitive field, including the preserved `256` annual horizon mismatch.

- **Risk: CPD behavior drift under implementation pressure.**  
  **Control:** Workstream 5 requires fit telemetry, retry logs, and fallback logs so warm starts, extra restarts, or altered failure handling are review-visible.

- **Risk: leakage via sequence construction.**  
  **Control:** Workstream 6 forces split manifests, gap reports, and dropped-fragment reports before any model training can start.

- **Risk: architecture drift into richer networks.**  
  **Control:** Invariant ledger plus Gate F explicitly block extra dense layers, extra recurrent layers, statefulness, and bidirectionality.

- **Risk: hidden per-asset modeling instead of a shared network.**  
  **Control:** Training runtime and candidate registry are organized around one shared model artifact and one loss definition across asset-time pairs.

- **Risk: notebook becomes the real implementation.**  
  **Control:** Notebook is Phase 6/WS10 only; it may orchestrate or visualize frozen artifacts but may not contain unique implementation logic.

- **Risk: paper-unspecified execution details are filled silently.**  
  **Control:** Any such item must appear first in the unresolved-execution-item register, then be frozen as an explicit project rule before coding tasks are written.

- **Risk: confusion between model-faithful and paper-universe-faithful replication.**  
  **Control:** Every evaluation artifact must declare that FTMO execution is model-faithful but not a literal recreation of the paper’s original futures universe.

## 10. Blockers or Spec Contradictions

No direct internal contradiction was identified in the authoritative spec.

There are, however, still **paper-unspecified execution-policy items** that must not be left implicit when moving from Plan to Tasks:

1. **Random-search sampling rule** in the 50-iteration search is not specified as with-replacement or without-replacement.
2. **Tie-breaking rule** for equal validation losses is not specified.
3. **Random seed policy** is not specified.
4. **Exact batching layout across assets** is not normatively closed by the spec.

These are not model-definition contradictions. They do not block project planning, but they must be converted into explicit project rules before Codex CLI implementation tasks begin.

Separate from that, the spec itself flags a **project-level deviation**: FTMO data cannot literally reproduce the paper’s original 50-futures universe. That is not a contradiction in model logic, but any later evaluation must label the difference correctly.

## 11. Readiness for Task Decomposition

The project is structurally ready to move from **Plan** to **Tasks**.

What is already ready for atomic tasking:

- phase order
- workstream boundaries
- artifact expectations
- dependency chain
- validation gates
- notebook-last endgame
- scope boundaries and anti-drift rules

What must be handled first in the task list before coding tasks start:

- freeze the remaining paper-unspecified execution-policy items into explicit project rules:
  - random-search sampling mode
  - tie-breaking rule
  - seed policy
  - batching convention across assets

Once those are captured as first-class tasks/artifacts, Codex CLI can implement the remaining pipeline without rethinking project structure or silently choosing methodology.

**The project is now plan-ready.**