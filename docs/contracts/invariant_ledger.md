# Implementation Invariant Ledger

## Authority and extraction order

This ledger is the binding output of `T-01`.

Source order used for extraction:

1. `spec_lstm_cpd_model_revised_sole_authority.md`
2. `lstm_cpd_implementation_plan.md`
3. `project_overview_slow_momentum_with_fast_reversion.md`

Methodological authority:

- The Spec is the sole methodological authority.
- The Implementation Plan and Project Overview supply workflow, scope, and project-context constraints.

## Binding invariants

| ID | Domain | Binding invariant | Source basis | Downstream consumers |
| --- | --- | --- | --- | --- |
| INV-01 | Governance | The project implements only the paper's LSTM + CPD model on the project-defined FTMO universe; it does not attempt full-paper replication. | Spec Sections 0, 15; Plan Sections 1-3; Project Overview Scope | All tasks |
| INV-02 | Governance | Any implementation detail not fully fixed by the Spec must be surfaced and explicitly frozen before coding proceeds; silent defaults are forbidden. | Spec Section 0; Plan Sections 3-4; Project Overview Replikationsprinzipien | T-02, T-03, T-19, T-20 |
| INV-03 | Data contract | Raw market input is a daily close series only. For the FTMO mapping, the faithful project overlay is the `D` timeframe with the `close` column as the canonical primitive series. | Spec Sections 3.1, 12.1, 13.1 | T-05, T-06, T-08, T-09, T-11 |
| INV-04 | Data contract | The project asset universe is the project-defined FTMO universe from `data/FTMO Data/ftmo_assets_nach_kategorie.md`; the implementation may be paper-faithful in model logic while remaining non-identical to the paper's original 50-futures universe. | Spec Section 13.2; Plan Sections 1-3; Project Overview Datenbasis | T-05, T-06, T-07, T-08, T-16 |
| INV-05 | Returns | The primitive return series is the one-day arithmetic close-to-close return `(p_t - p_{t-1}) / p_{t-1}`, not a log return. | Spec Sections 2, 3.2, 15 | T-09, T-10, T-13, T-17 |
| INV-06 | Volatility | The ex-ante scaling series is the per-asset 60-day exponentially weighted moving volatility `sigma_t`. | Spec Sections 3.3, 7.1, 15 | T-09, T-10, T-17, T-19 |
| INV-07 | Return features | The normalized return feature block contains exactly five features at horizons `{1, 21, 63, 126, 256}` using close-to-close arithmetic interval returns divided by `sigma_t * sqrt(horizon)`. The 256-day annual horizon must be preserved exactly. | Spec Section 4.1, Section 14, Section 15 | T-10, T-12, T-16 |
| INV-08 | MACD features | The MACD feature block contains exactly three features with pairs `(8,24)`, `(16,28)`, and `(32,96)`, using the Spec-defined two-stage normalization on daily close data only. | Spec Sections 4.2, 14, 15 | T-11, T-12, T-16 |
| INV-09 | Winsorization | Winsorization is applied only at the feature level, only after the five normalized returns and three MACDs are constructed, and only to those eight non-CPD features using trailing EWM mean plus/minus five trailing EWM standard deviations with half-life 252. Raw prices and CPD features are not winsorized. | Spec Sections 3.4, 14 | T-12, T-16 |
| INV-10 | CPD feature contract | Each timestep contributes exactly two CPD features, severity `nu_t` and normalized location `gamma_t`, and the model uses one CPD module with one LBW at a time. Allowed LBWs are `{10, 21, 63, 126, 252}`. | Spec Sections 1, 4.3, 5.1, 15 | T-13, T-14, T-15, T-16 |
| INV-11 | CPD preprocessing | Every CPD fit uses a rolling return window for the chosen LBW, standardized within the window to zero mean and unit variance before GP fitting. | Spec Sections 5.1-5.2 | T-13, T-14 |
| INV-12 | CPD fitting | The baseline CPD GP uses a Matérn 3/2 kernel in GPflow, optimized with SciPy L-BFGS-B, and its hyperparameters are reinitialized to `1` at every timestep. | Spec Sections 5.3, 12.3, 14 | T-13, T-15 |
| INV-13 | CPD changepoint behavior | The changepoint GP is initialized from the fitted baseline GP with `c = t - l/2`, `s = 1`, and a constrained changepoint location `c in (t-l, t)`. If the first changepoint fit fails, exactly one reinitialization-and-refit attempt is allowed; if that second fit fails, set `nu_t := nu_{t-1}` and `gamma_t := gamma_{t-1}`. | Spec Sections 5.5, 14 | T-13, T-14, T-15 |
| INV-14 | CPD role | CPD is an upstream precomputation stage. GP fits are not part of the differentiable graph and are not backpropagated through. | Spec Sections 2, 5, 12.3 | T-13, T-14, T-19 |
| INV-15 | Timestep contract | The per-timestep model input is exactly a 10-feature vector: 5 normalized returns, 3 MACDs, and 2 CPD outputs. | Spec Sections 2, 4.4, 6.8, 15 | T-16, T-18, T-19, T-21 |
| INV-16 | Splitting | Train/validation splitting is chronological and asset-local: first 90 percent of usable observations for training, final 10 percent for validation. | Spec Section 3.5, Section 15 | T-16, T-17, T-18 |
| INV-17 | Sequence construction | Training and validation data are partitioned into contiguous, non-overlapping sequences of exactly 63 timesteps. Terminal fragments shorter than 63 are discarded. | Spec Sections 2, 6.3-6.5, 14, 15 | T-17, T-18, T-19 |
| INV-18 | Missing-data handling | No missing-data imputation is allowed. Returns, features, CPD outputs, and next-step targets exist only where all upstream inputs exist; no gap bridging is allowed, and any gap-contaminated 63-step sequence must be discarded. | Spec Section 14 | T-07, T-08, T-16, T-17 |
| INV-19 | Minimum usable history | The project treats roughly 318 daily observations as the earliest usable upstream-history threshold for producing a full 63-step sequence with the 256-day feature horizon. | Spec Section 12.2; Plan Section 3 | T-07, T-16, T-17 |
| INV-20 | Architecture | The model is a single shared network across assets: exactly one stateless, unidirectional LSTM layer followed directly by one time-distributed dense layer with `tanh` activation that outputs one position scalar per timestep. Hidden Layer Size denotes the number of units in that single LSTM. | Spec Sections 1, 6.4, 6.7-6.8, 7.3, 14, 15 | T-19, T-20, T-21 |
| INV-21 | Dropout | During training only, dropout is applied at exactly two sites: on LSTM inputs and on LSTM sequence outputs immediately before the final dense head. One mask is sampled per sequence and reused across all timesteps. Validation and inference disable dropout. | Spec Sections 6.6, 14 | T-19, T-20, T-21 |
| INV-22 | Loss and target wiring | The optimization objective is Sharpe-ratio loss over volatility-scaled future returns across all asset-time pairs. The realized return uses the model output together with `sigma_t`, `r_{t+1}`, and the 15 percent target-volatility scaling factor. No class or sign label is part of the objective. | Spec Sections 7.1-7.3, 15 | T-17, T-19, T-20, T-21 |
| INV-23 | Training runtime | The neural runtime uses TensorFlow/Keras with Adam, a maximum budget of 300 epochs, patience 25 early stopping, and global gradient-norm clipping semantics for Max Gradient Norm. | Spec Sections 8.1-8.3, 14 | T-19, T-20, T-21 |
| INV-24 | Search and model selection | Hyperparameter search runs for 50 iterations over the paper-defined grid, and model selection is by minimum validation loss. The chosen LBW for the optimized-LBW variant is the LBW attached to the minimum-validation-loss candidate. | Spec Sections 8.3-8.4, 14 | T-20 and later search orchestration |
| INV-25 | Inference and evaluation | Inference is strictly causal: compute the latest CPD features, build the latest sequence, run the LSTM, and use only the final output position for the next day. If a paper-style experiment is activated later, it must use the Spec's expanding-window evaluation protocol and optional Exhibit-4 rescaling formula. | Spec Sections 9, 10, 14, 15 | T-20, T-21, later evaluation work |

## Ledger usage rule

All downstream tasks must treat these invariants as binding unless a later human-approved document explicitly supersedes them. Downstream implementations may operationalize an invariant, but they must not weaken, reinterpret, or silently broaden it.
