# Exclusions Ledger

## Purpose

This ledger is the binding exclusion output of `T-01`.

Each item below is explicitly out of scope for this replication unless a later human-approved document reopens it. The Spec remains the sole methodological authority for what counts as a strict replication.

## Explicit exclusions

| ID | Excluded design or behavior | Why excluded | Source basis | Impacted tasks |
| --- | --- | --- | --- | --- |
| EXC-01 | Full-paper replication, benchmark-strategy replication, or benchmark logic folded into the LSTM+CPD pipeline | The project scope is the paper's LSTM + CPD model only, not the full paper or its classical benchmarks. | Spec Section 0; Plan Section 2; Project Overview Scope | All tasks |
| EXC-02 | Literal recreation of the paper's original 50-futures universe inside the FTMO workflow | The FTMO universe is a project overlay and is not the paper's original universe. | Spec Section 13.2; Plan Section 2 | T-05, T-06, T-16 |
| EXC-03 | Intraday, weekly, monthly, or any non-`D` timeframe input in the canonical pipeline | Using non-daily inputs changes the structural day-based horizons and is therefore an adaptation rather than a replication. | Spec Sections 3.1, 12.1, 13.1 | T-06, T-08, T-09, T-11 |
| EXC-04 | OHLC-, volume-, or multi-field feature engineering beyond the daily close column | The model is formulated from closing price and derived returns only. | Spec Sections 3.1, 13.1 | T-06, T-08, T-09, T-11 |
| EXC-05 | External datasets, synthetic data, or data augmentation outside the project-defined FTMO sources | Project data inputs are strictly limited to the approved FTMO documents and files. | Plan Section 2; Project Overview Datenbasis | T-05 through T-18 |
| EXC-06 | Hard regime labels such as bull, bear, correction, or rebound used as model inputs or supervised targets | Regime language in the paper is explanatory only and is not part of the final model input/output contract. | Spec Section 11.1 | T-12, T-16, T-17, T-19 |
| EXC-07 | Changepoint-threshold trading rules based on severity cutoffs | Thresholds shown in the paper are for illustration and plotting, not the production trading rule. | Spec Section 11.2 | T-14, T-19, later evaluation |
| EXC-08 | Multiple CPD modules in parallel, multi-LBW feature tensors, or feeding several LBWs into the LSTM at once | The paper explicitly rejects this as the final design. | Spec Sections 11.3, 15 | T-14, T-16, T-18, T-19 |
| EXC-09 | Transaction-cost-adjusted loss as the default objective | The default objective is the Sharpe-ratio loss without transaction-cost adjustment. | Spec Sections 0, 11.4 | T-19, T-20 |
| EXC-10 | Supervised class labels, sign labels, or price-direction classification targets | Training is on the trading objective, not on a classification surrogate. | Spec Section 7.2 | T-17, T-19 |
| EXC-11 | Bidirectional LSTM variants | A bidirectional recurrent model is incompatible with the paper's causal online use case. | Spec Section 9 | T-19, T-21 |
| EXC-12 | Stateful recurrent execution that carries batch state across sequences | The model is explicitly stateless between batches. | Spec Section 6.4 | T-19, T-20 |
| EXC-13 | Additional recurrent layers beyond the single LSTM layer | The closed architecture contains exactly one recurrent layer. | Spec Section 14 | T-19, T-21 |
| EXC-14 | Additional dense hidden layers between the LSTM and the final output head | The single LSTM feeds directly into one time-distributed dense `tanh` output layer. | Spec Section 14 | T-19, T-21 |
| EXC-15 | Recurrent dropout or any dropout site other than LSTM inputs and LSTM outputs before the dense head | Dropout is closed to exactly two sites with no recurrent-state dropout. | Spec Sections 6.6, 14 | T-19, T-21 |
| EXC-16 | Generic GP optimizer restarts, multistart searches, or tolerance/bounds overrides beyond the paper-specified retry | The CPD fitting policy is deliberately narrow and audit-ready. | Spec Section 14 | T-13, T-15 |
| EXC-17 | Missing-data imputation, forward fill, back fill, interpolation, zero fill, or sequence stitching across gaps | Missing upstream values must lead to dropped timesteps or dropped sequences, never synthetic continuity. | Spec Section 14 | T-07, T-08, T-16, T-17 |
| EXC-18 | Padding, overlapping windows, wrapping, or variable-length training sequences | The training contract is fixed to contiguous, non-overlapping 63-step sequences only. | Spec Sections 6.5, 14 | T-17, T-18 |
| EXC-19 | Notebook-first development or notebook-only core implementation logic | The notebook is the final packaging artifact after implementation artifacts are frozen. | Plan Sections 1, 4, 8; Project Overview Arbeitsmodus | T-04, later notebook assembly |

## Usage rule

Any downstream design proposal that reintroduces one of these excluded behaviors must be treated as a scope change and escalated for explicit human review before implementation.
