# Execution Policy Rules

## Purpose

This document is the binding closure output of `T-03`.

It freezes only execution-policy items that the Spec leaves operationally open. Nothing in this document overrides a methodological closure already fixed by the Spec.

## Rule 1 - Random-search sampling mode

Enumerate the full discrete Cartesian product in this fixed nested-loop order:

1. `dropout`
2. `hidden_size`
3. `minibatch_size`
4. `learning_rate`
5. `max_grad_norm`
6. `lbw`

The searched value sets are:

- `dropout in {0.1, 0.2, 0.3, 0.4, 0.5}`
- `hidden_size in {5, 10, 20, 40, 80, 160}`
- `minibatch_size in {64, 128, 256}`
- `learning_rate in {1e-4, 1e-3, 1e-2, 1e-1}`
- `max_grad_norm in {1e-2, 1e0, 1e2}`
- `lbw in {10, 21, 63, 126, 252}`

After materializing that full ordered grid, draw exactly 50 unique candidates without replacement using:

`random.Random(20260421).sample(full_grid, 50)`

Candidate schedule rules:

- The sampled list is immutable once created.
- `candidate_index` means the zero-based position of a candidate inside that immutable sampled list.
- All later references to a candidate must use that same schedule order.

## Rule 2 - Tie-breaking

Select the candidate with minimum validation loss.

If multiple candidates share the exact same serialized minimum value, select the candidate with the smallest `candidate_index` in the immutable sampled schedule.

Tie-breaking rules:

- No secondary metric may be introduced.
- No manual review step is inserted between equal-loss candidates.
- The tie is resolved entirely by schedule position.

## Rule 3 - Seed policy

Set Python `random`, NumPy, and TensorFlow seeds to `20260421`.

Use:

- per-epoch shuffle seed `20260421 + epoch_index`
- candidate-specific deterministic seed `20260421 + candidate_index` only where a separate deterministic stream is needed

Seed-index rules:

- `epoch_index` is zero-based.
- `candidate_index` is the zero-based immutable sampled-schedule position defined in Rule 1.
- No task may invent additional uncontrolled random streams.

## Rule 4 - Batching across assets

Construct sequences independently per asset and per split, concatenate them into one shared dataset per LBW while preserving `asset_id` and sequence index metadata, shuffle only at sequence level once per epoch, keep the final smaller batch, and do not perform asset balancing or stratification.

Batching consequences:

- Splits stay asset-local before concatenation.
- Shuffling occurs after sequence construction, not before.
- Batches may contain sequences from multiple assets.
- Sequence metadata must remain recoverable from registry artifacts and run manifests.

## Enforcement

These rules are binding for every downstream task that touches search scheduling, candidate replay, seeds, batching, or deterministic training behavior.
