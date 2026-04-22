# Daily-Close Schema Contract

## Purpose

This document is the binding schema contract produced by `T-06`.

It defines how FTMO raw `D` timeframe files become admissible daily-close sources for later tasks.

## Authority

- Live Asana task authority: `T-06 — Resolve D-timeframe paths and daily-close schema contract`
- Methodological constraints: `docs/contracts/invariant_ledger.md` and `docs/contracts/exclusions_ledger.md`

## D-path resolution rules

The path manifest resolves exactly one repo-relative `D` timeframe file per allowed asset using these path families:

1. Standard path family:
   `data/FTMO Data/{category}/{symbol}/D/{symbol}_data.csv`
2. Forex path family:
   `data/FTMO Data/Forex/{base_ccy}/{symbol}/D/{symbol}_data.csv`
   where `base_ccy = symbol[:3]`

Resolution outcomes:

- If exactly one candidate exists, the asset is `RESOLVED`.
- If no candidate exists, the asset is `FAILED` with reason code `MISSING_D_PATH`.
- If more than one candidate exists, the asset is `FAILED` with reason code `AMBIGUOUS_D_PATH`.

## Admissible source columns

For a resolved `D` file:

- Accept exactly one source timestamp column chosen case-insensitively from:
  - `timestamp`
  - `datetime`
  - `date`
  - `time`
- Map that source column to canonical column `timestamp`.
- Accept exactly one source close column chosen case-insensitively from:
  - `close`
- Map that source column to canonical column `close`.

All other source columns are ignored for canonical daily-close admission, including `open`, `high`, `low`, `volume`, `spread`, and every non-`D` timeframe field.

## Validation and ordering rules

- Every timestamp value must be parseable as a timestamp.
- Every close value must be parseable as numeric.
- Canonical rows are ordered ascending by parsed timestamp.
- The contract refers only to `D` timeframe and `close` for canonical admission.

## Duplicate-timestamp rules

- If duplicate timestamps have identical close values, keep the last occurrence and log the duplicate count.
- If duplicate timestamps disagree on close, exclude the asset with reason code `DUPLICATE_TIMESTAMP_CONFLICT`.

## Exclusion conditions

Resolved files are excluded from canonical admission if any of the following holds:

- `MISSING_TIMESTAMP_COLUMN`
- `MULTIPLE_TIMESTAMP_COLUMNS`
- `MISSING_CLOSE_COLUMN`
- `MULTIPLE_CLOSE_COLUMNS`
- `UNPARSEABLE_TIMESTAMP`
- `NON_NUMERIC_CLOSE`
- `DUPLICATE_TIMESTAMP_CONFLICT`

## Deferred checks

This contract does not decide raw-availability outcomes such as empty series or insufficient history. Those checks are deferred to `T-07`.
