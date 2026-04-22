# Unresolved Execution-Policy Items

## Purpose

This register is the binding reporting output of `T-02`.

It lists execution-policy questions that are not fully closed by the Spec but must be frozen before implementation tasks rely on them. This document reports unresolved items only; it does not decide them.

## Unresolved items

| Item ID | Decision topic | Why unresolved at spec level | Downstream components affected | Later tasks that must consume the decision |
| --- | --- | --- | --- | --- |
| UEI-01 | Random-search sampling mode | The Spec fixes the search grid and the 50-iteration budget, but it does not specify whether candidate sampling is with or without replacement, nor how the candidate schedule is materialized for deterministic replay. | Search scheduler, candidate registry, run manifests, reproducibility records | T-03, T-20 and later search orchestration |
| UEI-02 | Tie-breaking rule for equal validation losses | The Spec selects by minimum validation loss, but it does not define what to do when multiple candidates share the same minimum value after serialization/logging. | Winner selection logic, model-selection report, run manifest | T-03, later search orchestration |
| UEI-03 | Random-seed policy | The Spec does not fix the global seed, epoch-level shuffle seed behavior, or any candidate-specific deterministic substream policy. | TensorFlow runtime reproducibility, candidate replay, shuffle reproducibility, run manifests | T-03, T-19, T-20, T-21 |
| UEI-04 | Batching convention across assets | The Spec closes sequence length and sequence shuffling, but it does not fully define whether assets remain isolated through batching or are concatenated into a shared dataset before minibatching. | Dataset registry, training loader, epoch shuffle behavior, batch metadata retention | T-03, T-18, T-20 |

## Completeness statement

No additional unresolved execution-policy items were discovered in the approved source documents that would force a later task to choose sampling, tie-breaking, seeding, or batching behavior silently.
