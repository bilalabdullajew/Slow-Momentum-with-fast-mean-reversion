"""Training runtime helpers for the LSTM CPD replication."""

from lstm_cpd.training.losses import SharpeLoss, sharpe_loss
from lstm_cpd.training.search_runner import (
    SearchCompletionRecord,
    load_search_completion_log,
    run_search_schedule,
)
from lstm_cpd.training.search_schedule import (
    build_search_schedule,
    load_search_schedule,
    materialize_search_schedule,
)
from lstm_cpd.training.selection import BestCandidateSelection, select_best_candidate
from lstm_cpd.training.train_candidate import (
    CandidateConfig,
    TrainingRunResult,
    candidate_config_to_payload,
    load_candidate_config,
    load_dataset_registry_entry,
    run_candidate_training,
    run_smoke_fidelity,
)

__all__ = [
    "SharpeLoss",
    "sharpe_loss",
    "CandidateConfig",
    "TrainingRunResult",
    "SearchCompletionRecord",
    "BestCandidateSelection",
    "candidate_config_to_payload",
    "load_candidate_config",
    "load_dataset_registry_entry",
    "run_candidate_training",
    "run_smoke_fidelity",
    "build_search_schedule",
    "load_search_schedule",
    "materialize_search_schedule",
    "run_search_schedule",
    "load_search_completion_log",
    "select_best_candidate",
]
