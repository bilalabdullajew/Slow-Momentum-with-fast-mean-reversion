from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.training.search_runner import (
    COMPLETION_STATUS_COMPLETED,
    SearchCompletionRecord,
    default_search_completion_log_path,
    load_search_completion_log,
)
from lstm_cpd.training.train_candidate import (
    CandidateConfig,
    candidate_config_to_payload,
)


DEFAULT_BEST_CANDIDATE_PATH = "artifacts/training/best_candidate.json"
DEFAULT_BEST_CONFIG_PATH = "artifacts/training/best_config.json"
DEFAULT_SEARCH_SUMMARY_REPORT_PATH = "artifacts/reports/search_summary_report.csv"
SEARCH_SUMMARY_REPORT_HEADER = (
    "candidate_index",
    "candidate_id",
    "status",
    "selected",
    "best_validation_loss",
    "failure_reason",
    "dropout",
    "hidden_size",
    "minibatch_size",
    "learning_rate",
    "max_grad_norm",
    "lbw",
    "best_model_path",
    "final_metadata_path",
)


@dataclass(frozen=True)
class BestCandidateSelection:
    selected_record: SearchCompletionRecord
    best_candidate_path: Path
    best_config_path: Path
    search_summary_report_path: Path


def default_best_candidate_path() -> Path:
    return default_project_root() / DEFAULT_BEST_CANDIDATE_PATH


def default_best_config_path() -> Path:
    return default_project_root() / DEFAULT_BEST_CONFIG_PATH


def default_search_summary_report_path() -> Path:
    return default_project_root() / DEFAULT_SEARCH_SUMMARY_REPORT_PATH


def _resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _candidate_config_from_record(record: SearchCompletionRecord) -> CandidateConfig:
    return record.candidate_config


def _best_loss_as_decimal(record: SearchCompletionRecord) -> Decimal:
    return Decimal(record.best_validation_loss)


def _best_candidate_payload(record: SearchCompletionRecord) -> dict[str, object]:
    return {
        **candidate_config_to_payload(record.candidate_config),
        "status": record.status,
        "best_validation_loss": record.best_validation_loss,
        "best_epoch_index": record.best_epoch_index,
        "epochs_completed": record.epochs_completed,
        "dataset_registry_path": str(record.dataset_registry_path),
        "candidate_dir": str(record.candidate_dir),
        "config_path": str(record.config_path),
        "best_model_path": str(record.best_model_path),
        "epoch_log_path": str(record.epoch_log_path),
        "validation_history_path": str(record.validation_history_path),
        "final_metadata_path": str(record.final_metadata_path),
        "artifacts": {
            "candidate_dir": str(record.candidate_dir),
            "config_path": str(record.config_path),
            "best_model_path": str(record.best_model_path),
            "epoch_log_path": str(record.epoch_log_path),
            "validation_history_path": str(record.validation_history_path),
            "final_metadata_path": str(record.final_metadata_path),
        },
    }


def _write_json(path: Path | str, payload: object) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return json_path


def write_search_summary_report(
    path: Path | str,
    *,
    records: Sequence[SearchCompletionRecord],
    selected_candidate_index: int,
) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(
        records,
        key=lambda record: record.candidate_config.candidate_index,
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(SEARCH_SUMMARY_REPORT_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in sorted_records:
            writer.writerow(
                {
                    "candidate_index": str(record.candidate_config.candidate_index),
                    "candidate_id": record.candidate_config.candidate_id,
                    "status": record.status,
                    "selected": (
                        "true"
                        if record.candidate_config.candidate_index
                        == selected_candidate_index
                        else "false"
                    ),
                    "best_validation_loss": record.best_validation_loss,
                    "failure_reason": record.failure_reason,
                    "dropout": format(record.candidate_config.dropout, ".17g"),
                    "hidden_size": str(record.candidate_config.hidden_size),
                    "minibatch_size": str(record.candidate_config.minibatch_size),
                    "learning_rate": format(
                        record.candidate_config.learning_rate,
                        ".17g",
                    ),
                    "max_grad_norm": format(
                        record.candidate_config.max_grad_norm,
                        ".17g",
                    ),
                    "lbw": str(record.candidate_config.lbw),
                    "best_model_path": str(record.best_model_path),
                    "final_metadata_path": str(record.final_metadata_path),
                }
            )
    return csv_path


def select_best_candidate(
    *,
    completion_log_path: Path | str = default_search_completion_log_path(),
    best_candidate_path: Path | str = default_best_candidate_path(),
    best_config_path: Path | str = default_best_config_path(),
    search_summary_report_path: Path | str = default_search_summary_report_path(),
    project_root: Path | str | None = None,
) -> BestCandidateSelection:
    root = Path(project_root) if project_root is not None else default_project_root()
    resolved_completion_log_path = _resolve_project_path(root, completion_log_path)
    resolved_best_candidate_path = _resolve_project_path(root, best_candidate_path)
    resolved_best_config_path = _resolve_project_path(root, best_config_path)
    resolved_search_summary_report_path = _resolve_project_path(
        root,
        search_summary_report_path,
    )

    records = load_search_completion_log(resolved_completion_log_path)
    if not records:
        raise ValueError("Search completion log is empty")
    successful_records = [
        record for record in records if record.status == COMPLETION_STATUS_COMPLETED
    ]
    if not successful_records:
        raise ValueError("Search completion log has no successful candidates")
    selected_record = min(
        successful_records,
        key=lambda record: (
            _best_loss_as_decimal(record),
            record.candidate_config.candidate_index,
        ),
    )
    _write_json(
        resolved_best_candidate_path,
        _best_candidate_payload(selected_record),
    )
    _write_json(
        resolved_best_config_path,
        candidate_config_to_payload(_candidate_config_from_record(selected_record)),
    )
    write_search_summary_report(
        resolved_search_summary_report_path,
        records=records,
        selected_candidate_index=selected_record.candidate_config.candidate_index,
    )
    return BestCandidateSelection(
        selected_record=selected_record,
        best_candidate_path=resolved_best_candidate_path,
        best_config_path=resolved_best_config_path,
        search_summary_report_path=resolved_search_summary_report_path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the winning completed candidate from the search log."
    )
    parser.add_argument(
        "--completion-log",
        type=Path,
        default=default_search_completion_log_path(),
    )
    parser.add_argument(
        "--best-candidate",
        type=Path,
        default=default_best_candidate_path(),
    )
    parser.add_argument(
        "--best-config",
        type=Path,
        default=default_best_config_path(),
    )
    parser.add_argument(
        "--summary-report",
        type=Path,
        default=default_search_summary_report_path(),
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selection = select_best_candidate(
        completion_log_path=args.completion_log,
        best_candidate_path=args.best_candidate,
        best_config_path=args.best_config,
        search_summary_report_path=args.summary_report,
        project_root=args.project_root,
    )
    print(
        "Selected candidate "
        f"{selection.selected_record.candidate_config.candidate_id} "
        f"with best_validation_loss={selection.selected_record.best_validation_loss}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
