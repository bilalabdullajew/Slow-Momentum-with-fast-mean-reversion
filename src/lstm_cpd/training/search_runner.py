from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.training.search_schedule import (
    default_search_schedule_json_path,
    load_search_schedule,
)
from lstm_cpd.training.train_candidate import (
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    CandidateConfig,
    DatasetRegistryEntry,
    TrainingArtifactPaths,
    TrainingRunResult,
    candidate_config_to_payload,
    load_candidate_config,
    run_candidate_training,
    validate_candidate_config,
)


DEFAULT_SEARCH_OUTPUT_ROOT = "artifacts/training"
DEFAULT_SEARCH_COMPLETION_LOG_PATH = "artifacts/training/search_completion_log.csv"
COMPLETION_STATUS_COMPLETED = "completed"
COMPLETION_STATUS_FAILED = "failed"
SEARCH_COMPLETION_LOG_HEADER = (
    "candidate_index",
    "candidate_id",
    "status",
    "failure_reason",
    "dropout",
    "hidden_size",
    "minibatch_size",
    "learning_rate",
    "max_grad_norm",
    "lbw",
    "best_validation_loss",
    "best_epoch_index",
    "epochs_completed",
    "dataset_registry_path",
    "candidate_dir",
    "config_path",
    "best_model_path",
    "epoch_log_path",
    "validation_history_path",
    "final_metadata_path",
)


@dataclass(frozen=True)
class SearchCandidateArtifactPaths:
    candidate_dir: Path
    training_artifacts: TrainingArtifactPaths
    final_metadata_path: Path


@dataclass(frozen=True)
class SearchCompletionRecord:
    candidate_config: CandidateConfig
    status: str
    failure_reason: str
    best_validation_loss: str
    best_epoch_index: str
    epochs_completed: str
    dataset_registry_path: Path
    candidate_dir: Path
    config_path: Path
    best_model_path: Path
    epoch_log_path: Path
    validation_history_path: Path
    final_metadata_path: Path


def default_search_output_root() -> Path:
    return default_project_root() / DEFAULT_SEARCH_OUTPUT_ROOT


def default_search_completion_log_path() -> Path:
    return default_project_root() / DEFAULT_SEARCH_COMPLETION_LOG_PATH


def _resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _serialize_float(value: float) -> str:
    return format(value, ".17g")


def _dataset_entry_to_payload(entry: DatasetRegistryEntry) -> dict[str, object]:
    return {
        "lbw": entry.lbw,
        "feature_columns": list(entry.feature_columns),
        "sequence_length": entry.sequence_length,
        "train_sequence_count": entry.train_sequence_count,
        "val_sequence_count": entry.val_sequence_count,
        "train_input_shape": list(entry.train_input_shape),
        "train_target_shape": list(entry.train_target_shape),
        "val_input_shape": list(entry.val_input_shape),
        "val_target_shape": list(entry.val_target_shape),
        "artifacts": {
            "train_inputs_path": entry.train_inputs_path,
            "train_target_scale_path": entry.train_target_scale_path,
            "val_inputs_path": entry.val_inputs_path,
            "val_target_scale_path": entry.val_target_scale_path,
            "train_sequence_index_path": entry.train_sequence_index_path,
            "val_sequence_index_path": entry.val_sequence_index_path,
        },
        "source_artifacts": {
            "split_manifest_path": entry.split_manifest_path,
            "sequence_manifest_path": entry.sequence_manifest_path,
            "target_alignment_registry_path": entry.target_alignment_registry_path,
        },
    }


def build_search_candidate_artifact_paths(
    search_output_root: Path | str,
    *,
    candidate_index: int,
) -> SearchCandidateArtifactPaths:
    candidate_dir = Path(search_output_root) / "candidates" / f"candidate_{candidate_index:03d}"
    return SearchCandidateArtifactPaths(
        candidate_dir=candidate_dir,
        training_artifacts=TrainingArtifactPaths(
            config_snapshot_path=candidate_dir / "config.json",
            best_model_path=candidate_dir / "best_model.keras",
            epoch_log_path=candidate_dir / "epoch_log.csv",
            validation_history_path=candidate_dir / "validation_history.csv",
        ),
        final_metadata_path=candidate_dir / "final_metadata.json",
    )


def _write_json(path: Path | str, payload: object) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return json_path


def _build_success_metadata_payload(
    training_result: TrainingRunResult,
    artifacts: SearchCandidateArtifactPaths,
) -> dict[str, object]:
    return {
        "status": COMPLETION_STATUS_COMPLETED,
        **candidate_config_to_payload(training_result.candidate_config),
        "dataset_registry_path": str(training_result.dataset_registry_path),
        "output_dir": str(training_result.output_dir),
        "initial_validation_loss": _serialize_float(
            training_result.initial_validation_loss
        ),
        "best_validation_loss": _serialize_float(training_result.best_validation_loss),
        "best_epoch_index": training_result.best_epoch_index,
        "epochs_completed": training_result.epochs_completed,
        "validation_losses": [
            _serialize_float(loss_value)
            for loss_value in training_result.validation_losses
        ],
        "dataset_entry": _dataset_entry_to_payload(training_result.dataset_entry),
        "artifacts": {
            "candidate_dir": str(artifacts.candidate_dir),
            "config_path": str(training_result.config_snapshot_path),
            "best_model_path": str(training_result.best_model_path),
            "epoch_log_path": str(training_result.epoch_log_path),
            "validation_history_path": str(training_result.validation_history_path),
            "final_metadata_path": str(artifacts.final_metadata_path),
        },
    }


def _write_failure_metadata(
    *,
    candidate_config: CandidateConfig,
    dataset_registry_path: Path,
    artifacts: SearchCandidateArtifactPaths,
    failure_reason: str,
) -> Path:
    payload = {
        "status": COMPLETION_STATUS_FAILED,
        **candidate_config_to_payload(candidate_config),
        "dataset_registry_path": str(dataset_registry_path),
        "failure_reason": failure_reason,
        "artifacts": {
            "candidate_dir": str(artifacts.candidate_dir),
            "config_path": str(artifacts.training_artifacts.config_snapshot_path),
            "best_model_path": str(artifacts.training_artifacts.best_model_path),
            "epoch_log_path": str(artifacts.training_artifacts.epoch_log_path),
            "validation_history_path": str(
                artifacts.training_artifacts.validation_history_path
            ),
            "final_metadata_path": str(artifacts.final_metadata_path),
        },
    }
    return _write_json(artifacts.final_metadata_path, payload)


def _build_success_completion_record(
    training_result: TrainingRunResult,
    artifacts: SearchCandidateArtifactPaths,
) -> SearchCompletionRecord:
    return SearchCompletionRecord(
        candidate_config=training_result.candidate_config,
        status=COMPLETION_STATUS_COMPLETED,
        failure_reason="",
        best_validation_loss=_serialize_float(training_result.best_validation_loss),
        best_epoch_index=(
            ""
            if training_result.best_epoch_index is None
            else str(training_result.best_epoch_index)
        ),
        epochs_completed=str(training_result.epochs_completed),
        dataset_registry_path=training_result.dataset_registry_path,
        candidate_dir=artifacts.candidate_dir,
        config_path=training_result.config_snapshot_path,
        best_model_path=training_result.best_model_path,
        epoch_log_path=training_result.epoch_log_path,
        validation_history_path=training_result.validation_history_path,
        final_metadata_path=artifacts.final_metadata_path,
    )


def _build_failure_completion_record(
    *,
    candidate_config: CandidateConfig,
    dataset_registry_path: Path,
    artifacts: SearchCandidateArtifactPaths,
    failure_reason: str,
) -> SearchCompletionRecord:
    return SearchCompletionRecord(
        candidate_config=candidate_config,
        status=COMPLETION_STATUS_FAILED,
        failure_reason=failure_reason,
        best_validation_loss="",
        best_epoch_index="",
        epochs_completed="",
        dataset_registry_path=dataset_registry_path,
        candidate_dir=artifacts.candidate_dir,
        config_path=artifacts.training_artifacts.config_snapshot_path,
        best_model_path=artifacts.training_artifacts.best_model_path,
        epoch_log_path=artifacts.training_artifacts.epoch_log_path,
        validation_history_path=artifacts.training_artifacts.validation_history_path,
        final_metadata_path=artifacts.final_metadata_path,
    )


def _search_completion_record_to_row(
    record: SearchCompletionRecord,
) -> dict[str, str]:
    return {
        "candidate_index": str(record.candidate_config.candidate_index),
        "candidate_id": record.candidate_config.candidate_id,
        "status": record.status,
        "failure_reason": record.failure_reason,
        "dropout": _serialize_float(record.candidate_config.dropout),
        "hidden_size": str(record.candidate_config.hidden_size),
        "minibatch_size": str(record.candidate_config.minibatch_size),
        "learning_rate": _serialize_float(record.candidate_config.learning_rate),
        "max_grad_norm": _serialize_float(record.candidate_config.max_grad_norm),
        "lbw": str(record.candidate_config.lbw),
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
    }


def load_search_completion_log(path: Path | str) -> tuple[SearchCompletionRecord, ...]:
    csv_path = Path(path)
    if not csv_path.exists():
        return tuple()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    records: list[SearchCompletionRecord] = []
    seen_indices: set[int] = set()
    for row in rows:
        candidate_config = CandidateConfig(
            candidate_id=str(row["candidate_id"]),
            candidate_index=int(row["candidate_index"]),
            dropout=float(row["dropout"]),
            hidden_size=int(row["hidden_size"]),
            minibatch_size=int(row["minibatch_size"]),
            learning_rate=float(row["learning_rate"]),
            max_grad_norm=float(row["max_grad_norm"]),
            lbw=int(row["lbw"]),
        )
        validate_candidate_config(candidate_config)
        if candidate_config.candidate_index in seen_indices:
            raise ValueError(
                "search_completion_log.csv contains duplicate candidate_index rows"
            )
        seen_indices.add(candidate_config.candidate_index)
        records.append(
            SearchCompletionRecord(
                candidate_config=candidate_config,
                status=str(row["status"]),
                failure_reason=str(row["failure_reason"]),
                best_validation_loss=str(row["best_validation_loss"]),
                best_epoch_index=str(row["best_epoch_index"]),
                epochs_completed=str(row["epochs_completed"]),
                dataset_registry_path=Path(row["dataset_registry_path"]),
                candidate_dir=Path(row["candidate_dir"]),
                config_path=Path(row["config_path"]),
                best_model_path=Path(row["best_model_path"]),
                epoch_log_path=Path(row["epoch_log_path"]),
                validation_history_path=Path(row["validation_history_path"]),
                final_metadata_path=Path(row["final_metadata_path"]),
            )
        )
    return tuple(sorted(records, key=lambda record: record.candidate_config.candidate_index))


def write_search_completion_log(
    path: Path | str,
    records: Sequence[SearchCompletionRecord],
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
            fieldnames=list(SEARCH_COMPLETION_LOG_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in sorted_records:
            writer.writerow(_search_completion_record_to_row(record))
    return csv_path


def _config_snapshot_matches_schedule(
    *,
    config_path: Path,
    candidate_config: CandidateConfig,
    dataset_registry_path: Path,
) -> bool:
    if not config_path.exists():
        return False
    try:
        loaded_config = load_candidate_config(config_path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if loaded_config != candidate_config:
        return False
    return str(payload.get("dataset_registry_path", "")) == str(dataset_registry_path)


def _completed_candidate_is_intact(
    *,
    record: SearchCompletionRecord,
    candidate_config: CandidateConfig,
    dataset_registry_path: Path,
    artifacts: SearchCandidateArtifactPaths,
) -> bool:
    if record.status != COMPLETION_STATUS_COMPLETED:
        return False
    if record.candidate_config != candidate_config:
        return False
    if record.dataset_registry_path != dataset_registry_path:
        return False
    if record.candidate_dir != artifacts.candidate_dir:
        return False
    if record.config_path != artifacts.training_artifacts.config_snapshot_path:
        return False
    if record.best_model_path != artifacts.training_artifacts.best_model_path:
        return False
    if record.epoch_log_path != artifacts.training_artifacts.epoch_log_path:
        return False
    if (
        record.validation_history_path
        != artifacts.training_artifacts.validation_history_path
    ):
        return False
    if record.final_metadata_path != artifacts.final_metadata_path:
        return False
    for required_path in (
        record.config_path,
        record.best_model_path,
        record.epoch_log_path,
        record.validation_history_path,
        record.final_metadata_path,
    ):
        if not required_path.exists():
            return False
    return _config_snapshot_matches_schedule(
        config_path=record.config_path,
        candidate_config=candidate_config,
        dataset_registry_path=dataset_registry_path,
    )


def run_search_schedule(
    *,
    schedule_json_path: Path | str = default_search_schedule_json_path(),
    dataset_registry_path: Path | str,
    search_output_root: Path | str = default_search_output_root(),
    completion_log_path: Path | str | None = None,
    project_root: Path | str | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
) -> tuple[SearchCompletionRecord, ...]:
    root = Path(project_root) if project_root is not None else default_project_root()
    resolved_schedule_json_path = _resolve_project_path(root, schedule_json_path)
    resolved_dataset_registry_path = _resolve_project_path(root, dataset_registry_path)
    resolved_search_output_root = _resolve_project_path(root, search_output_root)
    resolved_completion_log_path = (
        _resolve_project_path(root, completion_log_path)
        if completion_log_path is not None
        else resolved_search_output_root / "search_completion_log.csv"
    )

    schedule = load_search_schedule(resolved_schedule_json_path)
    schedule_indices = {candidate.candidate_index for candidate in schedule}
    records_by_index = {
        record.candidate_config.candidate_index: record
        for record in load_search_completion_log(resolved_completion_log_path)
    }
    unexpected_indices = set(records_by_index) - schedule_indices
    if unexpected_indices:
        raise ValueError(
            "Completion log contains candidate indices outside the active schedule: "
            f"{sorted(unexpected_indices)}"
        )

    for candidate_config in schedule:
        artifacts = build_search_candidate_artifact_paths(
            resolved_search_output_root,
            candidate_index=candidate_config.candidate_index,
        )
        existing_record = records_by_index.get(candidate_config.candidate_index)
        if existing_record is not None and _completed_candidate_is_intact(
            record=existing_record,
            candidate_config=candidate_config,
            dataset_registry_path=resolved_dataset_registry_path,
            artifacts=artifacts,
        ):
            records_by_index[candidate_config.candidate_index] = existing_record
            write_search_completion_log(
                resolved_completion_log_path,
                tuple(records_by_index.values()),
            )
            continue

        try:
            training_result = run_candidate_training(
                dataset_registry_path=resolved_dataset_registry_path,
                candidate_config=candidate_config,
                output_dir=artifacts.candidate_dir,
                project_root=root,
                artifact_paths=artifacts.training_artifacts,
                max_epochs=max_epochs,
                patience=patience,
            )
            _write_json(
                artifacts.final_metadata_path,
                _build_success_metadata_payload(training_result, artifacts),
            )
            records_by_index[candidate_config.candidate_index] = (
                _build_success_completion_record(training_result, artifacts)
            )
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            _write_failure_metadata(
                candidate_config=candidate_config,
                dataset_registry_path=resolved_dataset_registry_path,
                artifacts=artifacts,
                failure_reason=failure_reason,
            )
            records_by_index[candidate_config.candidate_index] = (
                _build_failure_completion_record(
                    candidate_config=candidate_config,
                    dataset_registry_path=resolved_dataset_registry_path,
                    artifacts=artifacts,
                    failure_reason=failure_reason,
                )
            )

        write_search_completion_log(
            resolved_completion_log_path,
            tuple(records_by_index.values()),
        )

    return tuple(
        sorted(
            records_by_index.values(),
            key=lambda record: record.candidate_config.candidate_index,
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the immutable search schedule sequentially."
    )
    parser.add_argument(
        "--schedule-json",
        type=Path,
        default=default_search_schedule_json_path(),
    )
    parser.add_argument("--dataset-registry", type=Path, required=True)
    parser.add_argument(
        "--search-output-root",
        type=Path,
        default=default_search_output_root(),
    )
    parser.add_argument("--completion-log", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = run_search_schedule(
        schedule_json_path=args.schedule_json,
        dataset_registry_path=args.dataset_registry,
        search_output_root=args.search_output_root,
        completion_log_path=args.completion_log,
        project_root=args.project_root,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    completed_count = sum(
        1 for record in records if record.status == COMPLETION_STATUS_COMPLETED
    )
    failed_count = sum(
        1 for record in records if record.status == COMPLETION_STATUS_FAILED
    )
    print(
        "Search schedule processed with "
        f"{completed_count} completed and {failed_count} failed candidates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
