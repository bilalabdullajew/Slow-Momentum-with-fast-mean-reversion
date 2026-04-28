from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.training.search_runner import (  # noqa: E402
    COMPLETION_STATUS_COMPLETED,
    COMPLETION_STATUS_FAILED,
    SearchCompletionRecord,
    run_search_schedule,
    write_search_completion_log,
)
from lstm_cpd.training.selection import select_best_candidate  # noqa: E402
from lstm_cpd.training.train_candidate import (  # noqa: E402
    CandidateConfig,
    DatasetRegistryEntry,
    TrainingArtifactPaths,
    TrainingRunResult,
    candidate_config_to_payload,
    load_candidate_config,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def make_candidate(candidate_index: int, *, lbw: int) -> CandidateConfig:
    return CandidateConfig(
        candidate_id=f"C-{candidate_index + 1:03d}",
        candidate_index=candidate_index,
        dropout=0.1 + (0.1 * candidate_index),
        hidden_size=10 * (candidate_index + 1),
        minibatch_size=64,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        lbw=lbw,
    )


def make_dataset_entry(lbw: int) -> DatasetRegistryEntry:
    return DatasetRegistryEntry(
        lbw=lbw,
        feature_columns=(
            "normalized_return_1",
            "normalized_return_21",
            "normalized_return_63",
            "normalized_return_126",
            "normalized_return_256",
            "macd_8_24",
            "macd_16_28",
            "macd_32_96",
            "nu",
            "gamma",
        ),
        sequence_length=63,
        train_sequence_count=8,
        val_sequence_count=4,
        train_input_shape=(8, 63, 10),
        train_target_shape=(8, 63),
        val_input_shape=(4, 63, 10),
        val_target_shape=(4, 63),
        train_inputs_path=f"artifacts/interim/datasets/lbw_{lbw}/train_inputs.npy",
        train_target_scale_path=f"artifacts/interim/datasets/lbw_{lbw}/train_target_scale.npy",
        val_inputs_path=f"artifacts/interim/datasets/lbw_{lbw}/val_inputs.npy",
        val_target_scale_path=f"artifacts/interim/datasets/lbw_{lbw}/val_target_scale.npy",
        train_sequence_index_path=f"artifacts/interim/datasets/lbw_{lbw}/train_sequence_index.csv",
        val_sequence_index_path=f"artifacts/interim/datasets/lbw_{lbw}/val_sequence_index.csv",
        split_manifest_path=f"artifacts/interim/datasets/lbw_{lbw}/split_manifest.csv",
        sequence_manifest_path=f"artifacts/interim/datasets/lbw_{lbw}/sequence_manifest.csv",
        target_alignment_registry_path=(
            f"artifacts/interim/datasets/lbw_{lbw}/target_alignment_registry.csv"
        ),
    )


def write_stub_training_artifacts(
    artifact_paths: TrainingArtifactPaths,
    *,
    candidate_config: CandidateConfig,
    dataset_registry_path: Path,
    best_validation_loss: float,
) -> None:
    for path in (
        artifact_paths.config_snapshot_path,
        artifact_paths.best_model_path,
        artifact_paths.epoch_log_path,
        artifact_paths.validation_history_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        artifact_paths.config_snapshot_path,
        {
            **candidate_config_to_payload(candidate_config),
            "dataset_registry_path": str(dataset_registry_path),
        },
    )
    artifact_paths.best_model_path.write_text("stub checkpoint\n", encoding="utf-8")
    write_csv(
        artifact_paths.epoch_log_path,
        (
            "epoch_index",
            "train_loss",
            "val_loss",
            "best_val_loss",
            "mean_gradient_norm",
            "improved",
        ),
        [["0", "-0.1", f"{best_validation_loss:.6f}", f"{best_validation_loss:.6f}", "0.4", "true"]],
    )
    write_csv(
        artifact_paths.validation_history_path,
        (
            "epoch_index",
            "val_loss",
            "best_so_far",
            "improved_vs_previous",
            "improved_vs_best",
        ),
        [["0", f"{best_validation_loss:.6f}", f"{best_validation_loss:.6f}", "true", "true"]],
    )


def make_completion_record(
    tmp_path: Path,
    *,
    candidate_index: int,
    lbw: int,
    status: str,
    best_validation_loss: str,
    failure_reason: str = "",
) -> SearchCompletionRecord:
    candidate_config = make_candidate(candidate_index, lbw=lbw)
    candidate_dir = (
        tmp_path / "artifacts/training/candidates" / f"candidate_{candidate_index:03d}"
    )
    config_path = candidate_dir / "config.json"
    best_model_path = candidate_dir / "best_model.keras"
    epoch_log_path = candidate_dir / "epoch_log.csv"
    validation_history_path = candidate_dir / "validation_history.csv"
    final_metadata_path = candidate_dir / "final_metadata.json"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        config_path,
        {
            **candidate_config_to_payload(candidate_config),
            "dataset_registry_path": str(tmp_path / "artifacts/interim/manifests/dataset_registry.json"),
        },
    )
    best_model_path.write_text("checkpoint\n", encoding="utf-8")
    epoch_log_path.write_text("epoch_index,val_loss\n0,-0.1\n", encoding="utf-8")
    validation_history_path.write_text(
        "epoch_index,val_loss\n0,-0.1\n",
        encoding="utf-8",
    )
    write_json(
        final_metadata_path,
        {
            "status": status,
            **candidate_config_to_payload(candidate_config),
        },
    )
    return SearchCompletionRecord(
        candidate_config=candidate_config,
        status=status,
        failure_reason=failure_reason,
        best_validation_loss=best_validation_loss,
        best_epoch_index="0" if status == COMPLETION_STATUS_COMPLETED else "",
        epochs_completed="1" if status == COMPLETION_STATUS_COMPLETED else "",
        dataset_registry_path=tmp_path / "artifacts/interim/manifests/dataset_registry.json",
        candidate_dir=candidate_dir,
        config_path=config_path,
        best_model_path=best_model_path,
        epoch_log_path=epoch_log_path,
        validation_history_path=validation_history_path,
        final_metadata_path=final_metadata_path,
    )


class T24SelectionTests(unittest.TestCase):
    def test_selection_prefers_minimum_loss_and_smaller_index_on_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            records = (
                make_completion_record(
                    tmp_path,
                    candidate_index=0,
                    lbw=10,
                    status=COMPLETION_STATUS_COMPLETED,
                    best_validation_loss="-0.50000000000000000",
                ),
                make_completion_record(
                    tmp_path,
                    candidate_index=1,
                    lbw=21,
                    status=COMPLETION_STATUS_COMPLETED,
                    best_validation_loss="-0.50000000000000000",
                ),
                make_completion_record(
                    tmp_path,
                    candidate_index=2,
                    lbw=63,
                    status=COMPLETION_STATUS_COMPLETED,
                    best_validation_loss="-0.41000000000000003",
                ),
                make_completion_record(
                    tmp_path,
                    candidate_index=3,
                    lbw=126,
                    status=COMPLETION_STATUS_FAILED,
                    best_validation_loss="",
                    failure_reason="RuntimeError: boom",
                ),
            )
            completion_log_path = tmp_path / "artifacts/training/search_completion_log.csv"
            write_search_completion_log(completion_log_path, records)

            selection = select_best_candidate(
                completion_log_path=completion_log_path,
                best_candidate_path=tmp_path / "artifacts/training/best_candidate.json",
                best_config_path=tmp_path / "artifacts/training/best_config.json",
                search_summary_report_path=tmp_path / "artifacts/reports/search_summary_report.csv",
                project_root=tmp_path,
            )

            self.assertEqual(selection.selected_record.candidate_config.candidate_index, 0)
            best_candidate_payload = json.loads(
                selection.best_candidate_path.read_text(encoding="utf-8")
            )
            self.assertEqual(best_candidate_payload["candidate_index"], 0)
            self.assertEqual(
                best_candidate_payload["best_model_path"],
                str(records[0].best_model_path),
            )

            loaded_best_config = load_candidate_config(selection.best_config_path)
            self.assertEqual(loaded_best_config, records[0].candidate_config)

            with selection.search_summary_report_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 4)
            self.assertEqual(summary_rows[0]["selected"], "true")
            self.assertEqual(summary_rows[3]["status"], COMPLETION_STATUS_FAILED)
            self.assertEqual(summary_rows[3]["selected"], "false")

    def test_selection_fails_when_no_successful_candidates_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            records = (
                make_completion_record(
                    tmp_path,
                    candidate_index=0,
                    lbw=10,
                    status=COMPLETION_STATUS_FAILED,
                    best_validation_loss="",
                    failure_reason="RuntimeError: no data",
                ),
                make_completion_record(
                    tmp_path,
                    candidate_index=1,
                    lbw=21,
                    status=COMPLETION_STATUS_FAILED,
                    best_validation_loss="",
                    failure_reason="RuntimeError: diverged",
                ),
            )
            completion_log_path = tmp_path / "artifacts/training/search_completion_log.csv"
            write_search_completion_log(completion_log_path, records)

            with self.assertRaisesRegex(
                ValueError,
                "no successful candidates",
            ):
                select_best_candidate(
                    completion_log_path=completion_log_path,
                    project_root=tmp_path,
                )

    def test_end_to_end_stubbed_search_to_selection_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            schedule_rel = Path("custom/search_schedule.json")
            dataset_registry_rel = Path(
                "artifacts/interim/manifests/dataset_registry.json"
            )
            search_output_root_rel = Path("artifacts/interim/training")
            report_rel = Path("artifacts/interim/reports/search_summary_report.csv")
            candidates = (
                make_candidate(0, lbw=10),
                make_candidate(1, lbw=63),
            )
            write_json(
                tmp_path / schedule_rel,
                [candidate_config_to_payload(candidate) for candidate in candidates],
            )
            write_json(tmp_path / dataset_registry_rel, [])
            best_losses = {0: -0.14, 1: -0.22}

            def stub_run_candidate_training(
                *,
                dataset_registry_path: Path | str,
                candidate_config: CandidateConfig,
                output_dir: Path | str,
                project_root: Path | str | None = None,
                artifact_stem: str = "candidate",
                artifact_paths: TrainingArtifactPaths | None = None,
                max_epochs: int = 300,
                patience: int = 25,
            ) -> TrainingRunResult:
                del project_root, artifact_stem, max_epochs, patience
                assert artifact_paths is not None
                resolved_dataset_registry_path = Path(dataset_registry_path)
                resolved_output_dir = Path(output_dir)
                resolved_output_dir.mkdir(parents=True, exist_ok=True)
                write_stub_training_artifacts(
                    artifact_paths,
                    candidate_config=candidate_config,
                    dataset_registry_path=resolved_dataset_registry_path,
                    best_validation_loss=best_losses[candidate_config.candidate_index],
                )
                return TrainingRunResult(
                    candidate_config=candidate_config,
                    dataset_registry_path=resolved_dataset_registry_path,
                    output_dir=resolved_output_dir,
                    config_snapshot_path=artifact_paths.config_snapshot_path,
                    best_model_path=artifact_paths.best_model_path,
                    epoch_log_path=artifact_paths.epoch_log_path,
                    validation_history_path=artifact_paths.validation_history_path,
                    initial_validation_loss=best_losses[candidate_config.candidate_index]
                    + 0.1,
                    best_validation_loss=best_losses[candidate_config.candidate_index],
                    best_epoch_index=0,
                    epochs_completed=1,
                    validation_losses=(best_losses[candidate_config.candidate_index],),
                    dataset_entry=make_dataset_entry(candidate_config.lbw),
                )

            with patch(
                "lstm_cpd.training.search_runner.run_candidate_training",
                side_effect=stub_run_candidate_training,
            ):
                run_search_schedule(
                    schedule_json_path=schedule_rel,
                    dataset_registry_path=dataset_registry_rel,
                    search_output_root=search_output_root_rel,
                    project_root=tmp_path,
                )

            selection = select_best_candidate(
                completion_log_path=tmp_path
                / search_output_root_rel
                / "search_completion_log.csv",
                best_candidate_path=tmp_path / "artifacts/interim/training/best_candidate.json",
                best_config_path=tmp_path / "artifacts/interim/training/best_config.json",
                search_summary_report_path=tmp_path / report_rel,
                project_root=tmp_path,
            )

            self.assertEqual(selection.selected_record.candidate_config.candidate_index, 1)
            best_candidate_payload = json.loads(
                selection.best_candidate_path.read_text(encoding="utf-8")
            )
            self.assertEqual(best_candidate_payload["candidate_id"], "C-002")
            self.assertTrue(
                best_candidate_payload["best_model_path"].endswith(
                    "candidate_001/best_model.keras"
                )
            )


if __name__ == "__main__":
    unittest.main()
