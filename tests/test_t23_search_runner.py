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
    load_search_completion_log,
    run_search_schedule,
)
from lstm_cpd.training.train_candidate import (  # noqa: E402
    CandidateConfig,
    DatasetRegistryEntry,
    TrainingArtifactPaths,
    TrainingRunResult,
    candidate_config_to_payload,
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
        hidden_size=5 * (candidate_index + 1),
        minibatch_size=64,
        learning_rate=1e-3 * (candidate_index + 1),
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
        [["0", "-0.1", f"{best_validation_loss:.6f}", f"{best_validation_loss:.6f}", "0.5", "true"]],
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


class T23SearchRunnerTests(unittest.TestCase):
    def test_runner_resumes_failed_candidates_without_duplicate_log_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            schedule_rel = Path("custom/search_schedule.json")
            dataset_registry_rel = Path(
                "artifacts/interim/manifests/dataset_registry.json"
            )
            search_output_root_rel = Path("artifacts/interim/training")
            schedule_candidates = (
                make_candidate(0, lbw=10),
                make_candidate(1, lbw=21),
                make_candidate(2, lbw=63),
            )
            write_json(
                tmp_path / schedule_rel,
                [
                    candidate_config_to_payload(candidate)
                    for candidate in schedule_candidates
                ],
            )
            write_json(tmp_path / dataset_registry_rel, [])

            invocation_order: list[int] = []
            fail_once = {1: True}
            best_losses = {0: -0.11, 1: -0.27, 2: -0.19}

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
                del artifact_stem, max_epochs, patience
                assert artifact_paths is not None
                invocation_order.append(candidate_config.candidate_index)
                if fail_once.get(candidate_config.candidate_index, False):
                    fail_once[candidate_config.candidate_index] = False
                    raise RuntimeError("simulated candidate failure")
                dataset_registry_path = Path(dataset_registry_path)
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                write_stub_training_artifacts(
                    artifact_paths,
                    candidate_config=candidate_config,
                    dataset_registry_path=dataset_registry_path,
                    best_validation_loss=best_losses[candidate_config.candidate_index],
                )
                return TrainingRunResult(
                    candidate_config=candidate_config,
                    dataset_registry_path=dataset_registry_path,
                    output_dir=output_dir,
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
                first_records = run_search_schedule(
                    schedule_json_path=schedule_rel,
                    dataset_registry_path=dataset_registry_rel,
                    search_output_root=search_output_root_rel,
                    project_root=tmp_path,
                )
                second_records = run_search_schedule(
                    schedule_json_path=schedule_rel,
                    dataset_registry_path=dataset_registry_rel,
                    search_output_root=search_output_root_rel,
                    project_root=tmp_path,
                )

            self.assertEqual(invocation_order[:3], [0, 1, 2])
            self.assertEqual(invocation_order[3:], [1])
            self.assertEqual(
                [record.status for record in first_records],
                [
                    COMPLETION_STATUS_COMPLETED,
                    COMPLETION_STATUS_FAILED,
                    COMPLETION_STATUS_COMPLETED,
                ],
            )
            self.assertEqual(
                [record.status for record in second_records],
                [
                    COMPLETION_STATUS_COMPLETED,
                    COMPLETION_STATUS_COMPLETED,
                    COMPLETION_STATUS_COMPLETED,
                ],
            )

            completion_log_path = (
                tmp_path / search_output_root_rel / "search_completion_log.csv"
            )
            loaded_records = load_search_completion_log(completion_log_path)
            self.assertEqual(len(loaded_records), 3)
            self.assertEqual(
                [record.candidate_config.candidate_index for record in loaded_records],
                [0, 1, 2],
            )
            self.assertTrue(
                all(record.status == COMPLETION_STATUS_COMPLETED for record in loaded_records)
            )

            with completion_log_path.open("r", encoding="utf-8", newline="") as handle:
                completion_rows = list(csv.DictReader(handle))
            self.assertEqual(len(completion_rows), 3)

            for candidate_index in range(3):
                candidate_dir = (
                    tmp_path
                    / search_output_root_rel
                    / "candidates"
                    / f"candidate_{candidate_index:03d}"
                )
                self.assertTrue((candidate_dir / "config.json").exists())
                self.assertTrue((candidate_dir / "best_model.keras").exists())
                self.assertTrue((candidate_dir / "epoch_log.csv").exists())
                self.assertTrue((candidate_dir / "final_metadata.json").exists())
                self.assertTrue((candidate_dir / "validation_history.csv").exists())

    def test_runner_writes_failure_metadata_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            schedule_path = tmp_path / "schedule.json"
            dataset_registry_path = tmp_path / "artifacts/interim/manifests/dataset_registry.json"
            candidates = (
                make_candidate(0, lbw=10),
                make_candidate(1, lbw=21),
            )
            write_json(
                schedule_path,
                [candidate_config_to_payload(candidate) for candidate in candidates],
            )
            write_json(dataset_registry_path, [])

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
                if candidate_config.candidate_index == 0:
                    raise RuntimeError("boom")
                write_stub_training_artifacts(
                    artifact_paths,
                    candidate_config=candidate_config,
                    dataset_registry_path=resolved_dataset_registry_path,
                    best_validation_loss=-0.2,
                )
                return TrainingRunResult(
                    candidate_config=candidate_config,
                    dataset_registry_path=resolved_dataset_registry_path,
                    output_dir=resolved_output_dir,
                    config_snapshot_path=artifact_paths.config_snapshot_path,
                    best_model_path=artifact_paths.best_model_path,
                    epoch_log_path=artifact_paths.epoch_log_path,
                    validation_history_path=artifact_paths.validation_history_path,
                    initial_validation_loss=-0.1,
                    best_validation_loss=-0.2,
                    best_epoch_index=0,
                    epochs_completed=1,
                    validation_losses=(-0.2,),
                    dataset_entry=make_dataset_entry(candidate_config.lbw),
                )

            with patch(
                "lstm_cpd.training.search_runner.run_candidate_training",
                side_effect=stub_run_candidate_training,
            ):
                records = run_search_schedule(
                    schedule_json_path=schedule_path,
                    dataset_registry_path=dataset_registry_path,
                    search_output_root=tmp_path / "artifacts/interim/training",
                    project_root=tmp_path,
                )

            self.assertEqual(
                [record.status for record in records],
                [COMPLETION_STATUS_FAILED, COMPLETION_STATUS_COMPLETED],
            )
            failure_payload = json.loads(
                (
                    tmp_path
                    / "artifacts/interim/training/candidates/candidate_000/final_metadata.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(failure_payload["status"], COMPLETION_STATUS_FAILED)
            self.assertIn("boom", failure_payload.get("failure_reason", ""))


if __name__ == "__main__":
    unittest.main()
