from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.training.train_candidate import (  # noqa: E402
    CandidateConfig,
    run_candidate_training,
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


def make_synthetic_dataset(sequence_count: int) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.zeros((sequence_count, 63, 10), dtype=np.float32)
    target_scale = np.zeros((sequence_count, 63), dtype=np.float32)
    for sequence_index in range(sequence_count):
        feature_value = 1.0 if sequence_index % 2 == 0 else -1.0
        inputs[sequence_index, :, 0] = feature_value
        inputs[sequence_index, :, 1] = np.linspace(0.0, 1.0, 63, dtype=np.float32)
        target_scale[sequence_index, :] = 0.05 * feature_value
    return inputs, target_scale


class T20TrainCandidateTests(unittest.TestCase):
    def test_runner_persists_checkpoint_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_inputs, train_targets = make_synthetic_dataset(8)
            val_inputs, val_targets = make_synthetic_dataset(4)
            dataset_root = tmp_path / "artifacts/interim/datasets/lbw_63"
            dataset_root.mkdir(parents=True, exist_ok=True)
            np.save(dataset_root / "train_inputs.npy", train_inputs)
            np.save(dataset_root / "train_target_scale.npy", train_targets)
            np.save(dataset_root / "val_inputs.npy", val_inputs)
            np.save(dataset_root / "val_target_scale.npy", val_targets)
            write_csv(
                dataset_root / "train_sequence_index.csv",
                (
                    "array_row_index",
                    "sequence_id",
                    "asset_id",
                    "lbw",
                    "split",
                    "start_timestamp",
                    "end_timestamp",
                    "start_timeline_index",
                    "end_timeline_index",
                ),
                [],
            )
            write_csv(
                dataset_root / "val_sequence_index.csv",
                (
                    "array_row_index",
                    "sequence_id",
                    "asset_id",
                    "lbw",
                    "split",
                    "start_timestamp",
                    "end_timestamp",
                    "start_timeline_index",
                    "end_timeline_index",
                ),
                [],
            )
            registry_path = tmp_path / "artifacts/interim/manifests/dataset_registry.json"
            write_json(
                registry_path,
                [
                    {
                        "lbw": 63,
                        "feature_columns": [
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
                        ],
                        "sequence_length": 63,
                        "train_sequence_count": 8,
                        "val_sequence_count": 4,
                        "train_input_shape": [8, 63, 10],
                        "train_target_shape": [8, 63],
                        "val_input_shape": [4, 63, 10],
                        "val_target_shape": [4, 63],
                        "artifacts": {
                            "train_inputs_path": "artifacts/interim/datasets/lbw_63/train_inputs.npy",
                            "train_target_scale_path": "artifacts/interim/datasets/lbw_63/train_target_scale.npy",
                            "val_inputs_path": "artifacts/interim/datasets/lbw_63/val_inputs.npy",
                            "val_target_scale_path": "artifacts/interim/datasets/lbw_63/val_target_scale.npy",
                            "train_sequence_index_path": "artifacts/interim/datasets/lbw_63/train_sequence_index.csv",
                            "val_sequence_index_path": "artifacts/interim/datasets/lbw_63/val_sequence_index.csv",
                        },
                        "source_artifacts": {
                            "split_manifest_path": "artifacts/interim/datasets/lbw_63/split_manifest.csv",
                            "sequence_manifest_path": "artifacts/interim/datasets/lbw_63/sequence_manifest.csv",
                            "target_alignment_registry_path": "artifacts/interim/datasets/lbw_63/target_alignment_registry.csv",
                        },
                    }
                ],
            )
            candidate_config = CandidateConfig(
                candidate_id="TEST",
                candidate_index=0,
                dropout=0.1,
                hidden_size=5,
                minibatch_size=2,
                learning_rate=1e-2,
                max_grad_norm=1.0,
                lbw=63,
            )

            result = run_candidate_training(
                dataset_registry_path=registry_path,
                candidate_config=candidate_config,
                output_dir=tmp_path / "artifacts/interim/training/run_one",
                project_root=tmp_path,
                artifact_stem="smoke",
                max_epochs=4,
                patience=2,
            )

            self.assertTrue(result.best_model_path.exists())
            self.assertTrue(result.config_snapshot_path.exists())
            self.assertTrue(result.epoch_log_path.exists())
            self.assertTrue(result.validation_history_path.exists())
            self.assertGreater(result.epochs_completed, 0)

            with result.epoch_log_path.open("r", encoding="utf-8", newline="") as handle:
                epoch_rows = list(csv.DictReader(handle))
            self.assertEqual(len(epoch_rows), result.epochs_completed)
            loaded_model = tf.keras.models.load_model(
                result.best_model_path,
                compile=False,
            )
            self.assertEqual(loaded_model.output_shape, (None, 63, 1))

    def test_runner_is_deterministic_for_identical_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_inputs, train_targets = make_synthetic_dataset(8)
            val_inputs, val_targets = make_synthetic_dataset(4)
            dataset_root = tmp_path / "artifacts/interim/datasets/lbw_63"
            dataset_root.mkdir(parents=True, exist_ok=True)
            np.save(dataset_root / "train_inputs.npy", train_inputs)
            np.save(dataset_root / "train_target_scale.npy", train_targets)
            np.save(dataset_root / "val_inputs.npy", val_inputs)
            np.save(dataset_root / "val_target_scale.npy", val_targets)
            write_csv(
                dataset_root / "train_sequence_index.csv",
                (
                    "array_row_index",
                    "sequence_id",
                    "asset_id",
                    "lbw",
                    "split",
                    "start_timestamp",
                    "end_timestamp",
                    "start_timeline_index",
                    "end_timeline_index",
                ),
                [],
            )
            write_csv(
                dataset_root / "val_sequence_index.csv",
                (
                    "array_row_index",
                    "sequence_id",
                    "asset_id",
                    "lbw",
                    "split",
                    "start_timestamp",
                    "end_timestamp",
                    "start_timeline_index",
                    "end_timeline_index",
                ),
                [],
            )
            registry_path = tmp_path / "artifacts/interim/manifests/dataset_registry.json"
            write_json(
                registry_path,
                [
                    {
                        "lbw": 63,
                        "feature_columns": [
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
                        ],
                        "sequence_length": 63,
                        "train_sequence_count": 8,
                        "val_sequence_count": 4,
                        "train_input_shape": [8, 63, 10],
                        "train_target_shape": [8, 63],
                        "val_input_shape": [4, 63, 10],
                        "val_target_shape": [4, 63],
                        "artifacts": {
                            "train_inputs_path": "artifacts/interim/datasets/lbw_63/train_inputs.npy",
                            "train_target_scale_path": "artifacts/interim/datasets/lbw_63/train_target_scale.npy",
                            "val_inputs_path": "artifacts/interim/datasets/lbw_63/val_inputs.npy",
                            "val_target_scale_path": "artifacts/interim/datasets/lbw_63/val_target_scale.npy",
                            "train_sequence_index_path": "artifacts/interim/datasets/lbw_63/train_sequence_index.csv",
                            "val_sequence_index_path": "artifacts/interim/datasets/lbw_63/val_sequence_index.csv",
                        },
                        "source_artifacts": {
                            "split_manifest_path": "artifacts/interim/datasets/lbw_63/split_manifest.csv",
                            "sequence_manifest_path": "artifacts/interim/datasets/lbw_63/sequence_manifest.csv",
                            "target_alignment_registry_path": "artifacts/interim/datasets/lbw_63/target_alignment_registry.csv",
                        },
                    }
                ],
            )
            candidate_config = CandidateConfig(
                candidate_id="TEST",
                candidate_index=1,
                dropout=0.1,
                hidden_size=5,
                minibatch_size=2,
                learning_rate=1e-2,
                max_grad_norm=1.0,
                lbw=63,
            )

            first_result = run_candidate_training(
                dataset_registry_path=registry_path,
                candidate_config=candidate_config,
                output_dir=tmp_path / "artifacts/interim/training/run_a",
                project_root=tmp_path,
                artifact_stem="candidate",
                max_epochs=4,
                patience=2,
            )
            second_result = run_candidate_training(
                dataset_registry_path=registry_path,
                candidate_config=candidate_config,
                output_dir=tmp_path / "artifacts/interim/training/run_b",
                project_root=tmp_path,
                artifact_stem="candidate",
                max_epochs=4,
                patience=2,
            )

            self.assertEqual(first_result.validation_losses, second_result.validation_losses)
            self.assertAlmostEqual(
                first_result.best_validation_loss,
                second_result.best_validation_loss,
                places=12,
            )

            with first_result.epoch_log_path.open("r", encoding="utf-8") as handle:
                first_epoch_log = handle.read()
            with second_result.epoch_log_path.open("r", encoding="utf-8") as handle:
                second_epoch_log = handle.read()
            self.assertEqual(first_epoch_log, second_epoch_log)


if __name__ == "__main__":
    unittest.main()
