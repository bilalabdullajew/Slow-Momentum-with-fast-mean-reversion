from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.datasets.registry import T18_SEQUENCE_INDEX_HEADER  # noqa: E402
from lstm_cpd.datasets.sequences import (  # noqa: E402
    T17_SEQUENCE_MANIFEST_HEADER,
    T17_TARGET_ALIGNMENT_HEADER,
)
from lstm_cpd.evaluation.validation_evaluation import (  # noqa: E402
    run_validation_evaluation,
)
from lstm_cpd.features.returns import CANONICAL_DAILY_CLOSE_HEADER  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def make_timestamp(day_offset: int) -> str:
    return (datetime(2024, 1, 1) + timedelta(days=day_offset)).isoformat()


def write_model_source_artifacts(project_root: Path, *, lbw: int) -> tuple[Path, Path]:
    config_path = project_root / "artifacts/interim/training/smoke_run/smoke_config.json"
    model_path = project_root / "artifacts/interim/training/smoke_run/smoke_best_model.keras"
    write_json(
        config_path,
        {
            "candidate_id": "SMOKE",
            "candidate_index": 0,
            "dropout": 0.1,
            "hidden_size": 20,
            "minibatch_size": 64,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "lbw": lbw,
            "dataset_registry_path": str(
                project_root / "artifacts/interim/manifests/dataset_registry.json"
            ),
        },
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("stub checkpoint\n", encoding="utf-8")
    return config_path, model_path


def write_validation_dataset_fixture(
    project_root: Path,
    *,
    target_scale: np.ndarray,
) -> Path:
    lbw = 63
    dataset_root = project_root / f"artifacts/interim/datasets/lbw_{lbw}"
    dataset_root.mkdir(parents=True, exist_ok=True)
    train_inputs = np.zeros((1, 63, 10), dtype=np.float32)
    train_targets = np.zeros((1, 63), dtype=np.float32)
    val_inputs = np.zeros((1, 63, 10), dtype=np.float32)
    np.save(dataset_root / "train_inputs.npy", train_inputs)
    np.save(dataset_root / "train_target_scale.npy", train_targets)
    np.save(dataset_root / "val_inputs.npy", val_inputs)
    np.save(dataset_root / "val_target_scale.npy", target_scale.astype(np.float32)[None, :])

    write_csv(
        dataset_root / "val_sequence_index.csv",
        T18_SEQUENCE_INDEX_HEADER,
        [
            [
                "0",
                "TEST__lbw_63__validation__00000000",
                "TEST",
                "63",
                "validation",
                make_timestamp(0),
                make_timestamp(62),
                "0",
                "62",
            ]
        ],
    )
    write_csv(
        dataset_root / "train_sequence_index.csv",
        T18_SEQUENCE_INDEX_HEADER,
        [
            [
                "0",
                "TEST__lbw_63__train__00000000",
                "TEST",
                "63",
                "train",
                make_timestamp(0),
                make_timestamp(62),
                "0",
                "62",
            ]
        ],
    )
    write_csv(
        dataset_root / "sequence_manifest.csv",
        T17_SEQUENCE_MANIFEST_HEADER,
        [
            [
                "TEST__lbw_63__validation__00000000",
                "TEST",
                "63",
                "validation",
                "0",
                make_timestamp(0),
                make_timestamp(62),
                "0",
                "62",
                "63",
            ]
        ],
    )
    target_rows = []
    for day_offset in range(63):
        model_inputs = [f"{0.1 * (index + 1):.6f}" for index in range(10)]
        target_rows.append(
            [
                "TEST__lbw_63__validation__00000000",
                "TEST",
                "63",
                "validation",
                str(day_offset),
                make_timestamp(day_offset),
                str(day_offset),
                "1.0",
                f"{target_scale[day_offset] / 0.15:.6f}",
                f"{target_scale[day_offset]:.8f}",
                *model_inputs,
            ]
        )
    write_csv(
        dataset_root / "target_alignment_registry.csv",
        T17_TARGET_ALIGNMENT_HEADER,
        target_rows,
    )
    write_csv(
        project_root / "artifacts/canonical_daily_close/TEST.csv",
        CANONICAL_DAILY_CLOSE_HEADER,
        [
            [make_timestamp(day_offset), "TEST", f"{100.0 + day_offset:.6f}"]
            for day_offset in range(64)
        ],
    )
    write_json(
        project_root / "artifacts/manifests/canonical_daily_close_manifest.json",
        [
            {
                "asset_id": "TEST",
                "symbol": "TEST",
                "category": "equity",
                "path_pattern": "unused",
                "source_d_file_path": "unused",
                "canonical_csv_path": "artifacts/canonical_daily_close/TEST.csv",
                "row_count": 64,
                "first_timestamp": make_timestamp(0),
                "last_timestamp": make_timestamp(63),
                "file_hash": "sha256:test-canonical",
            }
        ],
    )
    registry_path = project_root / "artifacts/interim/manifests/dataset_registry.json"
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
                "train_sequence_count": 1,
                "val_sequence_count": 1,
                "train_input_shape": [1, 63, 10],
                "train_target_shape": [1, 63],
                "val_input_shape": [1, 63, 10],
                "val_target_shape": [1, 63],
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
                    "target_alignment_registry_path": (
                        "artifacts/interim/datasets/lbw_63/target_alignment_registry.csv"
                    ),
                },
            }
        ],
    )
    return registry_path


class FakeValidationModel:
    def __init__(self, positions: np.ndarray) -> None:
        self.positions = positions.astype(np.float32)

    def __call__(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return tf.convert_to_tensor(self.positions, dtype=tf.float32)


class T26ValidationEvaluationTests(unittest.TestCase):
    def test_validation_evaluation_reconstructs_daily_returns_and_rescaling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_scale = np.linspace(-0.03, 0.05, 63, dtype=np.float32)
            registry_path = write_validation_dataset_fixture(tmp_path, target_scale=target_scale)
            config_path, model_path = write_model_source_artifacts(tmp_path, lbw=63)
            positions = np.linspace(-0.5, 0.5, 63, dtype=np.float32)[None, :, None]

            with patch(
                "lstm_cpd.evaluation.validation_evaluation.tf.keras.models.load_model",
                return_value=FakeValidationModel(positions),
            ):
                artifacts = run_validation_evaluation(
                    model_path=model_path,
                    candidate_config_path=config_path,
                    dataset_registry_path=registry_path,
                    canonical_manifest_input=(
                        tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"
                    ),
                    raw_validation_returns_output=(
                        tmp_path / "artifacts/evaluation/raw_validation_returns.csv"
                    ),
                    raw_validation_metrics_output=(
                        tmp_path / "artifacts/evaluation/raw_validation_metrics.json"
                    ),
                    rescaled_validation_returns_output=(
                        tmp_path / "artifacts/evaluation/rescaled_validation_returns.csv"
                    ),
                    rescaled_validation_metrics_output=(
                        tmp_path / "artifacts/evaluation/rescaled_validation_metrics.json"
                    ),
                    evaluation_report_output=(
                        tmp_path / "artifacts/evaluation/evaluation_report.md"
                    ),
                    project_root=tmp_path,
                )

            self.assertEqual(artifacts.daily_observation_count, 63)
            with artifacts.raw_validation_returns_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                raw_rows = list(csv.DictReader(handle))
            self.assertEqual(len(raw_rows), 63)
            self.assertEqual(raw_rows[0]["return_timestamp"], make_timestamp(1))
            expected_raw_returns = positions[0, :, 0] * target_scale
            self.assertAlmostEqual(
                float(raw_rows[0]["portfolio_return"]),
                float(expected_raw_returns[0]),
                places=7,
            )

            raw_metrics = json.loads(artifacts.raw_validation_metrics_path.read_text(encoding="utf-8"))
            rescaled_metrics = json.loads(
                artifacts.rescaled_validation_metrics_path.read_text(encoding="utf-8")
            )
            expected_annualized_volatility = float(
                np.std(expected_raw_returns, ddof=0) * math.sqrt(252)
            )
            self.assertAlmostEqual(
                raw_metrics["annualized_volatility"],
                expected_annualized_volatility,
                places=7,
            )
            self.assertAlmostEqual(
                rescaled_metrics["rescaling_factor"],
                0.15 / expected_annualized_volatility,
                places=7,
            )
            report_text = artifacts.evaluation_report_path.read_text(encoding="utf-8")
            self.assertIn("FTMO evaluation is model-faithful", report_text)

    def test_validation_evaluation_hard_fails_when_raw_volatility_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_scale = np.zeros(63, dtype=np.float32)
            registry_path = write_validation_dataset_fixture(tmp_path, target_scale=target_scale)
            config_path, model_path = write_model_source_artifacts(tmp_path, lbw=63)
            positions = np.zeros((1, 63, 1), dtype=np.float32)

            with patch(
                "lstm_cpd.evaluation.validation_evaluation.tf.keras.models.load_model",
                return_value=FakeValidationModel(positions),
            ):
                with self.assertRaisesRegex(ValueError, "annualized volatility is zero"):
                    run_validation_evaluation(
                        model_path=model_path,
                        candidate_config_path=config_path,
                        dataset_registry_path=registry_path,
                        canonical_manifest_input=(
                            tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"
                        ),
                        raw_validation_returns_output=(
                            tmp_path / "artifacts/evaluation/raw_validation_returns.csv"
                        ),
                        raw_validation_metrics_output=(
                            tmp_path / "artifacts/evaluation/raw_validation_metrics.json"
                        ),
                        rescaled_validation_returns_output=(
                            tmp_path / "artifacts/evaluation/rescaled_validation_returns.csv"
                        ),
                        rescaled_validation_metrics_output=(
                            tmp_path / "artifacts/evaluation/rescaled_validation_metrics.json"
                        ),
                        evaluation_report_output=(
                            tmp_path / "artifacts/evaluation/evaluation_report.md"
                        ),
                        project_root=tmp_path,
                    )


if __name__ == "__main__":
    unittest.main()
