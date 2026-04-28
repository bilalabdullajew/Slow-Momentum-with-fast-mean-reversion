from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.datasets.join_and_split import MODEL_INPUT_COLUMNS  # noqa: E402
from lstm_cpd.datasets.registry import (  # noqa: E402
    T18_SEQUENCE_INDEX_HEADER,
    build_t18_outputs,
)
from lstm_cpd.datasets.sequences import (  # noqa: E402
    T17_SEQUENCE_MANIFEST_HEADER,
    T17_TARGET_ALIGNMENT_HEADER,
)


def write_csv(path: Path, header: Sequence[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def make_timestamp(day_offset: int) -> str:
    return (datetime(2024, 1, 1) + timedelta(days=day_offset)).isoformat()


def make_sequence_manifest_row(
    *,
    sequence_id: str,
    split: str,
    start_day: int,
    start_timeline_index: int,
    lbw: int = 10,
    asset_id: str = "TEST",
) -> list[str]:
    return [
        sequence_id,
        asset_id,
        str(lbw),
        split,
        "0",
        make_timestamp(start_day),
        make_timestamp(start_day + 62),
        str(start_timeline_index),
        str(start_timeline_index + 62),
        "63",
    ]


def make_target_alignment_row(
    *,
    sequence_id: str,
    split: str,
    step_index: int,
    day_offset: int,
    timeline_index: int,
    lbw: int = 10,
    asset_id: str = "TEST",
) -> list[str]:
    feature_values = [str(step_index + feature_index + 1) for feature_index in range(10)]
    return [
        sequence_id,
        asset_id,
        str(lbw),
        split,
        str(step_index),
        make_timestamp(day_offset),
        str(timeline_index),
        "0.5",
        "0.01",
        str(0.15 / 0.5 * 0.01),
        *feature_values,
    ]


class T18DatasetRegistryTests(unittest.TestCase):
    def test_build_outputs_materializes_arrays_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lbw_dir = tmp_path / "artifacts/datasets/lbw_10"
            sequence_id_train = "TEST__lbw_10__train__00000000"
            sequence_id_val = "TEST__lbw_10__validation__00000100"

            write_csv(
                lbw_dir / "sequence_manifest.csv",
                T17_SEQUENCE_MANIFEST_HEADER,
                [
                    make_sequence_manifest_row(
                        sequence_id=sequence_id_train,
                        split="train",
                        start_day=0,
                        start_timeline_index=0,
                    ),
                    make_sequence_manifest_row(
                        sequence_id=sequence_id_val,
                        split="validation",
                        start_day=100,
                        start_timeline_index=100,
                    ),
                ],
            )
            target_rows: list[list[str]] = []
            for step_index in range(63):
                target_rows.append(
                    make_target_alignment_row(
                        sequence_id=sequence_id_train,
                        split="train",
                        step_index=step_index,
                        day_offset=step_index,
                        timeline_index=step_index,
                    )
                )
                target_rows.append(
                    make_target_alignment_row(
                        sequence_id=sequence_id_val,
                        split="validation",
                        step_index=step_index,
                        day_offset=100 + step_index,
                        timeline_index=100 + step_index,
                    )
                )
            write_csv(
                lbw_dir / "target_alignment_registry.csv",
                T17_TARGET_ALIGNMENT_HEADER,
                target_rows,
            )
            write_csv(lbw_dir / "split_manifest.csv", ["placeholder"], [])

            artifacts = build_t18_outputs(
                input_dir=tmp_path / "artifacts/datasets",
                output_dir=tmp_path / "artifacts/datasets",
                dataset_registry_output=tmp_path / "artifacts/manifests/dataset_registry.json",
                project_root=tmp_path,
                lbws=(10,),
            )

            train_inputs = np.load(tmp_path / "artifacts/datasets/lbw_10/train_inputs.npy")
            train_targets = np.load(
                tmp_path / "artifacts/datasets/lbw_10/train_target_scale.npy"
            )
            val_inputs = np.load(tmp_path / "artifacts/datasets/lbw_10/val_inputs.npy")
            val_targets = np.load(
                tmp_path / "artifacts/datasets/lbw_10/val_target_scale.npy"
            )

            self.assertEqual(train_inputs.shape, (1, 63, len(MODEL_INPUT_COLUMNS)))
            self.assertEqual(train_targets.shape, (1, 63))
            self.assertEqual(val_inputs.shape, (1, 63, len(MODEL_INPUT_COLUMNS)))
            self.assertEqual(val_targets.shape, (1, 63))
            self.assertAlmostEqual(float(train_inputs[0, 0, 0]), 1.0)
            self.assertAlmostEqual(float(val_targets[0, -1]), 0.003)

            with (tmp_path / "artifacts/datasets/lbw_10/train_sequence_index.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                train_index_rows = list(csv.DictReader(handle))
            with (tmp_path / "artifacts/datasets/lbw_10/val_sequence_index.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                val_index_rows = list(csv.DictReader(handle))
            self.assertEqual(len(train_index_rows), 1)
            self.assertEqual(len(val_index_rows), 1)
            self.assertEqual(list(T18_SEQUENCE_INDEX_HEADER), list(train_index_rows[0].keys()))

            registry_entries = json.loads(artifacts.dataset_registry_path.read_text(encoding="utf-8"))
            self.assertEqual(len(registry_entries), 1)
            self.assertEqual(registry_entries[0]["lbw"], 10)
            self.assertEqual(registry_entries[0]["train_input_shape"], [1, 63, 10])
            self.assertEqual(registry_entries[0]["val_input_shape"], [1, 63, 10])

    def test_build_outputs_rejects_incomplete_target_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lbw_dir = tmp_path / "artifacts/datasets/lbw_10"
            sequence_id_train = "TEST__lbw_10__train__00000000"

            write_csv(
                lbw_dir / "sequence_manifest.csv",
                T17_SEQUENCE_MANIFEST_HEADER,
                [
                    make_sequence_manifest_row(
                        sequence_id=sequence_id_train,
                        split="train",
                        start_day=0,
                        start_timeline_index=0,
                    )
                ],
            )
            target_rows = [
                make_target_alignment_row(
                    sequence_id=sequence_id_train,
                    split="train",
                    step_index=step_index,
                    day_offset=step_index,
                    timeline_index=step_index,
                )
                for step_index in range(62)
            ]
            write_csv(
                lbw_dir / "target_alignment_registry.csv",
                T17_TARGET_ALIGNMENT_HEADER,
                target_rows,
            )
            write_csv(lbw_dir / "split_manifest.csv", ["placeholder"], [])

            with self.assertRaisesRegex(ValueError, "length mismatch"):
                build_t18_outputs(
                    input_dir=tmp_path / "artifacts/datasets",
                    output_dir=tmp_path / "artifacts/datasets",
                    dataset_registry_output=tmp_path / "artifacts/manifests/dataset_registry.json",
                    project_root=tmp_path,
                    lbws=(10,),
                )


if __name__ == "__main__":
    unittest.main()
