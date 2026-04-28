from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.datasets.join_and_split import (  # noqa: E402
    CPD_OUTPUT_HEADER,
    T16_OUTPUT_HEADER,
    T16_SPLIT_MANIFEST_HEADER,
    build_t16_outputs,
)
from lstm_cpd.datasets.sequences import (  # noqa: E402
    REASON_GAP_IN_TIMELINE,
    REASON_MISSING_NEXT_RETURN,
    REASON_NO_FULL_VALIDATION_SEQUENCE,
    REASON_TERMINAL_FRAGMENT_LT_63,
    T17_DISCARDED_FRAGMENTS_HEADER,
    T17_GAP_EXCLUSION_HEADER,
    build_t17_outputs,
    load_sequence_manifest_csv,
    load_target_alignment_registry_csv,
)
from lstm_cpd.features.returns import RETURNS_VOLATILITY_HEADER  # noqa: E402
from lstm_cpd.features.winsorize import T12_OUTPUT_HEADER  # noqa: E402


def write_csv(path: Path, header: Sequence[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_timestamp(day_offset: int) -> str:
    return (datetime(2024, 1, 1) + timedelta(days=day_offset)).isoformat()


def make_joined_row(
    *,
    timestamp: str,
    asset_id: str = "TEST",
    lbw: int = 10,
    timeline_index: int,
    sigma_t: str = "0.5",
    next_arithmetic_return: str = "0.01",
) -> list[str]:
    feature_values = [str(index) for index in range(1, 11)]
    return [
        timestamp,
        asset_id,
        str(lbw),
        str(timeline_index),
        sigma_t,
        next_arithmetic_return,
        *feature_values,
    ]


def make_split_manifest_row(
    *,
    asset_id: str = "TEST",
    lbw: int = 10,
    usable_row_count: int,
    train_row_count: int,
    val_row_count: int,
    train_start_timestamp: str,
    train_end_timestamp: str,
    val_start_timestamp: str,
    val_end_timestamp: str,
    train_start_timeline_index: int,
    train_end_timeline_index: int,
    val_start_timeline_index: int,
    val_end_timeline_index: int,
) -> list[str]:
    return [
        asset_id,
        str(lbw),
        str(usable_row_count),
        str(train_row_count),
        str(val_row_count),
        train_start_timestamp,
        train_end_timestamp,
        val_start_timestamp,
        val_end_timestamp,
        str(train_start_timeline_index),
        str(train_end_timeline_index),
        str(val_start_timeline_index),
        str(val_end_timeline_index),
        "floor_90_rest_10",
    ]


def make_cpd_manifest_entry(
    *,
    asset_id: str = "AAPL",
    cpd_csv_path: str = "artifacts/features/cpd/lbw_10/AAPL_cpd.csv",
    row_count: int = 5330,
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "lbw": 10,
        "state": "present",
        "missing_reason": None,
        "cpd_csv_path": cpd_csv_path,
        "row_count": row_count,
        "canonical_row_count": row_count,
        "first_timestamp": "2007-04-27T00:00:00",
        "last_timestamp": "2026-02-20T00:00:00",
        "output_row_count": row_count,
        "retry_used_count": 0,
        "fallback_used_count": 0,
        "status_counts": {
            "success": row_count,
            "retry_success": 0,
            "fallback_previous": 0,
            "baseline_failure": 0,
            "changepoint_failure": 0,
            "invalid_window": 0,
        },
        "matches_canonical_timeline": True,
        "file_hash": "sha256:test",
    }


class T17SequenceTests(unittest.TestCase):
    def test_build_outputs_does_not_use_cross_split_next_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lbw_dir = tmp_path / "artifacts/datasets/lbw_10"
            joined_dir = lbw_dir / "joined_features"
            rows = [
                make_joined_row(
                    timestamp=make_timestamp(index),
                    timeline_index=index,
                    next_arithmetic_return="0.01",
                )
                for index in range(127)
            ]
            write_csv(joined_dir / "TEST.csv", T16_OUTPUT_HEADER, rows)
            write_csv(
                lbw_dir / "split_manifest.csv",
                T16_SPLIT_MANIFEST_HEADER,
                [
                    make_split_manifest_row(
                        usable_row_count=127,
                        train_row_count=63,
                        val_row_count=64,
                        train_start_timestamp=make_timestamp(0),
                        train_end_timestamp=make_timestamp(62),
                        val_start_timestamp=make_timestamp(63),
                        val_end_timestamp=make_timestamp(126),
                        train_start_timeline_index=0,
                        train_end_timeline_index=62,
                        val_start_timeline_index=63,
                        val_end_timeline_index=126,
                    )
                ],
            )

            artifacts = build_t17_outputs(
                input_dir=tmp_path / "artifacts/datasets",
                output_dir=tmp_path / "artifacts/datasets",
                project_root=tmp_path,
                lbws=(10,),
            )

            sequence_rows = load_sequence_manifest_csv(artifacts.sequence_manifest_paths[0])
            target_rows = load_target_alignment_registry_csv(
                artifacts.target_alignment_registry_paths[0]
            )

            self.assertEqual({row.split for row in sequence_rows}, {"validation"})
            self.assertEqual(len(sequence_rows), 1)
            self.assertEqual(len(target_rows), 63)
            self.assertEqual(target_rows[-1].timeline_index, 125)

    def test_build_outputs_reports_gap_missing_target_and_terminal_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lbw_dir = tmp_path / "artifacts/datasets/lbw_10"
            joined_dir = lbw_dir / "joined_features"

            train_indices = list(range(0, 10)) + list(range(11, 40)) + [40] + list(range(41, 71))
            val_indices = list(range(200, 264))
            all_indices = train_indices + val_indices
            rows = []
            for index, timeline_index in enumerate(all_indices):
                next_return = ""
                if timeline_index != 40:
                    next_return = "0.01"
                rows.append(
                    make_joined_row(
                        timestamp=make_timestamp(index),
                        timeline_index=timeline_index,
                        next_arithmetic_return=next_return,
                    )
                )

            write_csv(joined_dir / "TEST.csv", T16_OUTPUT_HEADER, rows)
            write_csv(
                lbw_dir / "split_manifest.csv",
                T16_SPLIT_MANIFEST_HEADER,
                [
                    make_split_manifest_row(
                        usable_row_count=len(rows),
                        train_row_count=70,
                        val_row_count=64,
                        train_start_timestamp=make_timestamp(0),
                        train_end_timestamp=make_timestamp(69),
                        val_start_timestamp=make_timestamp(70),
                        val_end_timestamp=make_timestamp(133),
                        train_start_timeline_index=train_indices[0],
                        train_end_timeline_index=train_indices[-1],
                        val_start_timeline_index=val_indices[0],
                        val_end_timeline_index=val_indices[-1],
                    )
                ],
            )

            artifacts = build_t17_outputs(
                input_dir=tmp_path / "artifacts/datasets",
                output_dir=tmp_path / "artifacts/datasets",
                project_root=tmp_path,
                lbws=(10,),
            )

            sequence_rows = load_sequence_manifest_csv(artifacts.sequence_manifest_paths[0])
            target_rows = load_target_alignment_registry_csv(
                artifacts.target_alignment_registry_paths[0]
            )
            with artifacts.discarded_fragment_report_paths[0].open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                discarded_rows = list(csv.DictReader(handle))
            with artifacts.gap_exclusion_report_paths[0].open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                gap_rows = list(csv.DictReader(handle))

            self.assertEqual(len(sequence_rows), 1)
            self.assertEqual(sequence_rows[0].split, "validation")
            self.assertEqual(len(target_rows), 63)
            self.assertEqual(len(gap_rows), 1)
            self.assertEqual(gap_rows[0]["reason_code"], REASON_GAP_IN_TIMELINE)
            self.assertEqual(gap_rows[0]["missing_steps"], "1")
            discarded_reason_codes = {row["reason_code"] for row in discarded_rows}
            self.assertIn(REASON_MISSING_NEXT_RETURN, discarded_reason_codes)
            self.assertIn(REASON_TERMINAL_FRAGMENT_LT_63, discarded_reason_codes)
            self.assertNotIn(REASON_NO_FULL_VALIDATION_SEQUENCE, discarded_reason_codes)
            self.assertEqual(
                list(T17_GAP_EXCLUSION_HEADER),
                list(gap_rows[0].keys()),
            )
            self.assertEqual(
                list(T17_DISCARDED_FRAGMENTS_HEADER),
                list(discarded_rows[0].keys()),
            )

    def test_build_outputs_excludes_asset_without_full_validation_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lbw_dir = tmp_path / "artifacts/datasets/lbw_10"
            joined_dir = lbw_dir / "joined_features"
            rows = [
                make_joined_row(
                    timestamp=make_timestamp(index),
                    timeline_index=index,
                )
                for index in range(76)
            ]
            write_csv(joined_dir / "TEST.csv", T16_OUTPUT_HEADER, rows)
            write_csv(
                lbw_dir / "split_manifest.csv",
                T16_SPLIT_MANIFEST_HEADER,
                [
                    make_split_manifest_row(
                        usable_row_count=76,
                        train_row_count=63,
                        val_row_count=13,
                        train_start_timestamp=make_timestamp(0),
                        train_end_timestamp=make_timestamp(62),
                        val_start_timestamp=make_timestamp(63),
                        val_end_timestamp=make_timestamp(75),
                        train_start_timeline_index=0,
                        train_end_timeline_index=62,
                        val_start_timeline_index=63,
                        val_end_timeline_index=75,
                    )
                ],
            )

            artifacts = build_t17_outputs(
                input_dir=tmp_path / "artifacts/datasets",
                output_dir=tmp_path / "artifacts/datasets",
                project_root=tmp_path,
                lbws=(10,),
            )

            sequence_rows = load_sequence_manifest_csv(artifacts.sequence_manifest_paths[0])
            target_rows = load_target_alignment_registry_csv(
                artifacts.target_alignment_registry_paths[0]
            )
            with artifacts.discarded_fragment_report_paths[0].open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                discarded_rows = list(csv.DictReader(handle))

            self.assertEqual(sequence_rows, [])
            self.assertEqual(target_rows, [])
            exclusion_row = next(
                row for row in discarded_rows if row["reason_code"] == REASON_NO_FULL_VALIDATION_SEQUENCE
            )
            self.assertEqual(exclusion_row["dropped_sequence_count"], "0")

    def test_build_outputs_against_real_aapl_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_dir = PROJECT_ROOT / "artifacts/features/base"
            manifest_path = tmp_path / "cpd_feature_store_manifest.json"
            write_json(manifest_path, [make_cpd_manifest_entry()])

            build_t16_outputs(
                base_input_dir=base_dir,
                returns_input_dir=base_dir,
                cpd_manifest_input=manifest_path,
                output_dir=tmp_path / "artifacts/datasets",
                project_root=PROJECT_ROOT,
                lbws=(10,),
            )
            artifacts = build_t17_outputs(
                input_dir=tmp_path / "artifacts/datasets",
                output_dir=tmp_path / "artifacts/datasets",
                project_root=tmp_path,
                lbws=(10,),
            )

            sequence_rows = load_sequence_manifest_csv(artifacts.sequence_manifest_paths[0])
            target_rows = load_target_alignment_registry_csv(
                artifacts.target_alignment_registry_paths[0]
            )

            self.assertGreater(len(sequence_rows), 0)
            self.assertEqual(len(target_rows), len(sequence_rows) * 63)
            self.assertEqual({row.split for row in sequence_rows}, {"train", "validation"})


if __name__ == "__main__":
    unittest.main()
