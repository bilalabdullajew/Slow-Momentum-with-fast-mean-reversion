from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.datasets.interim_materialization import (  # noqa: E402
    materialize_interim_datasets,
)
from lstm_cpd.datasets.join_and_split import CPD_OUTPUT_HEADER  # noqa: E402
from lstm_cpd.features.returns import (  # noqa: E402
    CANONICAL_DAILY_CLOSE_HEADER,
    RETURNS_VOLATILITY_HEADER,
)
from lstm_cpd.features.winsorize import T12_OUTPUT_HEADER  # noqa: E402


def write_csv(path: Path, header: Sequence[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_timestamp(day_offset: int) -> str:
    return (datetime(2024, 1, 1) + timedelta(days=day_offset)).isoformat()


def make_base_row(day_offset: int, *, asset_id: str = "TEST") -> list[str]:
    feature_values = [str(feature_index + 1) for feature_index in range(8)]
    return [make_timestamp(day_offset), asset_id, *feature_values]


def make_returns_row(day_offset: int, *, asset_id: str = "TEST") -> list[str]:
    arithmetic_return = "" if day_offset == 0 else "0.01"
    return [
        make_timestamp(day_offset),
        asset_id,
        str(100 + day_offset),
        arithmetic_return,
        "1.0",
    ]


def make_cpd_row(day_offset: int, *, asset_id: str = "TEST", lbw: int = 10) -> list[str]:
    return [
        make_timestamp(day_offset),
        asset_id,
        str(lbw),
        "0.25",
        "0.75",
        "success",
        str(lbw + 1),
        "0.1",
        "0.2",
        "false",
        "false",
        "",
        "",
        "",
        "",
        "",
    ]


class InterimMaterializationTests(unittest.TestCase):
    def test_materialize_interim_datasets_uses_completed_pairs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"

            canonical_manifest = [
                {
                    "asset_id": "TEST",
                    "symbol": "TEST",
                    "category": "equity",
                    "path_pattern": "unused",
                    "source_d_file_path": "unused",
                    "canonical_csv_path": "artifacts/canonical_daily_close/TEST.csv",
                    "row_count": 640,
                    "first_timestamp": make_timestamp(0),
                    "last_timestamp": make_timestamp(639),
                    "file_hash": "sha256:canonical-test",
                }
            ]
            write_json(
                project_root / "artifacts/manifests/canonical_daily_close_manifest.json",
                canonical_manifest,
            )
            write_csv(
                project_root / "artifacts/canonical_daily_close/TEST.csv",
                CANONICAL_DAILY_CLOSE_HEADER,
                [
                    [make_timestamp(day_offset), "TEST", str(100 + day_offset)]
                    for day_offset in range(640)
                ],
            )
            write_csv(
                project_root / "artifacts/features/base/TEST_base_features.csv",
                T12_OUTPUT_HEADER,
                [make_base_row(day_offset) for day_offset in range(640)],
            )
            write_csv(
                project_root / "artifacts/features/base/TEST_returns_volatility.csv",
                RETURNS_VOLATILITY_HEADER,
                [make_returns_row(day_offset) for day_offset in range(640)],
            )
            write_csv(
                project_root / "artifacts/features/cpd/lbw_10/TEST_cpd.csv",
                CPD_OUTPUT_HEADER,
                [make_cpd_row(day_offset) for day_offset in range(640)],
            )
            write_csv(
                project_root / "artifacts/reports/cpd_precompute_progress.csv",
                (
                    "lbw",
                    "asset_id",
                    "state",
                    "rows_written",
                    "last_timestamp",
                    "retry_count",
                    "fallback_count",
                    "started_at",
                    "finished_at",
                    "output_path",
                    "error_message",
                ),
                [
                    [
                        "10",
                        "TEST",
                        "completed",
                        "640",
                        make_timestamp(639),
                        "0",
                        "0",
                        "2026-04-23T12:00:00Z",
                        "2026-04-23T12:30:00Z",
                        "artifacts/features/cpd/lbw_10/TEST_cpd.csv",
                        "",
                    ],
                    [
                        "10",
                        "LATER",
                        "running",
                        "117",
                        make_timestamp(116),
                        "0",
                        "0",
                        "2026-04-23T12:35:00Z",
                        "",
                        "artifacts/features/cpd/lbw_10/LATER_cpd.csv",
                        "",
                    ],
                ],
            )

            artifacts = materialize_interim_datasets(
                progress_input=project_root / "artifacts/reports/cpd_precompute_progress.csv",
                canonical_manifest_input=(
                    project_root / "artifacts/manifests/canonical_daily_close_manifest.json"
                ),
                base_input_dir=project_root / "artifacts/features/base",
                returns_input_dir=project_root / "artifacts/features/base",
                cpd_input_dir=project_root / "artifacts/features/cpd",
                output_root=project_root / "artifacts/interim",
                project_root=project_root,
                coverage_summary_output=(
                    project_root / "artifacts/interim/manifests/interim_materialization_summary.json"
                ),
                dataset_registry_output=(
                    project_root / "artifacts/interim/manifests/dataset_registry.json"
                ),
                lbws=(10,),
            )

            registry_entries = json.loads(
                artifacts.dataset_registry_path.read_text(encoding="utf-8")
            )
            self.assertEqual(artifacts.materialized_lbws, (10,))
            self.assertEqual(len(registry_entries), 1)
            self.assertEqual(registry_entries[0]["lbw"], 10)
            self.assertEqual(registry_entries[0]["train_input_shape"], [9, 63, 10])
            self.assertEqual(registry_entries[0]["val_input_shape"], [1, 63, 10])

            summary = json.loads(artifacts.coverage_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "interim_partial_cpd_coverage")
            self.assertEqual(summary["materialized_lbws"], [10])
            self.assertEqual(summary["coverage_by_lbw"][0]["completed_asset_count"], 1)
            self.assertEqual(summary["coverage_by_lbw"][0]["selected_asset_ids"], ["TEST"])
            self.assertEqual(summary["coverage_by_lbw"][0]["progress_state_counts"]["running"], 1)

            self.assertTrue(
                (
                    project_root / "artifacts/interim/datasets/lbw_10/train_inputs.npy"
                ).exists()
            )
            self.assertTrue(
                (
                    project_root / "artifacts/interim/datasets/lbw_10/val_sequence_index.csv"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
