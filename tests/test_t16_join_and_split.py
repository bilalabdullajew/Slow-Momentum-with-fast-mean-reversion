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
    load_joined_feature_csv,
    load_split_manifest_csv,
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


def make_cpd_manifest_entry(
    *,
    asset_id: str = "TEST",
    lbw: int = 10,
    state: str = "present",
    cpd_csv_path: str = "artifacts/features/cpd/lbw_10/TEST_cpd.csv",
    row_count: int = 4,
    canonical_row_count: int = 4,
    matches_canonical_timeline: bool = True,
    file_hash: str | None = "sha256:test",
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "lbw": lbw,
        "state": state,
        "missing_reason": None,
        "cpd_csv_path": cpd_csv_path,
        "row_count": row_count,
        "canonical_row_count": canonical_row_count,
        "first_timestamp": "2024-01-01T00:00:00",
        "last_timestamp": "2024-01-04T00:00:00",
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
        "matches_canonical_timeline": matches_canonical_timeline,
        "file_hash": file_hash,
    }


def build_base_row(
    timestamp: str,
    *,
    asset_id: str = "TEST",
    values: tuple[str, ...] | None = None,
) -> list[str]:
    return [timestamp, asset_id, *(values or ("",) * 8)]


def build_returns_row(
    timestamp: str,
    *,
    asset_id: str = "TEST",
    close: str = "100",
    arithmetic_return: str = "",
    sigma_t: str = "",
) -> list[str]:
    return [timestamp, asset_id, close, arithmetic_return, sigma_t]


def build_cpd_row(
    timestamp: str,
    *,
    asset_id: str = "TEST",
    lbw: int = 10,
    nu: str = "",
    gamma: str = "",
    status: str = "success",
) -> list[str]:
    return [
        timestamp,
        asset_id,
        str(lbw),
        nu,
        gamma,
        status,
        "10",
        "",
        "",
        "false" if status == "success" else "true" if status == "retry_success" else "false",
        "true" if status == "fallback_previous" else "false",
        "",
        "",
        "2024-01-02T00:00:00" if status == "fallback_previous" else "",
        "" if status in {"success", "retry_success", "fallback_previous"} else "window_length",
        "" if status in {"success", "retry_success", "fallback_previous"} else "no outputs",
    ]


class T16JoinAndSplitTests(unittest.TestCase):
    def test_build_outputs_rejects_incomplete_cpd_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "artifacts/manifests/cpd_feature_store_manifest.json"
            write_json(
                manifest_path,
                [
                    make_cpd_manifest_entry(
                        state="missing",
                        matches_canonical_timeline=False,
                        file_hash=None,
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "incomplete"):
                build_t16_outputs(
                    cpd_manifest_input=manifest_path,
                    output_dir=tmp_path / "artifacts/datasets",
                    project_root=tmp_path,
                    lbws=(10,),
                )

    def test_build_outputs_filters_to_complete_timestep_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_dir = tmp_path / "artifacts/features/base"
            cpd_path = tmp_path / "artifacts/features/cpd/lbw_10/TEST_cpd.csv"
            manifest_path = tmp_path / "artifacts/manifests/cpd_feature_store_manifest.json"

            write_csv(
                base_dir / "TEST_base_features.csv",
                T12_OUTPUT_HEADER,
                [
                    build_base_row("2024-01-01T00:00:00"),
                    build_base_row(
                        "2024-01-02T00:00:00",
                        values=("1", "2", "3", "4", "5", "6", "7", "8"),
                    ),
                    build_base_row(
                        "2024-01-03T00:00:00",
                        values=("11", "12", "13", "14", "15", "16", "17", "18"),
                    ),
                    build_base_row(
                        "2024-01-04T00:00:00",
                        values=("21", "22", "23", "24", "25", "26", "27", "28"),
                    ),
                ],
            )
            write_csv(
                base_dir / "TEST_returns_volatility.csv",
                RETURNS_VOLATILITY_HEADER,
                [
                    build_returns_row("2024-01-01T00:00:00", sigma_t="0.10"),
                    build_returns_row(
                        "2024-01-02T00:00:00",
                        close="101",
                        arithmetic_return="0.01",
                        sigma_t="0.20",
                    ),
                    build_returns_row(
                        "2024-01-03T00:00:00",
                        close="102",
                        arithmetic_return="0.02",
                        sigma_t="0.30",
                    ),
                    build_returns_row(
                        "2024-01-04T00:00:00",
                        close="103",
                        arithmetic_return="0.03",
                        sigma_t="0.40",
                    ),
                ],
            )
            write_csv(
                cpd_path,
                CPD_OUTPUT_HEADER,
                [
                    build_cpd_row(
                        "2024-01-01T00:00:00",
                        status="invalid_window",
                    ),
                    build_cpd_row(
                        "2024-01-02T00:00:00",
                        nu="0.1",
                        gamma="0.2",
                    ),
                    build_cpd_row(
                        "2024-01-03T00:00:00",
                        nu="0.3",
                        gamma="0.4",
                        status="fallback_previous",
                    ),
                    build_cpd_row(
                        "2024-01-04T00:00:00",
                        nu="0.5",
                        gamma="0.6",
                    ),
                ],
            )
            write_json(
                manifest_path,
                [
                    make_cpd_manifest_entry(
                        cpd_csv_path="artifacts/features/cpd/lbw_10/TEST_cpd.csv",
                    )
                ],
            )

            artifacts = build_t16_outputs(
                base_input_dir=base_dir,
                returns_input_dir=base_dir,
                cpd_manifest_input=manifest_path,
                output_dir=tmp_path / "artifacts/datasets",
                project_root=tmp_path,
                lbws=(10,),
            )

            self.assertEqual(len(artifacts.joined_feature_paths), 1)
            self.assertEqual(len(artifacts.split_manifest_paths), 1)

            joined_rows = load_joined_feature_csv(artifacts.joined_feature_paths[0])
            split_rows = load_split_manifest_csv(artifacts.split_manifest_paths[0])

            self.assertEqual(list(T16_OUTPUT_HEADER), list(joined_rows[0].to_csv_row().keys()))
            self.assertEqual(len(joined_rows), 3)
            self.assertEqual([row.timeline_index for row in joined_rows], [1, 2, 3])
            self.assertAlmostEqual(joined_rows[0].next_arithmetic_return, 0.02)
            self.assertAlmostEqual(joined_rows[1].next_arithmetic_return, 0.03)
            self.assertIsNone(joined_rows[2].next_arithmetic_return)

            self.assertEqual(len(split_rows), 1)
            self.assertEqual(split_rows[0].usable_row_count, 3)
            self.assertEqual(split_rows[0].train_row_count, 2)
            self.assertEqual(split_rows[0].val_row_count, 1)
            self.assertEqual(list(T16_SPLIT_MANIFEST_HEADER), list(split_rows[0].to_csv_row().keys()))

    def test_build_outputs_against_real_aapl_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "cpd_feature_store_manifest.json"
            write_json(
                manifest_path,
                [
                    {
                        **make_cpd_manifest_entry(
                            asset_id="AAPL",
                            cpd_csv_path="artifacts/features/cpd/lbw_10/AAPL_cpd.csv",
                            row_count=5330,
                            canonical_row_count=5330,
                        ),
                        "first_timestamp": "2007-04-27T00:00:00",
                        "last_timestamp": "2026-02-20T00:00:00",
                    }
                ],
            )

            artifacts = build_t16_outputs(
                base_input_dir=PROJECT_ROOT / "artifacts/features/base",
                returns_input_dir=PROJECT_ROOT / "artifacts/features/base",
                cpd_manifest_input=manifest_path,
                output_dir=tmp_path / "artifacts/datasets",
                project_root=PROJECT_ROOT,
                lbws=(10,),
            )

            joined_rows = load_joined_feature_csv(artifacts.joined_feature_paths[0])
            split_rows = load_split_manifest_csv(artifacts.split_manifest_paths[0])

            self.assertGreater(len(joined_rows), 1000)
            self.assertIsNone(joined_rows[-1].next_arithmetic_return)
            self.assertEqual(split_rows[0].usable_row_count, len(joined_rows))
            self.assertEqual(
                split_rows[0].train_row_count + split_rows[0].val_row_count,
                len(joined_rows),
            )


if __name__ == "__main__":
    unittest.main()
