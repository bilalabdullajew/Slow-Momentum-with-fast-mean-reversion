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

from lstm_cpd.daily_close_contract import PathResolutionRecord  # noqa: E402
from lstm_cpd.raw_history_screening import (  # noqa: E402
    MINIMUM_RAW_HISTORY_OBSERVATIONS,
    REASON_DUPLICATE_TIMESTAMP_CONFLICT,
    REASON_EMPTY_SERIES,
    REASON_INSUFFICIENT_RAW_HISTORY,
    REASON_MISSING_FILE,
    REASON_SCHEMA_FAILURE,
    REASON_UNREADABLE_FILE,
    build_t07_outputs,
    screen_path_resolution_record,
)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class RawHistoryScreeningTests(unittest.TestCase):
    def test_resolution_failure_maps_to_missing_file(self) -> None:
        record = PathResolutionRecord(
            asset_id="EURUSD",
            symbol="EURUSD",
            category="Forex",
            resolution_status="FAILED",
            resolution_failure_reason="MISSING_D_PATH",
            path_pattern=None,
            d_file_path=None,
            candidate_paths=[],
        )

        screening = screen_path_resolution_record(record, repo_root=Path("/tmp"))

        self.assertEqual(screening.raw_eligibility_status, "EXCLUDED")
        self.assertEqual(screening.reason_code, REASON_MISSING_FILE)

    def test_resolution_failure_maps_non_missing_to_schema_failure(self) -> None:
        record = PathResolutionRecord(
            asset_id="EURUSD",
            symbol="EURUSD",
            category="Forex",
            resolution_status="FAILED",
            resolution_failure_reason="AMBIGUOUS_D_PATH",
            path_pattern=None,
            d_file_path=None,
            candidate_paths=[],
        )

        screening = screen_path_resolution_record(record, repo_root=Path("/tmp"))

        self.assertEqual(screening.raw_eligibility_status, "EXCLUDED")
        self.assertEqual(screening.reason_code, REASON_SCHEMA_FAILURE)

    def test_missing_resolved_file_maps_to_missing_file(self) -> None:
        record = PathResolutionRecord(
            asset_id="COCOA.c",
            symbol="COCOA.c",
            category="Agriculture",
            resolution_status="RESOLVED",
            resolution_failure_reason=None,
            path_pattern="category_symbol_d",
            d_file_path="data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
            candidate_paths=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            screening = screen_path_resolution_record(record, repo_root=Path(tmpdir))

        self.assertEqual(screening.reason_code, REASON_MISSING_FILE)

    def test_unreadable_file_maps_to_unreadable(self) -> None:
        record = PathResolutionRecord(
            asset_id="COCOA.c",
            symbol="COCOA.c",
            category="Agriculture",
            resolution_status="RESOLVED",
            resolution_failure_reason=None,
            path_pattern="category_symbol_d",
            d_file_path="data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
            candidate_paths=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            bad_path = repo_root / record.d_file_path
            bad_path.mkdir(parents=True)
            screening = screen_path_resolution_record(record, repo_root=repo_root)

        self.assertEqual(screening.reason_code, REASON_UNREADABLE_FILE)

    def test_empty_series_maps_to_empty_series(self) -> None:
        record = PathResolutionRecord(
            asset_id="COCOA.c",
            symbol="COCOA.c",
            category="Agriculture",
            resolution_status="RESOLVED",
            resolution_failure_reason=None,
            path_pattern="category_symbol_d",
            d_file_path="data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
            candidate_paths=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            write_csv(repo_root / record.d_file_path, ["time", "close"], [])
            screening = screen_path_resolution_record(record, repo_root=repo_root)

        self.assertEqual(screening.reason_code, REASON_EMPTY_SERIES)

    def test_duplicate_conflict_maps_to_duplicate_reason(self) -> None:
        record = PathResolutionRecord(
            asset_id="COCOA.c",
            symbol="COCOA.c",
            category="Agriculture",
            resolution_status="RESOLVED",
            resolution_failure_reason=None,
            path_pattern="category_symbol_d",
            d_file_path="data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
            candidate_paths=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            write_csv(
                repo_root / record.d_file_path,
                ["time", "close"],
                [["2024-01-01", "1.0"], ["2024-01-01", "2.0"]],
            )
            screening = screen_path_resolution_record(record, repo_root=repo_root)

        self.assertEqual(screening.reason_code, REASON_DUPLICATE_TIMESTAMP_CONFLICT)

    def test_short_series_maps_to_insufficient_history(self) -> None:
        record = PathResolutionRecord(
            asset_id="COCOA.c",
            symbol="COCOA.c",
            category="Agriculture",
            resolution_status="RESOLVED",
            resolution_failure_reason=None,
            path_pattern="category_symbol_d",
            d_file_path="data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
            candidate_paths=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            rows = [[f"2024-01-{index:02d}", str(index)] for index in range(1, 11)]
            write_csv(repo_root / record.d_file_path, ["time", "close"], rows)
            screening = screen_path_resolution_record(record, repo_root=repo_root)

        self.assertEqual(screening.reason_code, REASON_INSUFFICIENT_RAW_HISTORY)

    def test_long_series_maps_to_eligible(self) -> None:
        record = PathResolutionRecord(
            asset_id="COCOA.c",
            symbol="COCOA.c",
            category="Agriculture",
            resolution_status="RESOLVED",
            resolution_failure_reason=None,
            path_pattern="category_symbol_d",
            d_file_path="data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
            candidate_paths=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            start = datetime(2024, 1, 1)
            rows = [
                [(start + timedelta(days=index)).isoformat(), str(index)]
                for index in range(MINIMUM_RAW_HISTORY_OBSERVATIONS)
            ]
            write_csv(repo_root / record.d_file_path, ["time", "close"], rows)
            screening = screen_path_resolution_record(record, repo_root=repo_root)

        self.assertEqual(screening.raw_eligibility_status, "ELIGIBLE")
        self.assertEqual(screening.reason_code, "")
        self.assertEqual(screening.screened_row_count, MINIMUM_RAW_HISTORY_OBSERVATIONS)

    def test_build_outputs_against_real_data(self) -> None:
        repo_root = PROJECT_ROOT.parents[2]
        path_manifest_input = (
            PROJECT_ROOT / "artifacts/manifests/d_timeframe_path_manifest.json"
        )
        contract_input = PROJECT_ROOT / "docs/contracts/daily_close_schema_contract.md"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            eligibility_output = tmp_path / "asset_eligibility_report.csv"
            exclusion_output = tmp_path / "asset_exclusion_report.csv"
            screening_output = tmp_path / "minimum_history_screening_report.csv"

            records = build_t07_outputs(
                path_manifest_input=path_manifest_input,
                contract_input=contract_input,
                eligibility_report_output=eligibility_output,
                exclusion_report_output=exclusion_output,
                screening_report_output=screening_output,
                repo_root=repo_root,
            )

            with eligibility_output.open("r", encoding="utf-8", newline="") as handle:
                eligibility_rows = list(csv.DictReader(handle))
            with exclusion_output.open("r", encoding="utf-8", newline="") as handle:
                exclusion_rows = list(csv.DictReader(handle))
            with screening_output.open("r", encoding="utf-8", newline="") as handle:
                screening_rows = list(csv.DictReader(handle))

        self.assertEqual(len(records), 131)
        self.assertEqual(len(screening_rows), 131)
        self.assertEqual(len(eligibility_rows), 126)
        self.assertEqual(len(exclusion_rows), 5)
        self.assertEqual(
            {row["asset_id"] for row in exclusion_rows},
            {"COTTON.c", "SUGAR.c", "HEATOIL.c", "SOLUSD", "XCUUSD"},
        )
        self.assertTrue(
            all(row["reason_code"] == REASON_INSUFFICIENT_RAW_HISTORY for row in exclusion_rows)
        )
        screening_by_asset_id = {
            row["asset_id"]: row for row in screening_rows
        }
        self.assertTrue(
            all(
                screening_by_asset_id[row["asset_id"]]["raw_eligibility_status"] == "ELIGIBLE"
                for row in eligibility_rows
            )
        )


if __name__ == "__main__":
    unittest.main()
