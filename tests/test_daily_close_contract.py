from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.daily_close_contract import (  # noqa: E402
    PATH_PATTERN_FOREX,
    PATH_PATTERN_STANDARD,
    RESOLUTION_REASON_AMBIGUOUS_D_PATH,
    RESOLUTION_REASON_MISSING_D_PATH,
    SCHEMA_REASON_DUPLICATE_CONFLICT,
    SCHEMA_REASON_MISSING_CLOSE,
    SCHEMA_REASON_MISSING_TIMESTAMP,
    build_t06_outputs,
    inspect_daily_close_file,
    resolve_d_path,
)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class DailyCloseContractTests(unittest.TestCase):
    def test_resolve_standard_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ftmo_root = repo_root / "data/FTMO Data"
            file_path = ftmo_root / "Agriculture/COCOA.c/D/COCOA.c_data.csv"
            write_csv(file_path, ["time", "close"], [["2024-01-01", "1.0"]])

            resolution = resolve_d_path(
                {"asset_id": "COCOA.c", "symbol": "COCOA.c", "category": "Agriculture"},
                ftmo_root=ftmo_root,
                repo_root=repo_root,
            )

        self.assertEqual(resolution.resolution_status, "RESOLVED")
        self.assertEqual(resolution.path_pattern, PATH_PATTERN_STANDARD)
        self.assertEqual(
            resolution.d_file_path,
            "data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
        )

    def test_resolve_forex_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ftmo_root = repo_root / "data/FTMO Data"
            file_path = ftmo_root / "Forex/AUD/AUDCAD/D/AUDCAD_data.csv"
            write_csv(file_path, ["time", "close"], [["2024-01-01", "1.0"]])

            resolution = resolve_d_path(
                {"asset_id": "AUDCAD", "symbol": "AUDCAD", "category": "Forex"},
                ftmo_root=ftmo_root,
                repo_root=repo_root,
            )

        self.assertEqual(resolution.resolution_status, "RESOLVED")
        self.assertEqual(resolution.path_pattern, PATH_PATTERN_FOREX)
        self.assertEqual(
            resolution.d_file_path,
            "data/FTMO Data/Forex/AUD/AUDCAD/D/AUDCAD_data.csv",
        )

    def test_resolve_missing_and_ambiguous_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ftmo_root = repo_root / "data/FTMO Data"
            ambiguous_standard = ftmo_root / "Forex/AUDCAD/D/AUDCAD_data.csv"
            ambiguous_forex = ftmo_root / "Forex/AUD/AUDCAD/D/AUDCAD_data.csv"
            write_csv(ambiguous_standard, ["time", "close"], [["2024-01-01", "1.0"]])
            write_csv(ambiguous_forex, ["time", "close"], [["2024-01-01", "1.0"]])

            missing = resolve_d_path(
                {"asset_id": "EURUSD", "symbol": "EURUSD", "category": "Forex"},
                ftmo_root=ftmo_root,
                repo_root=repo_root,
            )
            ambiguous = resolve_d_path(
                {"asset_id": "AUDCAD", "symbol": "AUDCAD", "category": "Forex"},
                ftmo_root=ftmo_root,
                repo_root=repo_root,
            )

        self.assertEqual(missing.resolution_status, "FAILED")
        self.assertEqual(missing.resolution_failure_reason, RESOLUTION_REASON_MISSING_D_PATH)
        self.assertEqual(ambiguous.resolution_status, "FAILED")
        self.assertEqual(
            ambiguous.resolution_failure_reason, RESOLUTION_REASON_AMBIGUOUS_D_PATH
        )

    def test_schema_inspection_accepts_case_insensitive_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ftmo_root = repo_root / "data/FTMO Data"
            file_path = ftmo_root / "Agriculture/COCOA.c/D/COCOA.c_data.csv"
            write_csv(
                file_path,
                ["DATE", "Open", "Close"],
                [
                    ["2024-01-02", "0.0", "2.0"],
                    ["2024-01-01", "0.0", "1.0"],
                ],
            )
            resolution = resolve_d_path(
                {"asset_id": "COCOA.c", "symbol": "COCOA.c", "category": "Agriculture"},
                ftmo_root=ftmo_root,
                repo_root=repo_root,
            )

            inspection = inspect_daily_close_file(
                {"asset_id": "COCOA.c", "symbol": "COCOA.c", "category": "Agriculture"},
                resolution,
                repo_root=repo_root,
            )

        self.assertEqual(inspection.schema_status, "ADMISSIBLE")
        self.assertEqual(inspection.timestamp_column, "DATE")
        self.assertEqual(inspection.close_column, "Close")
        self.assertEqual(inspection.source_sorted_ascending, "false")
        self.assertEqual(inspection.canonical_first_timestamp, "2024-01-01")
        self.assertEqual(inspection.canonical_last_timestamp, "2024-01-02")

    def test_schema_inspection_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ftmo_root = repo_root / "data/FTMO Data"
            missing_timestamp_path = ftmo_root / "Agriculture/COCOA.c/D/COCOA.c_data.csv"
            write_csv(missing_timestamp_path, ["open", "close"], [["1", "2"]])
            resolution = resolve_d_path(
                {"asset_id": "COCOA.c", "symbol": "COCOA.c", "category": "Agriculture"},
                ftmo_root=ftmo_root,
                repo_root=repo_root,
            )
            missing_timestamp = inspect_daily_close_file(
                {"asset_id": "COCOA.c", "symbol": "COCOA.c", "category": "Agriculture"},
                resolution,
                repo_root=repo_root,
            )

            missing_close_path = ftmo_root / "Agriculture/COFFEE.c/D/COFFEE.c_data.csv"
            write_csv(missing_close_path, ["time", "open"], [["2024-01-01", "2"]])
            resolution_two = resolve_d_path(
                {"asset_id": "COFFEE.c", "symbol": "COFFEE.c", "category": "Agriculture"},
                ftmo_root=ftmo_root,
                repo_root=repo_root,
            )
            missing_close = inspect_daily_close_file(
                {"asset_id": "COFFEE.c", "symbol": "COFFEE.c", "category": "Agriculture"},
                resolution_two,
                repo_root=repo_root,
            )

        self.assertEqual(missing_timestamp.schema_reason_code, SCHEMA_REASON_MISSING_TIMESTAMP)
        self.assertEqual(missing_close.schema_reason_code, SCHEMA_REASON_MISSING_CLOSE)

    def test_schema_inspection_handles_identical_and_conflicting_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ftmo_root = repo_root / "data/FTMO Data"
            identical_path = ftmo_root / "Agriculture/COCOA.c/D/COCOA.c_data.csv"
            write_csv(
                identical_path,
                ["time", "close"],
                [
                    ["2024-01-01", "1.0"],
                    ["2024-01-01", "1.00"],
                    ["2024-01-02", "2.0"],
                ],
            )
            identical_resolution = resolve_d_path(
                {"asset_id": "COCOA.c", "symbol": "COCOA.c", "category": "Agriculture"},
                ftmo_root=ftmo_root,
                repo_root=repo_root,
            )
            identical = inspect_daily_close_file(
                {"asset_id": "COCOA.c", "symbol": "COCOA.c", "category": "Agriculture"},
                identical_resolution,
                repo_root=repo_root,
            )

            conflict_path = ftmo_root / "Agriculture/COFFEE.c/D/COFFEE.c_data.csv"
            write_csv(
                conflict_path,
                ["time", "close"],
                [
                    ["2024-01-01", "1.0"],
                    ["2024-01-01", "2.0"],
                ],
            )
            conflict_resolution = resolve_d_path(
                {"asset_id": "COFFEE.c", "symbol": "COFFEE.c", "category": "Agriculture"},
                ftmo_root=ftmo_root,
                repo_root=repo_root,
            )
            conflict = inspect_daily_close_file(
                {"asset_id": "COFFEE.c", "symbol": "COFFEE.c", "category": "Agriculture"},
                conflict_resolution,
                repo_root=repo_root,
            )

        self.assertEqual(identical.schema_status, "ADMISSIBLE")
        self.assertEqual(identical.duplicate_identical_count, 1)
        self.assertEqual(identical.canonical_row_count, 2)
        self.assertEqual(conflict.schema_status, "EXCLUDED")
        self.assertEqual(conflict.schema_reason_code, SCHEMA_REASON_DUPLICATE_CONFLICT)

    def test_build_outputs_against_real_data(self) -> None:
        repo_root = PROJECT_ROOT.parents[2]
        asset_manifest_path = (
            PROJECT_ROOT / "artifacts/manifests/ftmo_asset_universe.json"
        )
        ftmo_root = repo_root / "data/FTMO Data"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            path_manifest_output = tmp_path / "d_timeframe_path_manifest.json"
            contract_output = tmp_path / "daily_close_schema_contract.md"
            schema_report_output = tmp_path / "schema_inspection_report.csv"

            resolutions, inspections = build_t06_outputs(
                asset_manifest_path=asset_manifest_path,
                ftmo_root=ftmo_root,
                path_manifest_output=path_manifest_output,
                contract_output=contract_output,
                schema_report_output=schema_report_output,
                repo_root=repo_root,
            )

            manifest_rows = json.loads(path_manifest_output.read_text(encoding="utf-8"))
            with schema_report_output.open("r", encoding="utf-8", newline="") as handle:
                report_rows = list(csv.DictReader(handle))
            contract_text = contract_output.read_text(encoding="utf-8")

        self.assertEqual(len(resolutions), 131)
        self.assertEqual(len(inspections), 131)
        self.assertEqual(len(manifest_rows), 131)
        self.assertEqual(len(report_rows), 131)
        self.assertTrue(all(row["resolution_status"] == "RESOLVED" for row in manifest_rows))
        self.assertTrue(all(row["schema_status"] == "ADMISSIBLE" for row in report_rows))
        self.assertTrue(
            all(
                row["path_pattern"] == PATH_PATTERN_FOREX
                for row in manifest_rows
                if row["category"] == "Forex"
            )
        )
        self.assertTrue(
            all(
                row["path_pattern"] == PATH_PATTERN_STANDARD
                for row in manifest_rows
                if row["category"] != "Forex"
            )
        )
        self.assertIn("DUPLICATE_TIMESTAMP_CONFLICT", contract_text)
        self.assertIn("timestamp", contract_text)
        self.assertIn("close", contract_text)


if __name__ == "__main__":
    unittest.main()
