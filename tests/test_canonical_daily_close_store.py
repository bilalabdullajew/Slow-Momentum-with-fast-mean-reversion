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

from lstm_cpd.canonical_daily_close_store import (  # noqa: E402
    build_t08_outputs,
    serialize_canonical_daily_close_csv_bytes,
    sha256_prefixed,
)
from lstm_cpd.daily_close_contract import CanonicalDailyCloseRow, parse_timestamp_value  # noqa: E402


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class CanonicalDailyCloseStoreTests(unittest.TestCase):
    def test_serialize_canonical_csv_normalizes_timestamp_and_close(self) -> None:
        rows = [
            CanonicalDailyCloseRow(
                timestamp="2024-01-01",
                close=1.0,
                parsed_timestamp=parse_timestamp_value("2024-01-01"),
            ),
            CanonicalDailyCloseRow(
                timestamp="2024-01-02",
                close=0.9514,
                parsed_timestamp=parse_timestamp_value("2024-01-02"),
            ),
        ]

        payload = serialize_canonical_daily_close_csv_bytes("COCOA.c", rows)

        self.assertEqual(
            payload.decode("utf-8"),
            (
                "timestamp,asset_id,close\n"
                "2024-01-01T00:00:00,COCOA.c,1\n"
                "2024-01-02T00:00:00,COCOA.c,0.9514\n"
            ),
        )
        self.assertTrue(sha256_prefixed(payload).startswith("sha256:"))

    def test_build_outputs_rejects_unresolved_eligible_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            eligibility_report = tmp_path / "asset_eligibility_report.csv"
            path_manifest = tmp_path / "d_timeframe_path_manifest.json"
            contract_input = tmp_path / "daily_close_schema_contract.md"
            canonical_output_dir = tmp_path / "canonical"
            manifest_output = tmp_path / "canonical_manifest.json"

            write_csv(
                eligibility_report,
                [
                    "asset_id",
                    "symbol",
                    "category",
                    "path_pattern",
                    "d_file_path",
                    "raw_row_count",
                    "screened_row_count",
                    "first_timestamp",
                    "last_timestamp",
                ],
                [
                    [
                        "COCOA.c",
                        "COCOA.c",
                        "Agriculture",
                        "category_symbol_d",
                        "data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
                        "10",
                        "10",
                        "2024-01-01",
                        "2024-01-10",
                    ]
                ],
            )
            write_json(
                path_manifest,
                [
                    {
                        "asset_id": "COCOA.c",
                        "symbol": "COCOA.c",
                        "category": "Agriculture",
                        "resolution_status": "FAILED",
                        "resolution_failure_reason": "MISSING_D_PATH",
                        "path_pattern": None,
                        "d_file_path": None,
                        "candidate_paths": [],
                    }
                ],
            )
            contract_input.write_text("contract", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must resolve"):
                build_t08_outputs(
                    eligibility_report_input=eligibility_report,
                    path_manifest_input=path_manifest,
                    contract_input=contract_input,
                    canonical_output_dir=canonical_output_dir,
                    manifest_output=manifest_output,
                    repo_root=tmp_path,
                    project_root=tmp_path,
                )

    def test_build_outputs_rejects_schema_regression_for_eligible_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            eligibility_report = tmp_path / "asset_eligibility_report.csv"
            path_manifest = tmp_path / "d_timeframe_path_manifest.json"
            contract_input = tmp_path / "daily_close_schema_contract.md"
            canonical_output_dir = tmp_path / "canonical"
            manifest_output = tmp_path / "canonical_manifest.json"
            raw_file = tmp_path / "data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv"

            write_csv(
                eligibility_report,
                [
                    "asset_id",
                    "symbol",
                    "category",
                    "path_pattern",
                    "d_file_path",
                    "raw_row_count",
                    "screened_row_count",
                    "first_timestamp",
                    "last_timestamp",
                ],
                [
                    [
                        "COCOA.c",
                        "COCOA.c",
                        "Agriculture",
                        "category_symbol_d",
                        "data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
                        "2",
                        "2",
                        "2024-01-01",
                        "2024-01-02",
                    ]
                ],
            )
            write_json(
                path_manifest,
                [
                    {
                        "asset_id": "COCOA.c",
                        "symbol": "COCOA.c",
                        "category": "Agriculture",
                        "resolution_status": "RESOLVED",
                        "resolution_failure_reason": None,
                        "path_pattern": "category_symbol_d",
                        "d_file_path": "data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv",
                        "candidate_paths": [
                            "data/FTMO Data/Agriculture/COCOA.c/D/COCOA.c_data.csv"
                        ],
                    }
                ],
            )
            write_csv(
                raw_file,
                ["time", "close"],
                [["2024-01-01", "1.0"], ["2024-01-01", "2.0"]],
            )
            contract_input.write_text("contract", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "failed schema admission"):
                build_t08_outputs(
                    eligibility_report_input=eligibility_report,
                    path_manifest_input=path_manifest,
                    contract_input=contract_input,
                    canonical_output_dir=canonical_output_dir,
                    manifest_output=manifest_output,
                    repo_root=tmp_path,
                    project_root=tmp_path,
                )

    def test_build_outputs_against_real_data(self) -> None:
        repo_root = PROJECT_ROOT.parents[2]
        eligibility_report = (
            PROJECT_ROOT / "artifacts/reports/asset_eligibility_report.csv"
        )
        path_manifest = PROJECT_ROOT / "artifacts/manifests/d_timeframe_path_manifest.json"
        contract_input = PROJECT_ROOT / "docs/contracts/daily_close_schema_contract.md"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            canonical_output_dir = tmp_path / "artifacts/canonical_daily_close"
            manifest_output = tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"

            records = build_t08_outputs(
                eligibility_report_input=eligibility_report,
                path_manifest_input=path_manifest,
                contract_input=contract_input,
                canonical_output_dir=canonical_output_dir,
                manifest_output=manifest_output,
                repo_root=repo_root,
                project_root=tmp_path,
            )

            manifest_rows = json.loads(manifest_output.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 126)
            self.assertEqual(len(manifest_rows), 126)
            self.assertEqual(len(list(canonical_output_dir.glob("*.csv"))), 126)
            self.assertFalse((canonical_output_dir / "COTTON.c.csv").exists())
            self.assertFalse((canonical_output_dir / "SUGAR.c.csv").exists())
            self.assertFalse((canonical_output_dir / "HEATOIL.c.csv").exists())
            self.assertFalse((canonical_output_dir / "SOLUSD.csv").exists())
            self.assertFalse((canonical_output_dir / "XCUUSD.csv").exists())

            for record in manifest_rows:
                csv_path = tmp_path / record["canonical_csv_path"]
                self.assertTrue(csv_path.exists())
                self.assertTrue(csv_path.name.endswith(".csv"))

            sample_path = canonical_output_dir / "COCOA.c.csv"
            with sample_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, ["timestamp", "asset_id", "close"])
            self.assertEqual(rows[0]["asset_id"], "COCOA.c")
            self.assertEqual(rows[0]["timestamp"], "2023-01-19T00:00:00")
            self.assertEqual(rows[-1]["timestamp"], "2026-02-20T00:00:00")
            manifest_by_asset = {record["asset_id"]: record for record in manifest_rows}
            self.assertEqual(manifest_by_asset["COCOA.c"]["row_count"], 776)
            self.assertEqual(
                manifest_by_asset["COCOA.c"]["first_timestamp"], "2023-01-19T00:00:00"
            )
            self.assertEqual(
                manifest_by_asset["COCOA.c"]["last_timestamp"], "2026-02-20T00:00:00"
            )


if __name__ == "__main__":
    unittest.main()
