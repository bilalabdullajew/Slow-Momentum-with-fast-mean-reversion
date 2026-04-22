from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.features.normalized_returns import (  # noqa: E402
    build_t10_outputs,
    compute_interval_return,
    compute_normalized_return_features,
    load_returns_volatility_csv,
)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class T10NormalizedReturnsTests(unittest.TestCase):
    def test_compute_interval_return_uses_close_to_close_formula(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "TEST_returns_volatility.csv"
            write_csv(
                path,
                ["timestamp", "asset_id", "close", "arithmetic_return", "sigma_t"],
                [
                    ["2024-01-01T00:00:00", "TEST", "100", "", ""],
                    ["2024-01-02T00:00:00", "TEST", "105", "0.05", "2"],
                    ["2024-01-03T00:00:00", "TEST", "120", "0.14285714285714285", "2"],
                ],
            )
            records = load_returns_volatility_csv(path, expected_asset_id="TEST")

        self.assertIsNone(compute_interval_return(records, 0, 1))
        self.assertAlmostEqual(compute_interval_return(records, 1, 1), 0.05)
        self.assertAlmostEqual(compute_interval_return(records, 2, 2), 0.2)

    def test_compute_normalized_returns_keeps_full_timeline_and_blanks_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "TEST_returns_volatility.csv"
            write_csv(
                path,
                ["timestamp", "asset_id", "close", "arithmetic_return", "sigma_t"],
                [
                    ["2024-01-01T00:00:00", "TEST", "100", "", ""],
                    ["2024-01-02T00:00:00", "TEST", "110", "0.1", "2"],
                    ["2024-01-03T00:00:00", "TEST", "0", "-1", "2"],
                    ["2024-01-04T00:00:00", "TEST", "10", "", "2"],
                    ["2024-01-05T00:00:00", "TEST", "20", "1", "0"],
                ],
            )
            records = load_returns_volatility_csv(path, expected_asset_id="TEST")

        rows = compute_normalized_return_features(records, horizons=(1, 2))

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["normalized_return_1"], "")
        self.assertAlmostEqual(float(rows[1]["normalized_return_1"]), 0.05)
        self.assertAlmostEqual(float(rows[2]["normalized_return_1"]), -0.5)
        self.assertEqual(rows[3]["normalized_return_1"], "")
        self.assertEqual(rows[4]["normalized_return_1"], "")
        self.assertEqual(rows[1]["normalized_return_2"], "")
        self.assertAlmostEqual(
            float(rows[2]["normalized_return_2"]),
            -1.0 / (2.0 * math.sqrt(2.0)),
        )
        self.assertAlmostEqual(
            float(rows[3]["normalized_return_2"]),
            ((10.0 / 110.0) - 1.0) / (2.0 * math.sqrt(2.0)),
        )
        self.assertEqual(rows[4]["normalized_return_2"], "")

    def test_build_outputs_against_real_data(self) -> None:
        input_dir = PROJECT_ROOT / "artifacts/features/base"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "artifacts/features/base"
            output_paths = build_t10_outputs(
                input_dir=input_dir,
                output_dir=output_dir,
            )

            self.assertEqual(len(output_paths), 126)
            self.assertEqual(len(list(output_dir.glob("*_normalized_returns.csv"))), 126)

            cocoa_path = output_dir / "COCOA.c_normalized_returns.csv"
            with cocoa_path.open("r", encoding="utf-8", newline="") as handle:
                cocoa_rows = list(csv.DictReader(handle))
                cocoa_header = list(cocoa_rows[0].keys())

            self.assertEqual(
                cocoa_header,
                [
                    "timestamp",
                    "asset_id",
                    "normalized_return_1",
                    "normalized_return_21",
                    "normalized_return_63",
                    "normalized_return_126",
                    "normalized_return_256",
                ],
            )
            self.assertEqual(len(cocoa_rows), 776)
            self.assertEqual(cocoa_rows[0]["timestamp"], "2023-01-19T00:00:00")
            self.assertEqual(cocoa_rows[0]["asset_id"], "COCOA.c")
            self.assertTrue(
                all(cocoa_rows[0][field] == "" for field in cocoa_header[2:])
            )

            first_h1 = next(row for row in cocoa_rows if row["normalized_return_1"] != "")
            first_h21 = next(row for row in cocoa_rows if row["normalized_return_21"] != "")
            first_h63 = next(row for row in cocoa_rows if row["normalized_return_63"] != "")
            first_h126 = next(row for row in cocoa_rows if row["normalized_return_126"] != "")
            first_h256 = next(row for row in cocoa_rows if row["normalized_return_256"] != "")
            self.assertEqual(first_h1["timestamp"], "2023-04-17T00:00:00")
            self.assertEqual(first_h21["timestamp"], "2023-04-17T00:00:00")
            self.assertEqual(first_h63["timestamp"], "2023-04-20T00:00:00")
            self.assertEqual(first_h126["timestamp"], "2023-07-21T00:00:00")
            self.assertEqual(first_h256["timestamp"], "2024-01-26T00:00:00")

            algod_path = output_dir / "ALGUSD_normalized_returns.csv"
            with algod_path.open("r", encoding="utf-8", newline="") as handle:
                algod_rows = {row["timestamp"]: row for row in csv.DictReader(handle)}
            self.assertEqual(algod_rows["2020-04-19T00:00:00"]["normalized_return_1"], "")
            self.assertEqual(algod_rows["2020-05-09T00:00:00"]["normalized_return_21"], "")
            self.assertEqual(algod_rows["2020-06-20T00:00:00"]["normalized_return_63"], "")
            self.assertEqual(algod_rows["2020-08-26T00:00:00"]["normalized_return_126"], "")
            self.assertEqual(algod_rows["2021-01-26T00:00:00"]["normalized_return_256"], "")

            lnk_path = output_dir / "LNKUSD_normalized_returns.csv"
            with lnk_path.open("r", encoding="utf-8", newline="") as handle:
                lnk_rows = {row["timestamp"]: row for row in csv.DictReader(handle)}
            self.assertEqual(lnk_rows["2020-10-11T00:00:00"]["normalized_return_1"], "")
            self.assertEqual(lnk_rows["2020-10-31T00:00:00"]["normalized_return_21"], "")
            self.assertEqual(lnk_rows["2020-12-12T00:00:00"]["normalized_return_63"], "")
            self.assertEqual(lnk_rows["2021-02-13T00:00:00"]["normalized_return_126"], "")
            self.assertEqual(lnk_rows["2021-06-23T00:00:00"]["normalized_return_256"], "")

            xrp_path = output_dir / "XRPUSD_normalized_returns.csv"
            with xrp_path.open("r", encoding="utf-8", newline="") as handle:
                xrp_rows = {row["timestamp"]: row for row in csv.DictReader(handle)}
            xrp_zero_sigma_row = xrp_rows["2016-08-08T00:00:00"]
            self.assertTrue(
                all(xrp_zero_sigma_row[field] == "" for field in xrp_zero_sigma_row if field.startswith("normalized_return_"))
            )


if __name__ == "__main__":
    unittest.main()
