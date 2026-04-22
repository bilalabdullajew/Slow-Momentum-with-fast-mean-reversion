from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.features.returns import (  # noqa: E402
    CanonicalDailyCloseRecord,
    compute_arithmetic_returns,
)
from lstm_cpd.features.volatility import (  # noqa: E402
    build_returns_volatility_rows,
    build_t09_outputs,
    compute_ewm_volatility,
)


class T09ReturnsVolatilityTests(unittest.TestCase):
    def test_compute_arithmetic_returns_uses_exact_formula(self) -> None:
        rows = [
            CanonicalDailyCloseRecord("2024-01-01T00:00:00", "TEST", "100", 100.0),
            CanonicalDailyCloseRecord("2024-01-02T00:00:00", "TEST", "110", 110.0),
            CanonicalDailyCloseRecord("2024-01-03T00:00:00", "TEST", "0", 0.0),
            CanonicalDailyCloseRecord("2024-01-04T00:00:00", "TEST", "10", 10.0),
            CanonicalDailyCloseRecord("2024-01-05T00:00:00", "TEST", "20", 20.0),
        ]

        arithmetic_returns = compute_arithmetic_returns(rows)

        self.assertEqual(arithmetic_returns[0], None)
        self.assertAlmostEqual(arithmetic_returns[1], 0.1)
        self.assertEqual(arithmetic_returns[2], -1.0)
        self.assertEqual(arithmetic_returns[3], None)
        self.assertEqual(arithmetic_returns[4], 1.0)

    def test_compute_ewm_volatility_matches_frozen_pandas_call(self) -> None:
        arithmetic_returns = [None, 0.1, 0.2, 0.3, None, 0.4]

        actual = compute_ewm_volatility(
            arithmetic_returns,
            span=3,
            adjust=False,
            min_periods=3,
            bias=False,
        )
        expected = (
            pd.Series(arithmetic_returns, dtype="float64")
            .ewm(span=3, adjust=False, min_periods=3)
            .std(bias=False)
            .tolist()
        )

        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            if pd.isna(expected_value):
                self.assertEqual(actual_value, None)
            else:
                self.assertAlmostEqual(actual_value, float(expected_value))

    def test_build_rows_keeps_full_timeline_and_blank_warmup(self) -> None:
        rows = [
            CanonicalDailyCloseRecord("2024-01-01T00:00:00", "TEST", "100", 100.0),
            CanonicalDailyCloseRecord("2024-01-02T00:00:00", "TEST", "110", 110.0),
            CanonicalDailyCloseRecord("2024-01-03T00:00:00", "TEST", "0", 0.0),
            CanonicalDailyCloseRecord("2024-01-04T00:00:00", "TEST", "10", 10.0),
        ]

        output_rows = build_returns_volatility_rows(
            rows,
            span=2,
            adjust=False,
            min_periods=2,
            bias=False,
        )

        self.assertEqual(len(output_rows), 4)
        self.assertEqual(output_rows[0]["timestamp"], "2024-01-01T00:00:00")
        self.assertEqual(output_rows[0]["close"], "100")
        self.assertEqual(output_rows[0]["arithmetic_return"], "")
        self.assertEqual(output_rows[0]["sigma_t"], "")
        self.assertEqual(output_rows[2]["arithmetic_return"], "-1")
        self.assertNotEqual(output_rows[2]["sigma_t"], "")
        self.assertEqual(output_rows[3]["arithmetic_return"], "")
        self.assertEqual(output_rows[3]["sigma_t"], output_rows[2]["sigma_t"])

    def test_build_outputs_against_real_data(self) -> None:
        manifest_input = (
            PROJECT_ROOT / "artifacts/manifests/canonical_daily_close_manifest.json"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = tmp_path / "artifacts/features/base"

            output_paths = build_t09_outputs(
                canonical_manifest_input=manifest_input,
                output_dir=output_dir,
                project_root=PROJECT_ROOT,
            )

            self.assertEqual(len(output_paths), 126)
            self.assertEqual(len(list(output_dir.glob("*_returns_volatility.csv"))), 126)

            cocoa_path = output_dir / "COCOA.c_returns_volatility.csv"
            with cocoa_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                cocoa_rows = list(reader)

            self.assertEqual(
                reader.fieldnames,
                ["timestamp", "asset_id", "close", "arithmetic_return", "sigma_t"],
            )
            self.assertEqual(len(cocoa_rows), 776)
            self.assertEqual(cocoa_rows[0]["timestamp"], "2023-01-19T00:00:00")
            self.assertEqual(cocoa_rows[0]["asset_id"], "COCOA.c")
            self.assertEqual(cocoa_rows[0]["close"], "2596")
            self.assertEqual(cocoa_rows[0]["arithmetic_return"], "")
            self.assertEqual(cocoa_rows[0]["sigma_t"], "")
            self.assertNotEqual(cocoa_rows[1]["arithmetic_return"], "")
            first_sigma_row = next(row for row in cocoa_rows if row["sigma_t"] != "")
            self.assertEqual(first_sigma_row["timestamp"], "2023-04-17T00:00:00")

            algod_path = output_dir / "ALGUSD_returns_volatility.csv"
            with algod_path.open("r", encoding="utf-8", newline="") as handle:
                algod_rows = list(csv.DictReader(handle))
            algod_by_timestamp = {row["timestamp"]: row for row in algod_rows}
            self.assertEqual(
                algod_by_timestamp["2020-04-18T00:00:00"]["arithmetic_return"], "-1"
            )
            self.assertEqual(
                algod_by_timestamp["2020-04-19T00:00:00"]["arithmetic_return"], ""
            )
            self.assertEqual(
                algod_by_timestamp["2020-04-19T00:00:00"]["sigma_t"],
                algod_by_timestamp["2020-04-18T00:00:00"]["sigma_t"],
            )

            lnk_path = output_dir / "LNKUSD_returns_volatility.csv"
            with lnk_path.open("r", encoding="utf-8", newline="") as handle:
                lnk_rows = list(csv.DictReader(handle))
            lnk_by_timestamp = {row["timestamp"]: row for row in lnk_rows}
            self.assertEqual(
                lnk_by_timestamp["2020-10-10T00:00:00"]["arithmetic_return"], "-1"
            )
            self.assertEqual(
                lnk_by_timestamp["2020-10-11T00:00:00"]["arithmetic_return"], ""
            )
            self.assertEqual(
                lnk_by_timestamp["2020-10-11T00:00:00"]["sigma_t"],
                lnk_by_timestamp["2020-10-10T00:00:00"]["sigma_t"],
            )


if __name__ == "__main__":
    unittest.main()
