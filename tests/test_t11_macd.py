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

from lstm_cpd.features.macd import (  # noqa: E402
    T11_OUTPUT_HEADER,
    build_macd_feature_rows,
    build_t11_outputs,
    compute_exponentially_weighted_moving_average,
    compute_normalized_macd_feature,
)
from lstm_cpd.features.returns import CanonicalDailyCloseRecord  # noqa: E402


def make_rows(close_values: list[float]) -> list[CanonicalDailyCloseRecord]:
    return [
        CanonicalDailyCloseRecord(
            timestamp=f"2024-01-{index + 1:02d}T00:00:00",
            asset_id="TEST",
            close_text=format(value, ".17g"),
            close_value=value,
        )
        for index, value in enumerate(close_values)
    ]


class T11MacdTests(unittest.TestCase):
    def test_compute_ema_uses_recursive_alpha_over_period(self) -> None:
        close_values = [100.0, 102.0, 101.0, 105.0]

        actual = compute_exponentially_weighted_moving_average(
            close_values,
            period=4,
        )

        alpha = 0.25
        expected = [100.0]
        for value in close_values[1:]:
            expected.append(alpha * value + (1.0 - alpha) * expected[-1])

        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value)

    def test_normalized_macd_returns_none_when_price_std_is_zero(self) -> None:
        rows = make_rows([100.0] * 400)

        values = compute_normalized_macd_feature(
            rows,
            short_period=8,
            long_period=24,
        )

        self.assertEqual(len(values), 400)
        self.assertTrue(all(value is None for value in values))

    def test_build_rows_keeps_full_timeline_and_expected_warmup(self) -> None:
        rows = make_rows([100.0 + index for index in range(400)])

        output_rows = build_macd_feature_rows(rows)

        self.assertEqual(len(output_rows), 400)
        self.assertEqual(tuple(output_rows[0].keys()), T11_OUTPUT_HEADER)
        self.assertEqual(output_rows[0]["timestamp"], "2024-01-01T00:00:00")
        self.assertEqual(output_rows[0]["asset_id"], "TEST")
        self.assertTrue(all(output_rows[0][field] == "" for field in T11_OUTPUT_HEADER[2:]))
        self.assertTrue(all(output_rows[312][field] == "" for field in T11_OUTPUT_HEADER[2:]))
        self.assertTrue(all(output_rows[313][field] != "" for field in T11_OUTPUT_HEADER[2:]))

    def test_build_outputs_against_real_data(self) -> None:
        manifest_input = (
            PROJECT_ROOT / "artifacts/manifests/canonical_daily_close_manifest.json"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = tmp_path / "artifacts/features/base"

            output_paths = build_t11_outputs(
                canonical_manifest_input=manifest_input,
                output_dir=output_dir,
                project_root=PROJECT_ROOT,
            )

            self.assertEqual(len(output_paths), 126)
            self.assertEqual(len(list(output_dir.glob("*_macd_features.csv"))), 126)

            cocoa_path = output_dir / "COCOA.c_macd_features.csv"
            with cocoa_path.open("r", encoding="utf-8", newline="") as handle:
                cocoa_rows = list(csv.DictReader(handle))
                cocoa_header = list(cocoa_rows[0].keys())

            self.assertEqual(cocoa_header, list(T11_OUTPUT_HEADER))
            self.assertEqual(len(cocoa_rows), 776)
            self.assertEqual(cocoa_rows[0]["timestamp"], "2023-01-19T00:00:00")
            self.assertEqual(cocoa_rows[0]["asset_id"], "COCOA.c")
            self.assertTrue(all(cocoa_rows[0][field] == "" for field in T11_OUTPUT_HEADER[2:]))
            first_non_empty = next(
                row for row in cocoa_rows if row["macd_8_24"] != ""
            )
            self.assertEqual(first_non_empty["timestamp"], "2024-04-18T00:00:00")

            xrp_path = output_dir / "XRPUSD_macd_features.csv"
            with xrp_path.open("r", encoding="utf-8", newline="") as handle:
                xrp_rows = {row["timestamp"]: row for row in csv.DictReader(handle)}
            self.assertTrue(
                all(
                    xrp_rows["2016-05-16T00:00:00"][field] == ""
                    for field in T11_OUTPUT_HEADER[2:]
                )
            )
            self.assertTrue(
                all(
                    xrp_rows["2017-12-23T00:00:00"][field] == ""
                    for field in T11_OUTPUT_HEADER[2:]
                )
            )
            self.assertTrue(
                all(
                    xrp_rows["2017-12-24T00:00:00"][field] != ""
                    for field in T11_OUTPUT_HEADER[2:]
                )
            )


if __name__ == "__main__":
    unittest.main()
