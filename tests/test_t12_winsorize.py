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

from lstm_cpd.features.winsorize import (  # noqa: E402
    BASE_FEATURES_SUFFIX,
    FEATURE_COLUMNS,
    T12_OUTPUT_HEADER,
    build_base_feature_rows,
    build_t12_outputs,
    join_feature_rows,
    winsorize_feature_values,
)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def make_normalized_row(
    timestamp: str,
    asset_id: str = "TEST",
    values: tuple[str, str, str, str, str] = ("", "", "", "", ""),
) -> dict[str, str]:
    return {
        "timestamp": timestamp,
        "asset_id": asset_id,
        "normalized_return_1": values[0],
        "normalized_return_21": values[1],
        "normalized_return_63": values[2],
        "normalized_return_126": values[3],
        "normalized_return_256": values[4],
    }


def make_macd_row(
    timestamp: str,
    asset_id: str = "TEST",
    values: tuple[str, str, str] = ("", "", ""),
) -> dict[str, str]:
    return {
        "timestamp": timestamp,
        "asset_id": asset_id,
        "macd_8_24": values[0],
        "macd_16_28": values[1],
        "macd_32_96": values[2],
    }


class T12WinsorizeTests(unittest.TestCase):
    def test_winsorize_feature_values_clips_outlier_and_preserves_missing(self) -> None:
        values = [None, 1.0, 2.0, 100.0]

        actual = winsorize_feature_values(values, halflife=1.0, sigma_multiple=0.5)

        self.assertIsNone(actual[0])
        self.assertEqual(actual[1], 1.0)
        expected_series = pd.Series(values, dtype="float64")
        expected_mean = expected_series.ewm(halflife=1.0, adjust=False).mean()
        expected_std = expected_series.ewm(halflife=1.0, adjust=False).std(bias=False)
        expected_upper = float(expected_mean.iloc[3] + 0.5 * expected_std.iloc[3])
        self.assertAlmostEqual(actual[3], expected_upper)
        self.assertLess(actual[3], 100.0)

    def test_join_feature_rows_rejects_alignment_mismatch(self) -> None:
        normalized_rows = [make_normalized_row("2024-01-01T00:00:00")]
        macd_rows = [make_macd_row("2024-01-02T00:00:00")]

        with self.assertRaisesRegex(ValueError, "Feature alignment mismatch"):
            join_feature_rows(normalized_rows, macd_rows)

    def test_build_base_feature_rows_keeps_full_timeline_and_blanks(self) -> None:
        normalized_rows = [
            make_normalized_row("2024-01-01T00:00:00"),
            make_normalized_row(
                "2024-01-02T00:00:00",
                values=("1.5", "", "", "", ""),
            ),
        ]
        macd_rows = [
            make_macd_row("2024-01-01T00:00:00"),
            make_macd_row(
                "2024-01-02T00:00:00",
                values=("", "2.5", ""),
            ),
        ]

        joined_rows = join_feature_rows(normalized_rows, macd_rows)
        output_rows = build_base_feature_rows(
            joined_rows,
            halflife=1.0,
            sigma_multiple=1.0,
        )

        self.assertEqual(len(output_rows), 2)
        self.assertEqual(tuple(output_rows[0].keys()), T12_OUTPUT_HEADER)
        self.assertTrue(all(output_rows[0][column] == "" for column in FEATURE_COLUMNS))
        self.assertEqual(output_rows[1]["normalized_return_1"], "1.5")
        self.assertEqual(output_rows[1]["macd_16_28"], "2.5")
        self.assertEqual(output_rows[1]["macd_8_24"], "")

    def test_build_outputs_against_real_data(self) -> None:
        input_dir = PROJECT_ROOT / "artifacts/features/base"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = tmp_path / "artifacts/features/base"
            report_path = tmp_path / "artifacts/reports/feature_provenance_report.md"

            output_paths = build_t12_outputs(
                input_dir=input_dir,
                output_dir=output_dir,
                report_path=report_path,
            )

            self.assertEqual(len(output_paths), 126)
            self.assertEqual(len(list(output_dir.glob(f"*{BASE_FEATURES_SUFFIX}"))), 126)
            self.assertTrue(report_path.exists())

            cocoa_path = output_dir / "COCOA.c_base_features.csv"
            with cocoa_path.open("r", encoding="utf-8", newline="") as handle:
                cocoa_rows = list(csv.DictReader(handle))
                cocoa_header = list(cocoa_rows[0].keys())

            self.assertEqual(cocoa_header, list(T12_OUTPUT_HEADER))
            self.assertEqual(len(cocoa_rows), 776)
            self.assertEqual(cocoa_rows[0]["timestamp"], "2023-01-19T00:00:00")
            self.assertEqual(cocoa_rows[0]["asset_id"], "COCOA.c")
            self.assertTrue(all(cocoa_rows[0][field] == "" for field in T12_OUTPUT_HEADER[2:]))
            first_non_empty_row = next(
                row
                for row in cocoa_rows
                if any(row[field] != "" for field in T12_OUTPUT_HEADER[2:])
            )
            self.assertEqual(first_non_empty_row["timestamp"], "2023-04-17T00:00:00")

            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("Generated asset count: 126", report_text)
            self.assertIn("`{1,21,63,126,256}`", report_text)
            self.assertIn("`{(8,24),(16,28),(32,96)}`", report_text)
            self.assertIn("Winsorization applies only to the 8 non-CPD features.", report_text)


if __name__ == "__main__":
    unittest.main()
