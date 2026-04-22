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

from lstm_cpd.ftmo_asset_universe import (  # noqa: E402
    build_ftmo_asset_universe,
    parse_ftmo_asset_document,
)


class FtmoAssetUniverseTests(unittest.TestCase):
    def test_parse_preserves_categories_symbols_and_order(self) -> None:
        fixture = "\n".join(
            [
                "# FTMO Assets nach Kategorie",
                "",
                "## Category One",
                "- AAA",
                "- BBB",
                "",
                "## Category Two",
                "- CCC",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "assets.md"
            input_path.write_text(fixture, encoding="utf-8")

            records = parse_ftmo_asset_document(input_path)

        self.assertEqual(
            [(record.category, record.symbol, record.asset_id) for record in records],
            [
                ("Category One", "AAA", "AAA"),
                ("Category One", "BBB", "BBB"),
                ("Category Two", "CCC", "CCC"),
            ],
        )

    def test_parse_rejects_duplicate_symbols(self) -> None:
        fixture = "\n".join(
            [
                "## Category",
                "- AAA",
                "- AAA",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "assets.md"
            input_path.write_text(fixture, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate asset symbol"):
                parse_ftmo_asset_document(input_path)

    def test_parse_rejects_asset_before_first_category(self) -> None:
        fixture = "- AAA\n## Category\n- BBB\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "assets.md"
            input_path.write_text(fixture, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "before first category"):
                parse_ftmo_asset_document(input_path)

    def test_build_real_source_document_to_json_and_csv(self) -> None:
        input_path = PROJECT_ROOT.parents[2] / "data/FTMO Data/ftmo_assets_nach_kategorie.md"
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output_path = Path(tmpdir) / "ftmo_asset_universe.json"
            csv_output_path = Path(tmpdir) / "ftmo_asset_universe.csv"

            records = build_ftmo_asset_universe(
                input_path=input_path,
                json_output_path=json_output_path,
                csv_output_path=csv_output_path,
            )

            json_rows = json.loads(json_output_path.read_text(encoding="utf-8"))
            with csv_output_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(records), 131)
        self.assertEqual(records[0].category, "Agriculture")
        self.assertEqual(records[0].symbol, "COCOA.c")
        self.assertEqual(records[-1].category, "Metals CFD")
        self.assertEqual(records[-1].symbol, "XPTUSD")
        self.assertTrue(all(record.asset_id == record.symbol for record in records))
        self.assertEqual(
            json_rows,
            [
                {
                    "asset_id": record.asset_id,
                    "symbol": record.symbol,
                    "category": record.category,
                }
                for record in records
            ],
        )
        self.assertEqual(csv_rows, json_rows)


if __name__ == "__main__":
    unittest.main()
