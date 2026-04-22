from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AssetUniverseRecord:
    asset_id: str
    symbol: str
    category: str


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_repo_root() -> Path:
    return default_project_root().parents[2]


def default_input_path() -> Path:
    return default_repo_root() / "data/FTMO Data/ftmo_assets_nach_kategorie.md"


def default_json_output_path() -> Path:
    return default_project_root() / "artifacts/manifests/ftmo_asset_universe.json"


def default_csv_output_path() -> Path:
    return default_project_root() / "artifacts/manifests/ftmo_asset_universe.csv"


def parse_ftmo_asset_document(path: Path | str) -> list[AssetUniverseRecord]:
    source_path = Path(path)
    category: str | None = None
    records: list[AssetUniverseRecord] = []
    seen_symbols: set[str] = set()

    for line_number, raw_line in enumerate(
        source_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# ") and not line.startswith("## "):
            continue
        if line.startswith("## "):
            category = line[3:].strip()
            if not category:
                raise ValueError(
                    f"Empty category heading in {source_path} at line {line_number}"
                )
            continue
        if line.startswith("- "):
            if category is None:
                raise ValueError(
                    f"Asset entry before first category in {source_path} at line {line_number}"
                )
            symbol = line[2:].strip()
            if not symbol:
                raise ValueError(
                    f"Empty asset symbol in {source_path} at line {line_number}"
                )
            if symbol in seen_symbols:
                raise ValueError(
                    f"Duplicate asset symbol {symbol!r} in {source_path} at line {line_number}"
                )
            seen_symbols.add(symbol)
            records.append(
                AssetUniverseRecord(asset_id=symbol, symbol=symbol, category=category)
            )
            continue
        raise ValueError(f"Unsupported line in {source_path} at line {line_number}: {raw_line!r}")

    if not records:
        raise ValueError(f"No asset records found in {source_path}")
    return records


def _serialize_records(records: Iterable[AssetUniverseRecord]) -> list[dict[str, str]]:
    return [asdict(record) for record in records]


def write_manifests(
    records: Sequence[AssetUniverseRecord],
    json_path: Path | str,
    csv_path: Path | str,
) -> None:
    serialized = _serialize_records(records)
    json_output_path = Path(json_path)
    csv_output_path = Path(csv_path)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    json_output_path.write_text(
        json.dumps(serialized, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    with csv_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_id", "symbol", "category"])
        writer.writeheader()
        writer.writerows(serialized)


def build_ftmo_asset_universe(
    input_path: Path | str = default_input_path(),
    json_output_path: Path | str = default_json_output_path(),
    csv_output_path: Path | str = default_csv_output_path(),
) -> list[AssetUniverseRecord]:
    records = parse_ftmo_asset_document(input_path)
    write_manifests(records, json_output_path, csv_output_path)
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the FTMO asset-universe JSON and CSV manifests."
    )
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--json-output", type=Path, default=default_json_output_path())
    parser.add_argument("--csv-output", type=Path, default=default_csv_output_path())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = build_ftmo_asset_universe(
        input_path=args.input,
        json_output_path=args.json_output,
        csv_output_path=args.csv_output,
    )
    print(
        f"Wrote {len(records)} asset records to {args.json_output} and {args.csv_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

