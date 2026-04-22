from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.features.returns import (
    RETURNS_VOLATILITY_HEADER,
    CanonicalDailyCloseRecord,
    compute_arithmetic_returns,
    load_canonical_daily_close_csv,
    serialize_optional_float,
)


EWM_SPAN = 60
EWM_ADJUST = False
EWM_MIN_PERIODS = 60
EWM_BIAS = False


@dataclass(frozen=True)
class CanonicalDailyCloseManifestRecord:
    asset_id: str
    symbol: str
    category: str
    path_pattern: str
    source_d_file_path: str
    canonical_csv_path: str
    row_count: int
    first_timestamp: str
    last_timestamp: str
    file_hash: str


def default_canonical_manifest_input() -> Path:
    return default_project_root() / "artifacts/manifests/canonical_daily_close_manifest.json"


def default_output_dir() -> Path:
    return default_project_root() / "artifacts/features/base"


def load_canonical_daily_close_manifest(
    path: Path | str,
) -> list[CanonicalDailyCloseManifestRecord]:
    manifest_path = Path(path)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Canonical daily-close manifest must be a JSON array")

    required_fields = {
        "asset_id",
        "symbol",
        "category",
        "path_pattern",
        "source_d_file_path",
        "canonical_csv_path",
        "row_count",
        "first_timestamp",
        "last_timestamp",
        "file_hash",
    }
    records: list[CanonicalDailyCloseManifestRecord] = []
    seen_asset_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Canonical manifest row {index} is not an object")
        missing_fields = required_fields - set(row)
        if missing_fields:
            raise ValueError(
                f"Canonical manifest row {index} missing fields: {sorted(missing_fields)}"
            )
        asset_id = str(row["asset_id"])
        if asset_id in seen_asset_ids:
            raise ValueError(f"Canonical manifest has duplicate asset_id: {asset_id}")
        seen_asset_ids.add(asset_id)
        records.append(
            CanonicalDailyCloseManifestRecord(
                asset_id=asset_id,
                symbol=str(row["symbol"]),
                category=str(row["category"]),
                path_pattern=str(row["path_pattern"]),
                source_d_file_path=str(row["source_d_file_path"]),
                canonical_csv_path=str(row["canonical_csv_path"]),
                row_count=int(row["row_count"]),
                first_timestamp=str(row["first_timestamp"]),
                last_timestamp=str(row["last_timestamp"]),
                file_hash=str(row["file_hash"]),
            )
        )
    return records


def compute_ewm_volatility(
    arithmetic_returns: Sequence[float | None],
    *,
    span: int = EWM_SPAN,
    adjust: bool = EWM_ADJUST,
    min_periods: int = EWM_MIN_PERIODS,
    bias: bool = EWM_BIAS,
) -> list[float | None]:
    series = pd.Series(list(arithmetic_returns), dtype="float64")
    sigma_series = series.ewm(
        span=span,
        adjust=adjust,
        min_periods=min_periods,
    ).std(bias=bias)
    return [
        None if pd.isna(value) else float(value)
        for value in sigma_series.tolist()
    ]


def build_returns_volatility_rows(
    canonical_rows: Sequence[CanonicalDailyCloseRecord],
    *,
    span: int = EWM_SPAN,
    adjust: bool = EWM_ADJUST,
    min_periods: int = EWM_MIN_PERIODS,
    bias: bool = EWM_BIAS,
) -> list[dict[str, str]]:
    arithmetic_returns = compute_arithmetic_returns(canonical_rows)
    sigma_values = compute_ewm_volatility(
        arithmetic_returns,
        span=span,
        adjust=adjust,
        min_periods=min_periods,
        bias=bias,
    )
    return [
        {
            "timestamp": row.timestamp,
            "asset_id": row.asset_id,
            "close": row.close_text,
            "arithmetic_return": serialize_optional_float(arithmetic_return),
            "sigma_t": serialize_optional_float(sigma_t),
        }
        for row, arithmetic_return, sigma_t in zip(
            canonical_rows, arithmetic_returns, sigma_values
        )
    ]


def write_returns_volatility_csv(
    rows: Sequence[dict[str, str]],
    output_path: Path | str,
) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(RETURNS_VOLATILITY_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_canonical_alignment(
    manifest_record: CanonicalDailyCloseManifestRecord,
    canonical_rows: Sequence[CanonicalDailyCloseRecord],
) -> None:
    if len(canonical_rows) != manifest_record.row_count:
        raise ValueError(
            f"Canonical row count mismatch for {manifest_record.asset_id}: "
            f"{len(canonical_rows)} != {manifest_record.row_count}"
        )
    if not canonical_rows:
        raise ValueError(f"Canonical series is empty for {manifest_record.asset_id}")
    if canonical_rows[0].timestamp != manifest_record.first_timestamp:
        raise ValueError(f"Canonical first timestamp mismatch for {manifest_record.asset_id}")
    if canonical_rows[-1].timestamp != manifest_record.last_timestamp:
        raise ValueError(f"Canonical last timestamp mismatch for {manifest_record.asset_id}")
    if any(row.asset_id != manifest_record.asset_id for row in canonical_rows):
        raise ValueError(f"Canonical asset_id mismatch for {manifest_record.asset_id}")


def build_t09_outputs(
    canonical_manifest_input: Path | str = default_canonical_manifest_input(),
    output_dir: Path | str = default_output_dir(),
    project_root: Path | str | None = None,
) -> list[Path]:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    output_dir_path = Path(output_dir)
    manifest_records = load_canonical_daily_close_manifest(canonical_manifest_input)

    output_paths: list[Path] = []
    for manifest_record in manifest_records:
        canonical_csv_path = project_root_path / manifest_record.canonical_csv_path
        canonical_rows = load_canonical_daily_close_csv(
            canonical_csv_path, expected_asset_id=manifest_record.asset_id
        )
        validate_canonical_alignment(manifest_record, canonical_rows)
        rows = build_returns_volatility_rows(canonical_rows)
        output_path = output_dir_path / f"{manifest_record.asset_id}_returns_volatility.csv"
        write_returns_volatility_csv(rows, output_path)
        output_paths.append(output_path)

    for manifest_record in manifest_records:
        expected_output_path = output_dir_path / f"{manifest_record.asset_id}_returns_volatility.csv"
        if not expected_output_path.exists():
            raise ValueError(
                f"Missing returns-volatility output for {manifest_record.asset_id}"
            )
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build arithmetic returns and 60-day EWM volatility feature files."
    )
    parser.add_argument(
        "--canonical-manifest-input",
        type=Path,
        default=default_canonical_manifest_input(),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_paths = build_t09_outputs(
        canonical_manifest_input=args.canonical_manifest_input,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(f"Wrote {len(output_paths)} returns-volatility files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
