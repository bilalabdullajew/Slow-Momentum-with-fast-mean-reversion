from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date", "time")
PATH_PATTERN_STANDARD = "category_symbol_d"
PATH_PATTERN_FOREX = "forex_base_symbol_d"

RESOLUTION_REASON_MISSING_D_PATH = "MISSING_D_PATH"
RESOLUTION_REASON_AMBIGUOUS_D_PATH = "AMBIGUOUS_D_PATH"

SCHEMA_REASON_MISSING_TIMESTAMP = "MISSING_TIMESTAMP_COLUMN"
SCHEMA_REASON_MULTIPLE_TIMESTAMPS = "MULTIPLE_TIMESTAMP_COLUMNS"
SCHEMA_REASON_MISSING_CLOSE = "MISSING_CLOSE_COLUMN"
SCHEMA_REASON_MULTIPLE_CLOSE = "MULTIPLE_CLOSE_COLUMNS"
SCHEMA_REASON_UNPARSEABLE_TIMESTAMP = "UNPARSEABLE_TIMESTAMP"
SCHEMA_REASON_NON_NUMERIC_CLOSE = "NON_NUMERIC_CLOSE"
SCHEMA_REASON_DUPLICATE_CONFLICT = "DUPLICATE_TIMESTAMP_CONFLICT"


@dataclass(frozen=True)
class PathResolutionRecord:
    asset_id: str
    symbol: str
    category: str
    resolution_status: str
    resolution_failure_reason: str | None
    path_pattern: str | None
    d_file_path: str | None
    candidate_paths: list[str]


@dataclass(frozen=True)
class CanonicalDailyCloseRow:
    timestamp: str
    close: float
    parsed_timestamp: datetime


@dataclass(frozen=True)
class SchemaInspectionRecord:
    asset_id: str
    symbol: str
    category: str
    path_pattern: str | None
    resolution_status: str
    resolution_failure_reason: str | None
    d_file_path: str | None
    raw_header: str
    timestamp_column: str
    close_column: str
    schema_status: str
    schema_reason_code: str
    raw_row_count: int
    canonical_row_count: int
    duplicate_identical_count: int
    duplicate_conflict_count: int
    source_sorted_ascending: str
    canonical_first_timestamp: str
    canonical_last_timestamp: str


@dataclass(frozen=True)
class CanonicalDailyCloseExtraction:
    inspection: SchemaInspectionRecord
    canonical_rows: tuple[CanonicalDailyCloseRow, ...]


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_repo_root() -> Path:
    return default_project_root().parents[2]


def default_asset_manifest_path() -> Path:
    return default_project_root() / "artifacts/manifests/ftmo_asset_universe.json"


def default_ftmo_root() -> Path:
    return default_repo_root() / "data/FTMO Data"


def default_path_manifest_output() -> Path:
    return default_project_root() / "artifacts/manifests/d_timeframe_path_manifest.json"


def default_contract_output() -> Path:
    return default_project_root() / "docs/contracts/daily_close_schema_contract.md"


def default_schema_report_output() -> Path:
    return default_project_root() / "artifacts/reports/schema_inspection_report.csv"


def repo_relative_path(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def load_asset_manifest(path: Path | str) -> list[dict[str, str]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Asset manifest must be a JSON array")
    required_fields = {"asset_id", "symbol", "category"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Asset manifest row {index} is not an object")
        missing = required_fields - set(row)
        if missing:
            raise ValueError(f"Asset manifest row {index} missing fields: {sorted(missing)}")
    return rows


def candidate_d_paths(asset_row: dict[str, str], ftmo_root: Path) -> list[tuple[str, Path]]:
    symbol = asset_row["symbol"]
    category = asset_row["category"]
    candidates: list[tuple[str, Path]] = [
        (
            PATH_PATTERN_STANDARD,
            ftmo_root / category / symbol / "D" / f"{symbol}_data.csv",
        )
    ]
    if category == "Forex":
        candidates.append(
            (
                PATH_PATTERN_FOREX,
                ftmo_root / "Forex" / symbol[:3] / symbol / "D" / f"{symbol}_data.csv",
            )
        )
    return candidates


def resolve_d_path(
    asset_row: dict[str, str],
    ftmo_root: Path | str,
    repo_root: Path | str,
) -> PathResolutionRecord:
    ftmo_root_path = Path(ftmo_root)
    repo_root_path = Path(repo_root)
    candidates = candidate_d_paths(asset_row, ftmo_root_path)
    existing = [(pattern, path) for pattern, path in candidates if path.exists()]
    candidate_paths = [repo_relative_path(path, repo_root_path) for _, path in candidates]

    if len(existing) == 1:
        pattern, path = existing[0]
        return PathResolutionRecord(
            asset_id=asset_row["asset_id"],
            symbol=asset_row["symbol"],
            category=asset_row["category"],
            resolution_status="RESOLVED",
            resolution_failure_reason=None,
            path_pattern=pattern,
            d_file_path=repo_relative_path(path, repo_root_path),
            candidate_paths=candidate_paths,
        )

    failure_reason = (
        RESOLUTION_REASON_MISSING_D_PATH
        if not existing
        else RESOLUTION_REASON_AMBIGUOUS_D_PATH
    )
    return PathResolutionRecord(
        asset_id=asset_row["asset_id"],
        symbol=asset_row["symbol"],
        category=asset_row["category"],
        resolution_status="FAILED",
        resolution_failure_reason=failure_reason,
        path_pattern=None,
        d_file_path=None,
        candidate_paths=candidate_paths,
    )


def find_exactly_one_column(fieldnames: Sequence[str], accepted: set[str]) -> tuple[str, str | None]:
    matches = [name for name in fieldnames if name.lower() in accepted]
    if accepted == {"close"}:
        if not matches:
            return SCHEMA_REASON_MISSING_CLOSE, None
        if len(matches) > 1:
            return SCHEMA_REASON_MULTIPLE_CLOSE, None
        return "", matches[0]
    if not matches:
        return SCHEMA_REASON_MISSING_TIMESTAMP, None
    if len(matches) > 1:
        return SCHEMA_REASON_MULTIPLE_TIMESTAMPS, None
    return "", matches[0]


def parse_timestamp_value(raw_value: str) -> datetime:
    value = raw_value.strip()
    if not value:
        raise ValueError("Empty timestamp")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def parse_close_value(raw_value: str) -> float:
    value = raw_value.strip()
    if not value:
        raise ValueError("Empty close")
    return float(value)


def bool_to_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def inspect_daily_close_file(
    asset_row: dict[str, str],
    resolution: PathResolutionRecord,
    repo_root: Path | str,
) -> SchemaInspectionRecord:
    return extract_canonical_daily_close(asset_row, resolution, repo_root).inspection


def extract_canonical_daily_close(
    asset_row: dict[str, str],
    resolution: PathResolutionRecord,
    repo_root: Path | str,
) -> CanonicalDailyCloseExtraction:
    repo_root_path = Path(repo_root)
    if resolution.d_file_path is None:
        return CanonicalDailyCloseExtraction(
            inspection=SchemaInspectionRecord(
                asset_id=asset_row["asset_id"],
                symbol=asset_row["symbol"],
                category=asset_row["category"],
                path_pattern=resolution.path_pattern or "",
                resolution_status=resolution.resolution_status,
                resolution_failure_reason=resolution.resolution_failure_reason or "",
                d_file_path="",
                raw_header="",
                timestamp_column="",
                close_column="",
                schema_status="NOT_INSPECTED",
                schema_reason_code="",
                raw_row_count=0,
                canonical_row_count=0,
                duplicate_identical_count=0,
                duplicate_conflict_count=0,
                source_sorted_ascending="",
                canonical_first_timestamp="",
                canonical_last_timestamp="",
            ),
            canonical_rows=(),
        )

    path = repo_root_path / resolution.d_file_path
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        raw_header = json.dumps(fieldnames, ensure_ascii=True)

        timestamp_error, timestamp_column = find_exactly_one_column(
            fieldnames, set(TIMESTAMP_CANDIDATES)
        )
        if timestamp_error:
            return CanonicalDailyCloseExtraction(
                inspection=schema_failure_record(
                    asset_row,
                    resolution,
                    raw_header,
                    schema_reason_code=timestamp_error,
                ),
                canonical_rows=(),
            )

        close_error, close_column = find_exactly_one_column(fieldnames, {"close"})
        if close_error:
            return CanonicalDailyCloseExtraction(
                inspection=schema_failure_record(
                    asset_row,
                    resolution,
                    raw_header,
                    timestamp_column=timestamp_column or "",
                    schema_reason_code=close_error,
                ),
                canonical_rows=(),
            )

        parsed_rows: list[CanonicalDailyCloseRow] = []
        raw_row_count = 0
        source_sorted_ascending = True
        previous_timestamp: datetime | None = None
        duplicate_identical_count = 0
        duplicate_conflict_count = 0
        deduped_by_timestamp: dict[str, CanonicalDailyCloseRow] = {}

        for row in reader:
            raw_row_count += 1
            try:
                parsed_timestamp = parse_timestamp_value(row[timestamp_column])  # type: ignore[index]
            except Exception:
                return CanonicalDailyCloseExtraction(
                    inspection=schema_failure_record(
                        asset_row,
                        resolution,
                        raw_header,
                        timestamp_column=timestamp_column or "",
                        close_column=close_column or "",
                        raw_row_count=raw_row_count,
                        schema_reason_code=SCHEMA_REASON_UNPARSEABLE_TIMESTAMP,
                    ),
                    canonical_rows=(),
                )
            try:
                close_value = parse_close_value(row[close_column])  # type: ignore[index]
            except Exception:
                return CanonicalDailyCloseExtraction(
                    inspection=schema_failure_record(
                        asset_row,
                        resolution,
                        raw_header,
                        timestamp_column=timestamp_column or "",
                        close_column=close_column or "",
                        raw_row_count=raw_row_count,
                        schema_reason_code=SCHEMA_REASON_NON_NUMERIC_CLOSE,
                    ),
                    canonical_rows=(),
                )

            if previous_timestamp is not None and parsed_timestamp < previous_timestamp:
                source_sorted_ascending = False
            previous_timestamp = parsed_timestamp

            canonical_row = CanonicalDailyCloseRow(
                timestamp=(row[timestamp_column] or "").strip(),  # type: ignore[index]
                close=close_value,
                parsed_timestamp=parsed_timestamp,
            )
            key = parsed_timestamp.isoformat()
            prior = deduped_by_timestamp.get(key)
            if prior is None:
                deduped_by_timestamp[key] = canonical_row
                parsed_rows.append(canonical_row)
                continue
            if prior.close == canonical_row.close:
                duplicate_identical_count += 1
                deduped_by_timestamp[key] = canonical_row
                continue
            duplicate_conflict_count += 1
            return CanonicalDailyCloseExtraction(
                inspection=schema_failure_record(
                    asset_row,
                    resolution,
                    raw_header,
                    timestamp_column=timestamp_column or "",
                    close_column=close_column or "",
                    raw_row_count=raw_row_count,
                    duplicate_identical_count=duplicate_identical_count,
                    duplicate_conflict_count=duplicate_conflict_count,
                    source_sorted_ascending=source_sorted_ascending,
                    schema_reason_code=SCHEMA_REASON_DUPLICATE_CONFLICT,
                ),
                canonical_rows=(),
            )

    canonical_rows = sorted(deduped_by_timestamp.values(), key=lambda row: row.parsed_timestamp)
    return CanonicalDailyCloseExtraction(
        inspection=SchemaInspectionRecord(
            asset_id=asset_row["asset_id"],
            symbol=asset_row["symbol"],
            category=asset_row["category"],
            path_pattern=resolution.path_pattern or "",
            resolution_status=resolution.resolution_status,
            resolution_failure_reason=resolution.resolution_failure_reason or "",
            d_file_path=resolution.d_file_path or "",
            raw_header=raw_header,
            timestamp_column=timestamp_column or "",
            close_column=close_column or "",
            schema_status="ADMISSIBLE",
            schema_reason_code="",
            raw_row_count=raw_row_count,
            canonical_row_count=len(canonical_rows),
            duplicate_identical_count=duplicate_identical_count,
            duplicate_conflict_count=duplicate_conflict_count,
            source_sorted_ascending=bool_to_text(source_sorted_ascending),
            canonical_first_timestamp=canonical_rows[0].timestamp if canonical_rows else "",
            canonical_last_timestamp=canonical_rows[-1].timestamp if canonical_rows else "",
        ),
        canonical_rows=tuple(canonical_rows),
    )


def schema_failure_record(
    asset_row: dict[str, str],
    resolution: PathResolutionRecord,
    raw_header: str,
    schema_reason_code: str,
    timestamp_column: str = "",
    close_column: str = "",
    raw_row_count: int = 0,
    duplicate_identical_count: int = 0,
    duplicate_conflict_count: int = 0,
    source_sorted_ascending: bool | None = None,
) -> SchemaInspectionRecord:
    return SchemaInspectionRecord(
        asset_id=asset_row["asset_id"],
        symbol=asset_row["symbol"],
        category=asset_row["category"],
        path_pattern=resolution.path_pattern or "",
        resolution_status=resolution.resolution_status,
        resolution_failure_reason=resolution.resolution_failure_reason or "",
        d_file_path=resolution.d_file_path or "",
        raw_header=raw_header,
        timestamp_column=timestamp_column,
        close_column=close_column,
        schema_status="EXCLUDED",
        schema_reason_code=schema_reason_code,
        raw_row_count=raw_row_count,
        canonical_row_count=0,
        duplicate_identical_count=duplicate_identical_count,
        duplicate_conflict_count=duplicate_conflict_count,
        source_sorted_ascending=bool_to_text(source_sorted_ascending),
        canonical_first_timestamp="",
        canonical_last_timestamp="",
    )


def build_daily_close_schema_contract() -> str:
    return """# Daily-Close Schema Contract

## Purpose

This document is the binding schema contract produced by `T-06`.

It defines how FTMO raw `D` timeframe files become admissible daily-close sources for later tasks.

## Authority

- Live Asana task authority: `T-06 — Resolve D-timeframe paths and daily-close schema contract`
- Methodological constraints: `docs/contracts/invariant_ledger.md` and `docs/contracts/exclusions_ledger.md`

## D-path resolution rules

The path manifest resolves exactly one repo-relative `D` timeframe file per allowed asset using these path families:

1. Standard path family:
   `data/FTMO Data/{category}/{symbol}/D/{symbol}_data.csv`
2. Forex path family:
   `data/FTMO Data/Forex/{base_ccy}/{symbol}/D/{symbol}_data.csv`
   where `base_ccy = symbol[:3]`

Resolution outcomes:

- If exactly one candidate exists, the asset is `RESOLVED`.
- If no candidate exists, the asset is `FAILED` with reason code `MISSING_D_PATH`.
- If more than one candidate exists, the asset is `FAILED` with reason code `AMBIGUOUS_D_PATH`.

## Admissible source columns

For a resolved `D` file:

- Accept exactly one source timestamp column chosen case-insensitively from:
  - `timestamp`
  - `datetime`
  - `date`
  - `time`
- Map that source column to canonical column `timestamp`.
- Accept exactly one source close column chosen case-insensitively from:
  - `close`
- Map that source column to canonical column `close`.

All other source columns are ignored for canonical daily-close admission, including `open`, `high`, `low`, `volume`, `spread`, and every non-`D` timeframe field.

## Validation and ordering rules

- Every timestamp value must be parseable as a timestamp.
- Every close value must be parseable as numeric.
- Canonical rows are ordered ascending by parsed timestamp.
- The contract refers only to `D` timeframe and `close` for canonical admission.

## Duplicate-timestamp rules

- If duplicate timestamps have identical close values, keep the last occurrence and log the duplicate count.
- If duplicate timestamps disagree on close, exclude the asset with reason code `DUPLICATE_TIMESTAMP_CONFLICT`.

## Exclusion conditions

Resolved files are excluded from canonical admission if any of the following holds:

- `MISSING_TIMESTAMP_COLUMN`
- `MULTIPLE_TIMESTAMP_COLUMNS`
- `MISSING_CLOSE_COLUMN`
- `MULTIPLE_CLOSE_COLUMNS`
- `UNPARSEABLE_TIMESTAMP`
- `NON_NUMERIC_CLOSE`
- `DUPLICATE_TIMESTAMP_CONFLICT`

## Deferred checks

This contract does not decide raw-availability outcomes such as empty series or insufficient history. Those checks are deferred to `T-07`.
"""


def write_json(records: Iterable[dict[str, object]], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(records), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_schema_report(
    records: Sequence[SchemaInspectionRecord],
    output_path: Path | str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "asset_id",
        "symbol",
        "category",
        "path_pattern",
        "resolution_status",
        "resolution_failure_reason",
        "d_file_path",
        "raw_header",
        "timestamp_column",
        "close_column",
        "schema_status",
        "schema_reason_code",
        "raw_row_count",
        "canonical_row_count",
        "duplicate_identical_count",
        "duplicate_conflict_count",
        "source_sorted_ascending",
        "canonical_first_timestamp",
        "canonical_last_timestamp",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def build_t06_outputs(
    asset_manifest_path: Path | str = default_asset_manifest_path(),
    ftmo_root: Path | str = default_ftmo_root(),
    path_manifest_output: Path | str = default_path_manifest_output(),
    contract_output: Path | str = default_contract_output(),
    schema_report_output: Path | str = default_schema_report_output(),
    repo_root: Path | str | None = None,
) -> tuple[list[PathResolutionRecord], list[SchemaInspectionRecord]]:
    repo_root_path = Path(repo_root) if repo_root is not None else default_repo_root()
    asset_rows = load_asset_manifest(asset_manifest_path)
    resolutions: list[PathResolutionRecord] = []
    inspections: list[SchemaInspectionRecord] = []
    for asset_row in asset_rows:
        resolution = resolve_d_path(asset_row, ftmo_root, repo_root_path)
        resolutions.append(resolution)
        inspections.append(inspect_daily_close_file(asset_row, resolution, repo_root_path))

    write_json([asdict(record) for record in resolutions], path_manifest_output)
    write_schema_report(inspections, schema_report_output)
    contract_path = Path(contract_output)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(build_daily_close_schema_contract(), encoding="utf-8")
    return resolutions, inspections


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve FTMO D-file paths and build the daily-close schema contract artifacts."
    )
    parser.add_argument("--asset-manifest", type=Path, default=default_asset_manifest_path())
    parser.add_argument("--ftmo-root", type=Path, default=default_ftmo_root())
    parser.add_argument(
        "--path-manifest-output", type=Path, default=default_path_manifest_output()
    )
    parser.add_argument("--contract-output", type=Path, default=default_contract_output())
    parser.add_argument(
        "--schema-report-output", type=Path, default=default_schema_report_output()
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolutions, inspections = build_t06_outputs(
        asset_manifest_path=args.asset_manifest,
        ftmo_root=args.ftmo_root,
        path_manifest_output=args.path_manifest_output,
        contract_output=args.contract_output,
        schema_report_output=args.schema_report_output,
        repo_root=args.repo_root,
    )
    resolved_count = sum(record.resolution_status == "RESOLVED" for record in resolutions)
    admissible_count = sum(record.schema_status == "ADMISSIBLE" for record in inspections)
    print(
        "Wrote "
        f"{len(resolutions)} path records ({resolved_count} resolved) and "
        f"{len(inspections)} schema rows ({admissible_count} admissible)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
