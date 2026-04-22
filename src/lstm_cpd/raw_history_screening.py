from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lstm_cpd.daily_close_contract import (
    PathResolutionRecord,
    RESOLUTION_REASON_MISSING_D_PATH,
    SCHEMA_REASON_DUPLICATE_CONFLICT,
    default_project_root,
    default_repo_root,
    inspect_daily_close_file,
)


MINIMUM_RAW_HISTORY_OBSERVATIONS = 318

REASON_MISSING_FILE = "MISSING_FILE"
REASON_UNREADABLE_FILE = "UNREADABLE_FILE"
REASON_SCHEMA_FAILURE = "SCHEMA_FAILURE"
REASON_EMPTY_SERIES = "EMPTY_SERIES"
REASON_DUPLICATE_TIMESTAMP_CONFLICT = "DUPLICATE_TIMESTAMP_CONFLICT"
REASON_INSUFFICIENT_RAW_HISTORY = "INSUFFICIENT_RAW_HISTORY"


@dataclass(frozen=True)
class RawHistoryScreeningRecord:
    asset_id: str
    symbol: str
    category: str
    path_pattern: str
    d_file_path: str
    resolution_status: str
    resolution_failure_reason: str
    schema_status: str
    schema_reason_code: str
    raw_row_count: int
    screened_row_count: int
    first_timestamp: str
    last_timestamp: str
    raw_eligibility_status: str
    reason_code: str


def default_path_manifest_input() -> Path:
    return default_project_root() / "artifacts/manifests/d_timeframe_path_manifest.json"


def default_contract_input() -> Path:
    return default_project_root() / "docs/contracts/daily_close_schema_contract.md"


def default_eligibility_report_output() -> Path:
    return default_project_root() / "artifacts/reports/asset_eligibility_report.csv"


def default_exclusion_report_output() -> Path:
    return default_project_root() / "artifacts/reports/asset_exclusion_report.csv"


def default_screening_report_output() -> Path:
    return (
        default_project_root()
        / "artifacts/reports/minimum_history_screening_report.csv"
    )


def load_path_manifest(path: Path | str) -> list[PathResolutionRecord]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Path manifest must be a JSON array")
    required_fields = {
        "asset_id",
        "symbol",
        "category",
        "resolution_status",
        "resolution_failure_reason",
        "path_pattern",
        "d_file_path",
        "candidate_paths",
    }
    records: list[PathResolutionRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Path manifest row {index} is not an object")
        missing = required_fields - set(row)
        if missing:
            raise ValueError(f"Path manifest row {index} missing fields: {sorted(missing)}")
        candidate_paths = row["candidate_paths"]
        if not isinstance(candidate_paths, list) or not all(
            isinstance(value, str) for value in candidate_paths
        ):
            raise ValueError(
                f"Path manifest row {index} has invalid candidate_paths payload"
            )
        records.append(
            PathResolutionRecord(
                asset_id=str(row["asset_id"]),
                symbol=str(row["symbol"]),
                category=str(row["category"]),
                resolution_status=str(row["resolution_status"]),
                resolution_failure_reason=(
                    None
                    if row["resolution_failure_reason"] in (None, "")
                    else str(row["resolution_failure_reason"])
                ),
                path_pattern=None if row["path_pattern"] in (None, "") else str(row["path_pattern"]),
                d_file_path=None if row["d_file_path"] in (None, "") else str(row["d_file_path"]),
                candidate_paths=list(candidate_paths),
            )
        )
    return records


def require_contract(path: Path | str) -> Path:
    contract_path = Path(path)
    if not contract_path.exists():
        raise FileNotFoundError(f"Missing daily-close schema contract: {contract_path}")
    return contract_path


def unresolved_reason_code(resolution: PathResolutionRecord) -> str:
    if resolution.resolution_failure_reason == RESOLUTION_REASON_MISSING_D_PATH:
        return REASON_MISSING_FILE
    return REASON_SCHEMA_FAILURE


def unresolved_screening_record(resolution: PathResolutionRecord) -> RawHistoryScreeningRecord:
    return RawHistoryScreeningRecord(
        asset_id=resolution.asset_id,
        symbol=resolution.symbol,
        category=resolution.category,
        path_pattern=resolution.path_pattern or "",
        d_file_path=resolution.d_file_path or "",
        resolution_status=resolution.resolution_status,
        resolution_failure_reason=resolution.resolution_failure_reason or "",
        schema_status="NOT_INSPECTED",
        schema_reason_code="",
        raw_row_count=0,
        screened_row_count=0,
        first_timestamp="",
        last_timestamp="",
        raw_eligibility_status="EXCLUDED",
        reason_code=unresolved_reason_code(resolution),
    )


def inspection_to_screening_record(
    resolution: PathResolutionRecord,
    inspection,
) -> RawHistoryScreeningRecord:
    reason_code = ""
    eligibility = "ELIGIBLE"

    if inspection.schema_status == "EXCLUDED":
        eligibility = "EXCLUDED"
        if inspection.schema_reason_code == SCHEMA_REASON_DUPLICATE_CONFLICT:
            reason_code = REASON_DUPLICATE_TIMESTAMP_CONFLICT
        else:
            reason_code = REASON_SCHEMA_FAILURE
    elif inspection.canonical_row_count == 0:
        eligibility = "EXCLUDED"
        reason_code = REASON_EMPTY_SERIES
    elif inspection.canonical_row_count < MINIMUM_RAW_HISTORY_OBSERVATIONS:
        eligibility = "EXCLUDED"
        reason_code = REASON_INSUFFICIENT_RAW_HISTORY

    return RawHistoryScreeningRecord(
        asset_id=resolution.asset_id,
        symbol=resolution.symbol,
        category=resolution.category,
        path_pattern=resolution.path_pattern or "",
        d_file_path=resolution.d_file_path or "",
        resolution_status=resolution.resolution_status,
        resolution_failure_reason=resolution.resolution_failure_reason or "",
        schema_status=inspection.schema_status,
        schema_reason_code=inspection.schema_reason_code,
        raw_row_count=inspection.raw_row_count,
        screened_row_count=inspection.canonical_row_count,
        first_timestamp=inspection.canonical_first_timestamp,
        last_timestamp=inspection.canonical_last_timestamp,
        raw_eligibility_status=eligibility,
        reason_code=reason_code,
    )


def unreadable_screening_record(
    resolution: PathResolutionRecord,
    reason_code: str,
) -> RawHistoryScreeningRecord:
    return RawHistoryScreeningRecord(
        asset_id=resolution.asset_id,
        symbol=resolution.symbol,
        category=resolution.category,
        path_pattern=resolution.path_pattern or "",
        d_file_path=resolution.d_file_path or "",
        resolution_status=resolution.resolution_status,
        resolution_failure_reason=resolution.resolution_failure_reason or "",
        schema_status="NOT_INSPECTED",
        schema_reason_code="",
        raw_row_count=0,
        screened_row_count=0,
        first_timestamp="",
        last_timestamp="",
        raw_eligibility_status="EXCLUDED",
        reason_code=reason_code,
    )


def screen_path_resolution_record(
    resolution: PathResolutionRecord,
    repo_root: Path | str,
) -> RawHistoryScreeningRecord:
    repo_root_path = Path(repo_root)
    if resolution.resolution_status != "RESOLVED" or resolution.d_file_path is None:
        return unresolved_screening_record(resolution)

    raw_path = repo_root_path / resolution.d_file_path
    if not raw_path.exists():
        return unreadable_screening_record(resolution, REASON_MISSING_FILE)

    asset_row = {
        "asset_id": resolution.asset_id,
        "symbol": resolution.symbol,
        "category": resolution.category,
    }
    try:
        inspection = inspect_daily_close_file(asset_row, resolution, repo_root_path)
    except FileNotFoundError:
        return unreadable_screening_record(resolution, REASON_MISSING_FILE)
    except (OSError, UnicodeDecodeError, csv.Error):
        return unreadable_screening_record(resolution, REASON_UNREADABLE_FILE)

    return inspection_to_screening_record(resolution, inspection)


def screening_record_to_row(record: RawHistoryScreeningRecord) -> dict[str, object]:
    return {
        "asset_id": record.asset_id,
        "symbol": record.symbol,
        "category": record.category,
        "path_pattern": record.path_pattern,
        "d_file_path": record.d_file_path,
        "resolution_status": record.resolution_status,
        "resolution_failure_reason": record.resolution_failure_reason,
        "schema_status": record.schema_status,
        "schema_reason_code": record.schema_reason_code,
        "raw_row_count": record.raw_row_count,
        "screened_row_count": record.screened_row_count,
        "first_timestamp": record.first_timestamp,
        "last_timestamp": record.last_timestamp,
        "raw_eligibility_status": record.raw_eligibility_status,
        "reason_code": record.reason_code,
    }


def eligibility_row(record: RawHistoryScreeningRecord) -> dict[str, object]:
    return {
        "asset_id": record.asset_id,
        "symbol": record.symbol,
        "category": record.category,
        "path_pattern": record.path_pattern,
        "d_file_path": record.d_file_path,
        "raw_row_count": record.raw_row_count,
        "screened_row_count": record.screened_row_count,
        "first_timestamp": record.first_timestamp,
        "last_timestamp": record.last_timestamp,
    }


def exclusion_row(record: RawHistoryScreeningRecord) -> dict[str, object]:
    return {
        "asset_id": record.asset_id,
        "symbol": record.symbol,
        "category": record.category,
        "path_pattern": record.path_pattern,
        "d_file_path": record.d_file_path,
        "raw_row_count": record.raw_row_count,
        "screened_row_count": record.screened_row_count,
        "reason_code": record.reason_code,
        "first_timestamp": record.first_timestamp,
        "last_timestamp": record.last_timestamp,
    }


def write_csv_rows(
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
    output_path: Path | str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_t07_outputs(
    path_manifest_input: Path | str = default_path_manifest_input(),
    contract_input: Path | str = default_contract_input(),
    eligibility_report_output: Path | str = default_eligibility_report_output(),
    exclusion_report_output: Path | str = default_exclusion_report_output(),
    screening_report_output: Path | str = default_screening_report_output(),
    repo_root: Path | str | None = None,
) -> list[RawHistoryScreeningRecord]:
    require_contract(contract_input)
    repo_root_path = Path(repo_root) if repo_root is not None else default_repo_root()
    path_records = load_path_manifest(path_manifest_input)
    screening_records = [
        screen_path_resolution_record(record, repo_root_path) for record in path_records
    ]

    all_rows = [screening_record_to_row(record) for record in screening_records]
    eligibility_rows = [
        eligibility_row(record)
        for record in screening_records
        if record.raw_eligibility_status == "ELIGIBLE"
    ]
    exclusion_rows = [
        exclusion_row(record)
        for record in screening_records
        if record.raw_eligibility_status == "EXCLUDED"
    ]

    write_csv_rows(
        all_rows,
        [
            "asset_id",
            "symbol",
            "category",
            "path_pattern",
            "d_file_path",
            "resolution_status",
            "resolution_failure_reason",
            "schema_status",
            "schema_reason_code",
            "raw_row_count",
            "screened_row_count",
            "first_timestamp",
            "last_timestamp",
            "raw_eligibility_status",
            "reason_code",
        ],
        screening_report_output,
    )
    write_csv_rows(
        eligibility_rows,
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
        eligibility_report_output,
    )
    write_csv_rows(
        exclusion_rows,
        [
            "asset_id",
            "symbol",
            "category",
            "path_pattern",
            "d_file_path",
            "raw_row_count",
            "screened_row_count",
            "reason_code",
            "first_timestamp",
            "last_timestamp",
        ],
        exclusion_report_output,
    )
    return screening_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen raw D-file availability and minimum-history sufficiency."
    )
    parser.add_argument("--path-manifest-input", type=Path, default=default_path_manifest_input())
    parser.add_argument("--contract-input", type=Path, default=default_contract_input())
    parser.add_argument(
        "--eligibility-report-output",
        type=Path,
        default=default_eligibility_report_output(),
    )
    parser.add_argument(
        "--exclusion-report-output",
        type=Path,
        default=default_exclusion_report_output(),
    )
    parser.add_argument(
        "--screening-report-output",
        type=Path,
        default=default_screening_report_output(),
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = build_t07_outputs(
        path_manifest_input=args.path_manifest_input,
        contract_input=args.contract_input,
        eligibility_report_output=args.eligibility_report_output,
        exclusion_report_output=args.exclusion_report_output,
        screening_report_output=args.screening_report_output,
        repo_root=args.repo_root,
    )
    eligible = sum(record.raw_eligibility_status == "ELIGIBLE" for record in records)
    excluded = len(records) - eligible
    print(
        f"Wrote {len(records)} screening rows ({eligible} eligible, {excluded} excluded)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
