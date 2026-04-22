from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from lstm_cpd.daily_close_contract import (
    CanonicalDailyCloseRow,
    default_project_root,
    default_repo_root,
    extract_canonical_daily_close,
    parse_timestamp_value,
    write_json,
)
from lstm_cpd.raw_history_screening import (
    default_contract_input,
    default_path_manifest_input,
    load_path_manifest,
    require_contract,
)


CANONICAL_CSV_HEADER = ("timestamp", "asset_id", "close")


@dataclass(frozen=True)
class EligibleAssetRecord:
    asset_id: str
    symbol: str
    category: str
    path_pattern: str
    d_file_path: str
    raw_row_count: int
    screened_row_count: int
    first_timestamp: str
    last_timestamp: str


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


def default_eligibility_report_input() -> Path:
    return default_project_root() / "artifacts/reports/asset_eligibility_report.csv"


def default_canonical_output_dir() -> Path:
    return default_project_root() / "artifacts/canonical_daily_close"


def default_manifest_output() -> Path:
    return default_project_root() / "artifacts/manifests/canonical_daily_close_manifest.json"


def project_relative_path(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def load_eligibility_report(path: Path | str) -> list[EligibleAssetRecord]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    required_fields = {
        "asset_id",
        "symbol",
        "category",
        "path_pattern",
        "d_file_path",
        "raw_row_count",
        "screened_row_count",
        "first_timestamp",
        "last_timestamp",
    }
    if reader.fieldnames is None:
        raise ValueError(f"Eligibility report is missing a header: {report_path}")
    missing_fields = required_fields - set(reader.fieldnames)
    if missing_fields:
        raise ValueError(
            f"Eligibility report missing fields: {sorted(missing_fields)}"
        )

    seen_asset_ids: set[str] = set()
    records: list[EligibleAssetRecord] = []
    for index, row in enumerate(rows):
        asset_id = (row["asset_id"] or "").strip()
        if not asset_id:
            raise ValueError(f"Eligibility row {index} has empty asset_id")
        if asset_id in seen_asset_ids:
            raise ValueError(f"Eligibility report has duplicate asset_id: {asset_id}")
        seen_asset_ids.add(asset_id)
        records.append(
            EligibleAssetRecord(
                asset_id=asset_id,
                symbol=(row["symbol"] or "").strip(),
                category=(row["category"] or "").strip(),
                path_pattern=(row["path_pattern"] or "").strip(),
                d_file_path=(row["d_file_path"] or "").strip(),
                raw_row_count=int(row["raw_row_count"]),
                screened_row_count=int(row["screened_row_count"]),
                first_timestamp=(row["first_timestamp"] or "").strip(),
                last_timestamp=(row["last_timestamp"] or "").strip(),
            )
        )
    return records


def build_path_manifest_index(path_manifest_input: Path | str):
    index = {}
    for record in load_path_manifest(path_manifest_input):
        if record.asset_id in index:
            raise ValueError(f"Path manifest has duplicate asset_id: {record.asset_id}")
        index[record.asset_id] = record
    return index


def normalize_timestamp(row: CanonicalDailyCloseRow) -> str:
    return row.parsed_timestamp.isoformat()


def normalize_close(close: float) -> str:
    text = format(Decimal(str(close)).normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def serialize_canonical_daily_close_csv_bytes(
    asset_id: str,
    rows: Sequence[CanonicalDailyCloseRow],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CANONICAL_CSV_HEADER)
    for row in rows:
        writer.writerow(
            [
                normalize_timestamp(row),
                asset_id,
                normalize_close(row.close),
            ]
        )
    return buffer.getvalue().encode("utf-8")


def sha256_prefixed(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def validate_resolution_alignment(
    eligible: EligibleAssetRecord,
    resolution,
) -> None:
    if resolution is None:
        raise ValueError(f"Eligible asset missing from path manifest: {eligible.asset_id}")
    if resolution.resolution_status != "RESOLVED":
        raise ValueError(
            f"Eligible asset must resolve in path manifest: {eligible.asset_id}"
        )
    if resolution.symbol != eligible.symbol or resolution.category != eligible.category:
        raise ValueError(
            f"Eligibility/path-manifest mismatch for {eligible.asset_id}"
        )
    if (resolution.path_pattern or "") != eligible.path_pattern:
        raise ValueError(f"Path pattern mismatch for {eligible.asset_id}")
    if (resolution.d_file_path or "") != eligible.d_file_path:
        raise ValueError(f"D-file path mismatch for {eligible.asset_id}")


def validate_screening_alignment(
    eligible: EligibleAssetRecord,
    extraction_rows: Sequence[CanonicalDailyCloseRow],
) -> None:
    if len(extraction_rows) != eligible.screened_row_count:
        raise ValueError(
            f"Screened row count mismatch for {eligible.asset_id}: "
            f"{len(extraction_rows)} != {eligible.screened_row_count}"
        )
    if not extraction_rows:
        raise ValueError(f"Eligible asset produced an empty canonical series: {eligible.asset_id}")

    first_timestamp = normalize_timestamp(extraction_rows[0])
    last_timestamp = normalize_timestamp(extraction_rows[-1])
    expected_first_timestamp = parse_timestamp_value(eligible.first_timestamp).isoformat()
    expected_last_timestamp = parse_timestamp_value(eligible.last_timestamp).isoformat()
    if first_timestamp != expected_first_timestamp:
        raise ValueError(f"First timestamp mismatch for {eligible.asset_id}")
    if last_timestamp != expected_last_timestamp:
        raise ValueError(f"Last timestamp mismatch for {eligible.asset_id}")


def validate_canonical_output_dir(
    canonical_output_dir: Path,
    expected_asset_ids: set[str],
) -> None:
    actual_asset_ids = {
        path.stem for path in canonical_output_dir.glob("*.csv") if path.is_file()
    }
    if actual_asset_ids != expected_asset_ids:
        missing = sorted(expected_asset_ids - actual_asset_ids)
        unexpected = sorted(actual_asset_ids - expected_asset_ids)
        raise ValueError(
            "Canonical daily-close directory does not reconcile with the eligible asset set: "
            f"missing={missing}, unexpected={unexpected}"
        )


def validate_manifest_records(
    records: Sequence[CanonicalDailyCloseManifestRecord],
    canonical_output_dir: Path,
    manifest_output: Path,
    project_root: Path,
) -> None:
    validate_canonical_output_dir(canonical_output_dir, {record.asset_id for record in records})
    manifest_payload = json.loads(manifest_output.read_text(encoding="utf-8"))
    if len(manifest_payload) != len(records):
        raise ValueError("Canonical manifest row count does not match in-memory records")

    for record in records:
        csv_path = project_root / record.canonical_csv_path
        if not csv_path.exists():
            raise ValueError(f"Missing canonical CSV for {record.asset_id}")
        payload = csv_path.read_bytes()
        if sha256_prefixed(payload) != record.file_hash:
            raise ValueError(f"File hash mismatch for {record.asset_id}")
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CANONICAL_CSV_HEADER:
                raise ValueError(f"Canonical CSV header mismatch for {record.asset_id}")
            rows = list(reader)
        if len(rows) != record.row_count:
            raise ValueError(f"Canonical CSV row count mismatch for {record.asset_id}")
        if rows:
            if rows[0]["timestamp"] != record.first_timestamp:
                raise ValueError(f"Canonical CSV first timestamp mismatch for {record.asset_id}")
            if rows[-1]["timestamp"] != record.last_timestamp:
                raise ValueError(f"Canonical CSV last timestamp mismatch for {record.asset_id}")
            if any(row["asset_id"] != record.asset_id for row in rows):
                raise ValueError(f"Canonical CSV asset_id mismatch for {record.asset_id}")


def build_t08_outputs(
    eligibility_report_input: Path | str = default_eligibility_report_input(),
    path_manifest_input: Path | str = default_path_manifest_input(),
    contract_input: Path | str = default_contract_input(),
    canonical_output_dir: Path | str = default_canonical_output_dir(),
    manifest_output: Path | str = default_manifest_output(),
    repo_root: Path | str | None = None,
    project_root: Path | str | None = None,
) -> list[CanonicalDailyCloseManifestRecord]:
    require_contract(contract_input)
    repo_root_path = Path(repo_root) if repo_root is not None else default_repo_root()
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    canonical_output_dir_path = Path(canonical_output_dir)
    manifest_output_path = Path(manifest_output)

    eligible_records = load_eligibility_report(eligibility_report_input)
    path_manifest_index = build_path_manifest_index(path_manifest_input)

    manifest_records: list[CanonicalDailyCloseManifestRecord] = []
    canonical_output_dir_path.mkdir(parents=True, exist_ok=True)

    for eligible in eligible_records:
        resolution = path_manifest_index.get(eligible.asset_id)
        validate_resolution_alignment(eligible, resolution)

        asset_row = {
            "asset_id": eligible.asset_id,
            "symbol": eligible.symbol,
            "category": eligible.category,
        }
        extraction = extract_canonical_daily_close(asset_row, resolution, repo_root_path)
        if extraction.inspection.schema_status != "ADMISSIBLE":
            raise ValueError(
                f"Eligible asset failed schema admission during T-08: {eligible.asset_id}"
            )

        validate_screening_alignment(eligible, extraction.canonical_rows)

        output_path = canonical_output_dir_path / f"{eligible.asset_id}.csv"
        payload = serialize_canonical_daily_close_csv_bytes(
            eligible.asset_id, extraction.canonical_rows
        )
        output_path.write_bytes(payload)

        manifest_records.append(
            CanonicalDailyCloseManifestRecord(
                asset_id=eligible.asset_id,
                symbol=eligible.symbol,
                category=eligible.category,
                path_pattern=eligible.path_pattern,
                source_d_file_path=eligible.d_file_path,
                canonical_csv_path=project_relative_path(output_path, project_root_path),
                row_count=len(extraction.canonical_rows),
                first_timestamp=normalize_timestamp(extraction.canonical_rows[0]),
                last_timestamp=normalize_timestamp(extraction.canonical_rows[-1]),
                file_hash=sha256_prefixed(payload),
            )
        )

    write_json([asdict(record) for record in manifest_records], manifest_output_path)
    validate_manifest_records(
        manifest_records,
        canonical_output_dir=canonical_output_dir_path,
        manifest_output=manifest_output_path,
        project_root=project_root_path,
    )
    return manifest_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the canonical daily-close store for eligible FTMO assets."
    )
    parser.add_argument(
        "--eligibility-report-input",
        type=Path,
        default=default_eligibility_report_input(),
    )
    parser.add_argument(
        "--path-manifest-input", type=Path, default=default_path_manifest_input()
    )
    parser.add_argument("--contract-input", type=Path, default=default_contract_input())
    parser.add_argument(
        "--canonical-output-dir", type=Path, default=default_canonical_output_dir()
    )
    parser.add_argument("--manifest-output", type=Path, default=default_manifest_output())
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = build_t08_outputs(
        eligibility_report_input=args.eligibility_report_input,
        path_manifest_input=args.path_manifest_input,
        contract_input=args.contract_input,
        canonical_output_dir=args.canonical_output_dir,
        manifest_output=args.manifest_output,
        repo_root=args.repo_root,
    )
    print(f"Wrote {len(records)} canonical daily-close files and 1 manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
