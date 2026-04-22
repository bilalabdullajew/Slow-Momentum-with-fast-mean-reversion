from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lstm_cpd.canonical_daily_close_store import sha256_prefixed
from lstm_cpd.cpd.precompute import (
    CPD_CSV_SUFFIX,
    CPDFeatureRow,
    load_cpd_feature_csv,
    project_relative_path,
)
from lstm_cpd.cpd.precompute_contract import ALLOWED_CPD_LBWS, CPD_RESULT_STATUSES
from lstm_cpd.daily_close_contract import bool_to_text, default_project_root
from lstm_cpd.features.returns import load_canonical_daily_close_csv
from lstm_cpd.features.volatility import (
    CanonicalDailyCloseManifestRecord,
    load_canonical_daily_close_manifest,
)


TELEMETRY_HEADER = (
    "asset_id",
    "lbw",
    "cpd_csv_path",
    "row_count",
    "canonical_row_count",
    "first_timestamp",
    "last_timestamp",
    "success_count",
    "retry_success_count",
    "fallback_previous_count",
    "baseline_failure_count",
    "changepoint_failure_count",
    "invalid_window_count",
    "output_row_count",
    "retry_used_count",
    "fallback_used_count",
)

FAILURE_LEDGER_HEADER = (
    "asset_id",
    "lbw",
    "timestamp",
    "status",
    "window_size",
    "retry_used",
    "fallback_used",
    "failure_stage",
    "failure_message",
    "cpd_csv_path",
)

FALLBACK_LEDGER_HEADER = (
    "asset_id",
    "lbw",
    "timestamp",
    "status",
    "fallback_source_timestamp",
    "nu",
    "gamma",
    "nlml_baseline",
    "retry_used",
    "fallback_used",
    "failure_stage",
    "failure_message",
    "cpd_csv_path",
)


@dataclass(frozen=True)
class T15OutputArtifacts:
    telemetry_report_path: Path
    failure_ledger_path: Path
    fallback_ledger_path: Path
    manifest_output_path: Path


def default_input_dir() -> Path:
    return default_project_root() / "artifacts/features/cpd"


def default_canonical_manifest_input() -> Path:
    return default_project_root() / "artifacts/manifests/canonical_daily_close_manifest.json"


def default_telemetry_report_path() -> Path:
    return default_project_root() / "artifacts/reports/cpd_fit_telemetry.csv"


def default_failure_ledger_path() -> Path:
    return default_project_root() / "artifacts/reports/cpd_failure_ledger.csv"


def default_fallback_ledger_path() -> Path:
    return default_project_root() / "artifacts/reports/cpd_fallback_ledger.csv"


def default_manifest_output_path() -> Path:
    return default_project_root() / "artifacts/manifests/cpd_feature_store_manifest.json"


def _write_csv_rows(
    path: Path | str,
    *,
    header: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_manifest_json(
    path: Path | str,
    *,
    entries: Sequence[dict[str, object]],
) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(list(entries), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _collect_actual_relative_paths(input_dir: Path | str) -> set[str]:
    input_dir_path = Path(input_dir)
    return {
        path.relative_to(input_dir_path).as_posix()
        for path in input_dir_path.rglob(f"*{CPD_CSV_SUFFIX}")
        if path.is_file()
    }


def _status_counts(rows: Sequence[CPDFeatureRow]) -> dict[str, int]:
    counts = {status: 0 for status in CPD_RESULT_STATUSES}
    for row in rows:
        counts[row.status] += 1
    return counts


def _validate_canonical_timeline(
    manifest_record: CanonicalDailyCloseManifestRecord,
    *,
    rows: Sequence[CPDFeatureRow],
    project_root: Path | str,
) -> None:
    if len(rows) != manifest_record.row_count:
        raise ValueError(
            f"CPD row count mismatch for {manifest_record.asset_id}: "
            f"{len(rows)} != {manifest_record.row_count}"
        )
    canonical_rows = load_canonical_daily_close_csv(
        Path(project_root) / manifest_record.canonical_csv_path,
        expected_asset_id=manifest_record.asset_id,
    )
    if len(canonical_rows) != len(rows):
        raise ValueError(
            f"Canonical/CPD row count mismatch for {manifest_record.asset_id}: "
            f"{len(canonical_rows)} != {len(rows)}"
        )
    for row_index, (canonical_row, cpd_row) in enumerate(zip(canonical_rows, rows)):
        if canonical_row.timestamp != cpd_row.timestamp:
            raise ValueError(
                f"Canonical/CPD timestamp mismatch at row {row_index} for "
                f"{manifest_record.asset_id}: {canonical_row.timestamp} != "
                f"{cpd_row.timestamp}"
            )


def build_t15_outputs(
    input_dir: Path | str = default_input_dir(),
    canonical_manifest_input: Path | str = default_canonical_manifest_input(),
    *,
    project_root: Path | str | None = None,
    telemetry_report_path: Path | str = default_telemetry_report_path(),
    failure_ledger_path: Path | str = default_failure_ledger_path(),
    fallback_ledger_path: Path | str = default_fallback_ledger_path(),
    manifest_output_path: Path | str = default_manifest_output_path(),
    lbws: Sequence[int] = ALLOWED_CPD_LBWS,
) -> T15OutputArtifacts:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    input_dir_path = Path(input_dir)
    requested_lbws = tuple(dict.fromkeys(int(lbw) for lbw in lbws))
    if not requested_lbws:
        raise ValueError("At least one lbw is required")
    invalid_lbws = [lbw for lbw in requested_lbws if lbw not in ALLOWED_CPD_LBWS]
    if invalid_lbws:
        raise ValueError(f"Unsupported lbws requested: {invalid_lbws}")
    manifest_records = load_canonical_daily_close_manifest(canonical_manifest_input)

    expected_relative_paths = {
        f"lbw_{lbw}/{record.asset_id}{CPD_CSV_SUFFIX}"
        for record in manifest_records
        for lbw in requested_lbws
    }
    actual_relative_paths = _collect_actual_relative_paths(input_dir_path)
    if expected_relative_paths != actual_relative_paths:
        missing = sorted(expected_relative_paths - actual_relative_paths)
        extra = sorted(actual_relative_paths - expected_relative_paths)
        raise ValueError(
            "CPD feature-store mismatch against expected asset/LBW grid: "
            f"missing={missing}, extra={extra}"
        )

    telemetry_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []
    fallback_rows: list[dict[str, str]] = []
    manifest_entries: list[dict[str, object]] = []
    for manifest_record in manifest_records:
        for lbw in requested_lbws:
            relative_path = f"lbw_{lbw}/{manifest_record.asset_id}{CPD_CSV_SUFFIX}"
            cpd_path = input_dir_path / relative_path
            rows = load_cpd_feature_csv(
                cpd_path,
                expected_asset_id=manifest_record.asset_id,
                expected_lbw=lbw,
            )
            _validate_canonical_timeline(
                manifest_record,
                rows=rows,
                project_root=project_root_path,
            )
            if not rows:
                raise ValueError(f"CPD file is empty: {cpd_path}")

            counts = _status_counts(rows)
            retry_used_count = sum(1 for row in rows if row.retry_used)
            fallback_used_count = sum(1 for row in rows if row.fallback_used)
            output_row_count = sum(1 for row in rows if row.has_outputs)
            telemetry_rows.append(
                {
                    "asset_id": manifest_record.asset_id,
                    "lbw": str(lbw),
                    "cpd_csv_path": project_relative_path(cpd_path, project_root_path),
                    "row_count": str(len(rows)),
                    "canonical_row_count": str(manifest_record.row_count),
                    "first_timestamp": rows[0].timestamp,
                    "last_timestamp": rows[-1].timestamp,
                    "success_count": str(counts["success"]),
                    "retry_success_count": str(counts["retry_success"]),
                    "fallback_previous_count": str(counts["fallback_previous"]),
                    "baseline_failure_count": str(counts["baseline_failure"]),
                    "changepoint_failure_count": str(counts["changepoint_failure"]),
                    "invalid_window_count": str(counts["invalid_window"]),
                    "output_row_count": str(output_row_count),
                    "retry_used_count": str(retry_used_count),
                    "fallback_used_count": str(fallback_used_count),
                }
            )

            for row in rows:
                base_fields = {
                    "asset_id": manifest_record.asset_id,
                    "lbw": str(lbw),
                    "timestamp": row.timestamp,
                    "status": row.status,
                    "cpd_csv_path": project_relative_path(cpd_path, project_root_path),
                }
                if row.status in {
                    "invalid_window",
                    "baseline_failure",
                    "changepoint_failure",
                }:
                    failure_rows.append(
                        {
                            **base_fields,
                            "window_size": str(row.window_size),
                            "retry_used": bool_to_text(row.retry_used),
                            "fallback_used": bool_to_text(row.fallback_used),
                            "failure_stage": row.failure_stage or "",
                            "failure_message": row.failure_message or "",
                        }
                    )
                if row.status == "fallback_previous":
                    fallback_rows.append(
                        {
                            **base_fields,
                            "fallback_source_timestamp": row.fallback_source_timestamp or "",
                            "nu": "" if row.nu is None else format(row.nu, ".17g"),
                            "gamma": "" if row.gamma is None else format(row.gamma, ".17g"),
                            "nlml_baseline": (
                                ""
                                if row.nlml_baseline is None
                                else format(row.nlml_baseline, ".17g")
                            ),
                            "retry_used": bool_to_text(row.retry_used),
                            "fallback_used": bool_to_text(row.fallback_used),
                            "failure_stage": row.failure_stage or "",
                            "failure_message": row.failure_message or "",
                        }
                    )

            payload = cpd_path.read_bytes()
            manifest_entries.append(
                {
                    "asset_id": manifest_record.asset_id,
                    "lbw": lbw,
                    "cpd_csv_path": project_relative_path(cpd_path, project_root_path),
                    "row_count": len(rows),
                    "canonical_row_count": manifest_record.row_count,
                    "first_timestamp": rows[0].timestamp,
                    "last_timestamp": rows[-1].timestamp,
                    "output_row_count": output_row_count,
                    "retry_used_count": retry_used_count,
                    "fallback_used_count": fallback_used_count,
                    "status_counts": counts,
                    "matches_canonical_timeline": True,
                    "file_hash": sha256_prefixed(payload),
                }
            )

    _write_csv_rows(
        telemetry_report_path,
        header=TELEMETRY_HEADER,
        rows=telemetry_rows,
    )
    _write_csv_rows(
        failure_ledger_path,
        header=FAILURE_LEDGER_HEADER,
        rows=failure_rows,
    )
    _write_csv_rows(
        fallback_ledger_path,
        header=FALLBACK_LEDGER_HEADER,
        rows=fallback_rows,
    )
    _write_manifest_json(
        manifest_output_path,
        entries=manifest_entries,
    )
    return T15OutputArtifacts(
        telemetry_report_path=Path(telemetry_report_path),
        failure_ledger_path=Path(failure_ledger_path),
        fallback_ledger_path=Path(fallback_ledger_path),
        manifest_output_path=Path(manifest_output_path),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate CPD telemetry, failure ledgers, and manifest outputs."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument(
        "--canonical-manifest-input",
        type=Path,
        default=default_canonical_manifest_input(),
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument(
        "--telemetry-report-path",
        type=Path,
        default=default_telemetry_report_path(),
    )
    parser.add_argument(
        "--failure-ledger-path",
        type=Path,
        default=default_failure_ledger_path(),
    )
    parser.add_argument(
        "--fallback-ledger-path",
        type=Path,
        default=default_fallback_ledger_path(),
    )
    parser.add_argument(
        "--manifest-output-path",
        type=Path,
        default=default_manifest_output_path(),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_t15_outputs(
        input_dir=args.input_dir,
        canonical_manifest_input=args.canonical_manifest_input,
        project_root=args.project_root,
        telemetry_report_path=args.telemetry_report_path,
        failure_ledger_path=args.failure_ledger_path,
        fallback_ledger_path=args.fallback_ledger_path,
        manifest_output_path=args.manifest_output_path,
    )
    print(
        "Wrote CPD telemetry artifacts: "
        f"{outputs.telemetry_report_path}, "
        f"{outputs.failure_ledger_path}, "
        f"{outputs.fallback_ledger_path}, "
        f"{outputs.manifest_output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
