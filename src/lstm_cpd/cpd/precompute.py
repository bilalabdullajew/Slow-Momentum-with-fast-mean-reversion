from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - only exercised on non-POSIX systems.
    fcntl = None

from lstm_cpd.cpd.fit_window import fit_cpd_window
from lstm_cpd.cpd.precompute_contract import (
    ALLOWED_CPD_LBWS,
    CPDPreviousOutputs,
    CPD_RESULT_STATUSES,
    CPDWindowInput,
    CPDWindowResult,
    STATUS_BASELINE_FAILURE,
    STATUS_CHANGEPOINT_FAILURE,
    STATUS_FALLBACK_PREVIOUS,
    STATUS_INVALID_WINDOW,
    STATUS_RETRY_SUCCESS,
    STATUS_SUCCESS,
    is_allowed_lbw,
)
from lstm_cpd.daily_close_contract import bool_to_text, default_project_root
from lstm_cpd.features.returns import (
    RETURNS_VOLATILITY_HEADER,
    CanonicalDailyCloseRecord,
    load_canonical_daily_close_csv,
    serialize_optional_float,
)
from lstm_cpd.features.volatility import (
    CanonicalDailyCloseManifestRecord,
    load_canonical_daily_close_manifest,
)


RETURNS_VOLATILITY_SUFFIX = "_returns_volatility.csv"
CPD_CSV_SUFFIX = "_cpd.csv"
CPD_PARTIAL_SUFFIX = ".partial.csv"
CPD_CHECKPOINT_SUFFIX = ".checkpoint.json"
DEFAULT_FLUSH_ROWS = 25
MACOS_MAX_WORKERS = 3

DETERMINISTIC_WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
}

CPD_OUTPUT_HEADER = (
    "timestamp",
    "asset_id",
    "lbw",
    "nu",
    "gamma",
    "status",
    "window_size",
    "nlml_baseline",
    "nlml_changepoint",
    "retry_used",
    "fallback_used",
    "location_c",
    "steepness_s",
    "fallback_source_timestamp",
    "failure_stage",
    "failure_message",
)

PROGRESS_REPORT_HEADER = (
    "lbw",
    "asset_id",
    "state",
    "rows_written",
    "last_timestamp",
    "retry_count",
    "fallback_count",
    "started_at",
    "finished_at",
    "output_path",
    "error_message",
)

PROGRESS_STATE_PENDING = "pending"
PROGRESS_STATE_RUNNING = "running"
PROGRESS_STATE_COMPLETED = "completed"
PROGRESS_STATE_FAILED = "failed"
PROGRESS_STATES = (
    PROGRESS_STATE_PENDING,
    PROGRESS_STATE_RUNNING,
    PROGRESS_STATE_COMPLETED,
    PROGRESS_STATE_FAILED,
)

_UNSET = object()


@dataclass(frozen=True)
class ReturnsVolatilityRecord:
    timestamp: str
    asset_id: str
    arithmetic_return: float | None


@dataclass(frozen=True)
class CPDFeatureRow:
    timestamp: str
    asset_id: str
    lbw: int
    nu: float | None
    gamma: float | None
    status: str
    window_size: int
    nlml_baseline: float | None
    nlml_changepoint: float | None
    retry_used: bool
    fallback_used: bool
    location_c: float | None
    steepness_s: float | None
    fallback_source_timestamp: str | None
    failure_stage: str | None
    failure_message: str | None

    @property
    def has_outputs(self) -> bool:
        return self.nu is not None and self.gamma is not None


@dataclass(frozen=True)
class T14ChainTask:
    manifest_record: CanonicalDailyCloseManifestRecord
    lbw: int
    canonical_csv_path: Path
    returns_csv_path: Path
    output_path: Path

    @property
    def asset_id(self) -> str:
        return self.manifest_record.asset_id

    @property
    def partial_output_path(self) -> Path:
        return self.output_path.with_name(f"{self.output_path.stem}{CPD_PARTIAL_SUFFIX}")

    @property
    def checkpoint_path(self) -> Path:
        return self.output_path.with_name(f"{self.output_path.name}{CPD_CHECKPOINT_SUFFIX}")


@dataclass(frozen=True)
class T14ChainResumeState:
    rows_written: int
    previous_outputs: CPDPreviousOutputs | None
    previous_output_timestamp: str | None
    last_timestamp: str | None
    retry_count: int
    fallback_count: int


@dataclass(frozen=True)
class T14ProgressRow:
    lbw: int
    asset_id: str
    state: str
    rows_written: int
    last_timestamp: str | None
    retry_count: int
    fallback_count: int
    started_at: str | None
    finished_at: str | None
    output_path: str
    error_message: str | None


class T14ChainStopRequested(RuntimeError):
    pass


def default_canonical_manifest_input() -> Path:
    return default_project_root() / "artifacts/manifests/canonical_daily_close_manifest.json"


def default_returns_input_dir() -> Path:
    return default_project_root() / "artifacts/features/base"


def default_output_dir() -> Path:
    return default_project_root() / "artifacts/features/cpd"


def default_progress_report_path(project_root: Path | str | None = None) -> Path:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    return project_root_path / "artifacts/reports/cpd_precompute_progress.csv"


def default_parallel_workers() -> int:
    return 1


def clamp_parallel_workers(max_workers: int) -> int:
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive: {max_workers}")
    if platform.system() == "Darwin":
        return min(max_workers, MACOS_MAX_WORKERS)
    return max_workers


def project_relative_path(path: Path | str, project_root: Path | str) -> str:
    resolved_path = Path(path).resolve()
    resolved_root = Path(project_root).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _serialize_optional_text(value: str | None) -> str:
    if value is None:
        return ""
    sanitized = value.replace("\r", " ").replace("\n", " ")
    return sanitized.strip()


def _utcnow_text() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_optional_float_text(
    text: str,
    *,
    csv_path: Path,
    row_index: int,
    column_name: str,
) -> float | None:
    if text == "":
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(
            f"Row {row_index} column {column_name} has non-finite value: {csv_path}"
        )
    return value


def _parse_bool_text(
    text: str,
    *,
    csv_path: Path,
    row_index: int,
    column_name: str,
) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(
        f"Row {row_index} column {column_name} must be 'true' or 'false': {csv_path}"
    )


def _validate_requested_lbws(lbws: Sequence[int]) -> tuple[int, ...]:
    unique_lbws = tuple(dict.fromkeys(int(lbw) for lbw in lbws))
    if not unique_lbws:
        raise ValueError("At least one lbw is required")
    invalid_lbws = [lbw for lbw in unique_lbws if not is_allowed_lbw(lbw)]
    if invalid_lbws:
        raise ValueError(f"Unsupported lbws requested: {invalid_lbws}")
    return unique_lbws


def _normalize_asset_filter(asset_ids: Sequence[str] | None) -> tuple[str, ...] | None:
    if asset_ids is None:
        return None
    normalized = tuple(dict.fromkeys(str(asset_id) for asset_id in asset_ids))
    if not normalized:
        return None
    if any(asset_id == "" for asset_id in normalized):
        raise ValueError("asset_id filters must not be empty")
    return normalized


def _build_asset_path_map(
    input_dir: Path | str,
    *,
    suffix: str,
) -> dict[str, Path]:
    input_dir_path = Path(input_dir)
    asset_map: dict[str, Path] = {}
    for path in sorted(input_dir_path.glob(f"*{suffix}")):
        if not path.is_file():
            continue
        asset_id = path.name[: -len(suffix)]
        if asset_id in asset_map:
            raise ValueError(f"Duplicate input asset_id for suffix {suffix}: {asset_id}")
        asset_map[asset_id] = path
    return asset_map


def collect_returns_volatility_paths(
    manifest_records: Sequence[CanonicalDailyCloseManifestRecord],
    input_dir: Path | str,
) -> dict[str, Path]:
    actual_paths = _build_asset_path_map(
        input_dir,
        suffix=RETURNS_VOLATILITY_SUFFIX,
    )
    expected_assets = {record.asset_id for record in manifest_records}
    actual_assets = set(actual_paths)
    if expected_assets != actual_assets:
        only_manifest = sorted(expected_assets - actual_assets)
        only_inputs = sorted(actual_assets - expected_assets)
        raise ValueError(
            "Returns-volatility asset mismatch against canonical manifest: "
            f"only_manifest={only_manifest}, only_inputs={only_inputs}"
        )
    return actual_paths


def load_returns_volatility_csv(
    path: Path | str,
    *,
    expected_asset_id: str | None = None,
) -> list[ReturnsVolatilityRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RETURNS_VOLATILITY_HEADER:
            raise ValueError(f"Returns-volatility header mismatch: {csv_path}")

        rows: list[ReturnsVolatilityRecord] = []
        previous_timestamp: str | None = None
        for row_index, row in enumerate(reader):
            timestamp = row["timestamp"]
            asset_id = row["asset_id"]
            if not timestamp:
                raise ValueError(
                    f"Returns-volatility row {row_index} missing timestamp: {csv_path}"
                )
            if not asset_id:
                raise ValueError(
                    f"Returns-volatility row {row_index} missing asset_id: {csv_path}"
                )
            if expected_asset_id is not None and asset_id != expected_asset_id:
                raise ValueError(
                    f"Returns-volatility row {row_index} asset_id mismatch for "
                    f"{csv_path}: {asset_id}"
                )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(
                    f"Returns-volatility timestamps must be strictly ascending in "
                    f"{csv_path}: {previous_timestamp} then {timestamp}"
                )
            previous_timestamp = timestamp

            close_text = row["close"]
            if close_text == "":
                raise ValueError(
                    f"Returns-volatility row {row_index} missing close: {csv_path}"
                )
            close_value = float(close_text)
            if not math.isfinite(close_value):
                raise ValueError(
                    f"Returns-volatility row {row_index} has non-finite close: {csv_path}"
                )

            arithmetic_return = _parse_optional_float_text(
                row["arithmetic_return"],
                csv_path=csv_path,
                row_index=row_index,
                column_name="arithmetic_return",
            )
            _parse_optional_float_text(
                row["sigma_t"],
                csv_path=csv_path,
                row_index=row_index,
                column_name="sigma_t",
            )

            rows.append(
                ReturnsVolatilityRecord(
                    timestamp=timestamp,
                    asset_id=asset_id,
                    arithmetic_return=arithmetic_return,
                )
            )
    return rows


def validate_canonical_returns_alignment(
    manifest_record: CanonicalDailyCloseManifestRecord,
    canonical_rows: Sequence[CanonicalDailyCloseRecord],
    returns_rows: Sequence[ReturnsVolatilityRecord],
) -> None:
    if len(canonical_rows) != manifest_record.row_count:
        raise ValueError(
            f"Canonical row count mismatch for {manifest_record.asset_id}: "
            f"{len(canonical_rows)} != {manifest_record.row_count}"
        )
    if len(returns_rows) != manifest_record.row_count:
        raise ValueError(
            f"Returns-volatility row count mismatch for {manifest_record.asset_id}: "
            f"{len(returns_rows)} != {manifest_record.row_count}"
        )
    if not canonical_rows:
        raise ValueError(f"Canonical series is empty for {manifest_record.asset_id}")
    if canonical_rows[0].timestamp != manifest_record.first_timestamp:
        raise ValueError(f"Canonical first timestamp mismatch for {manifest_record.asset_id}")
    if canonical_rows[-1].timestamp != manifest_record.last_timestamp:
        raise ValueError(f"Canonical last timestamp mismatch for {manifest_record.asset_id}")

    for row_index, (canonical_row, returns_row) in enumerate(
        zip(canonical_rows, returns_rows)
    ):
        if canonical_row.asset_id != manifest_record.asset_id:
            raise ValueError(
                f"Canonical asset_id mismatch at row {row_index} for "
                f"{manifest_record.asset_id}"
            )
        if returns_row.asset_id != manifest_record.asset_id:
            raise ValueError(
                f"Returns-volatility asset_id mismatch at row {row_index} for "
                f"{manifest_record.asset_id}"
            )
        if canonical_row.timestamp != returns_row.timestamp:
            raise ValueError(
                f"Canonical/returns timestamp mismatch at row {row_index} for "
                f"{manifest_record.asset_id}: {canonical_row.timestamp} != "
                f"{returns_row.timestamp}"
            )


def build_window_returns(
    returns_rows: Sequence[ReturnsVolatilityRecord],
    *,
    end_index: int,
    lbw: int,
) -> tuple[float, ...]:
    start_index = max(0, end_index - lbw)
    values: list[float] = []
    for row in returns_rows[start_index : end_index + 1]:
        if row.arithmetic_return is None:
            values.append(math.nan)
        else:
            values.append(row.arithmetic_return)
    return tuple(values)


def build_t14_task_specs(
    canonical_manifest_input: Path | str = default_canonical_manifest_input(),
    returns_input_dir: Path | str = default_returns_input_dir(),
    output_dir: Path | str = default_output_dir(),
    *,
    project_root: Path | str | None = None,
    lbws: Sequence[int] = ALLOWED_CPD_LBWS,
    asset_ids: Sequence[str] | None = None,
) -> list[T14ChainTask]:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    output_dir_path = Path(output_dir)
    requested_lbws = _validate_requested_lbws(lbws)
    manifest_records = load_canonical_daily_close_manifest(canonical_manifest_input)
    asset_filter = _normalize_asset_filter(asset_ids)
    if asset_filter is not None:
        manifest_by_asset = {record.asset_id: record for record in manifest_records}
        missing_assets = sorted(set(asset_filter) - set(manifest_by_asset))
        if missing_assets:
            raise ValueError(f"Unknown asset_id filters: {missing_assets}")
        manifest_records = [manifest_by_asset[asset_id] for asset_id in asset_filter]

    returns_paths = collect_returns_volatility_paths(manifest_records, returns_input_dir)

    tasks: list[T14ChainTask] = []
    for manifest_record in manifest_records:
        canonical_csv_path = project_root_path / manifest_record.canonical_csv_path
        returns_csv_path = returns_paths[manifest_record.asset_id]
        for lbw in requested_lbws:
            tasks.append(
                T14ChainTask(
                    manifest_record=manifest_record,
                    lbw=lbw,
                    canonical_csv_path=canonical_csv_path,
                    returns_csv_path=returns_csv_path,
                    output_path=output_dir_path / f"lbw_{lbw}" / f"{manifest_record.asset_id}{CPD_CSV_SUFFIX}",
                )
            )
    tasks.sort(key=lambda task: (task.asset_id, task.lbw))
    return tasks


def _serialize_cpd_result_row(
    *,
    row: ReturnsVolatilityRecord,
    lbw: int,
    result: CPDWindowResult,
    fallback_source_timestamp: str | None,
) -> dict[str, str]:
    return {
        "timestamp": row.timestamp,
        "asset_id": row.asset_id,
        "lbw": str(lbw),
        "nu": serialize_optional_float(result.nu),
        "gamma": serialize_optional_float(result.gamma),
        "status": result.status,
        "window_size": str(result.window_size),
        "nlml_baseline": serialize_optional_float(result.nlml_baseline),
        "nlml_changepoint": serialize_optional_float(result.nlml_changepoint),
        "retry_used": bool_to_text(result.retry_used),
        "fallback_used": bool_to_text(result.fallback_used),
        "location_c": serialize_optional_float(result.location_c),
        "steepness_s": serialize_optional_float(result.steepness_s),
        "fallback_source_timestamp": _serialize_optional_text(fallback_source_timestamp),
        "failure_stage": _serialize_optional_text(result.failure_stage),
        "failure_message": _serialize_optional_text(result.failure_message),
    }


def compute_cpd_feature_row(
    returns_rows: Sequence[ReturnsVolatilityRecord],
    *,
    end_index: int,
    lbw: int,
    previous_outputs: CPDPreviousOutputs | None,
    previous_output_timestamp: str | None,
    fit_window_fn: Callable[[CPDWindowInput], CPDWindowResult] = fit_cpd_window,
) -> tuple[dict[str, str], CPDPreviousOutputs | None, str | None]:
    row = returns_rows[end_index]
    result = fit_window_fn(
        CPDWindowInput(
            lbw=lbw,
            window_returns=build_window_returns(
                returns_rows,
                end_index=end_index,
                lbw=lbw,
            ),
            window_end_timestamp=row.timestamp,
            previous_outputs=previous_outputs,
        )
    )
    fallback_source_timestamp = (
        previous_output_timestamp if result.status == STATUS_FALLBACK_PREVIOUS else None
    )
    output_row = _serialize_cpd_result_row(
        row=row,
        lbw=lbw,
        result=result,
        fallback_source_timestamp=fallback_source_timestamp,
    )
    if result.has_outputs:
        return (
            output_row,
            CPDPreviousOutputs(nu=result.nu, gamma=result.gamma),
            row.timestamp,
        )
    return output_row, None, None


def build_cpd_feature_rows(
    returns_rows: Sequence[ReturnsVolatilityRecord],
    *,
    lbw: int,
    fit_window_fn: Callable[[CPDWindowInput], CPDWindowResult] = fit_cpd_window,
) -> list[dict[str, str]]:
    if not is_allowed_lbw(lbw):
        raise ValueError(f"Unsupported lbw: {lbw}")

    output_rows: list[dict[str, str]] = []
    previous_outputs: CPDPreviousOutputs | None = None
    previous_output_timestamp: str | None = None
    for end_index, _ in enumerate(returns_rows):
        output_row, previous_outputs, previous_output_timestamp = compute_cpd_feature_row(
            returns_rows,
            end_index=end_index,
            lbw=lbw,
            previous_outputs=previous_outputs,
            previous_output_timestamp=previous_output_timestamp,
            fit_window_fn=fit_window_fn,
        )
        output_rows.append(output_row)

    return output_rows


def write_cpd_feature_csv(
    rows: Sequence[dict[str, str]],
    output_path: Path | str,
) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(CPD_OUTPUT_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_cpd_feature_csv(
    path: Path | str,
    *,
    expected_asset_id: str | None = None,
    expected_lbw: int | None = None,
) -> list[CPDFeatureRow]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CPD_OUTPUT_HEADER:
            raise ValueError(f"CPD feature header mismatch: {csv_path}")

        rows: list[CPDFeatureRow] = []
        previous_timestamp: str | None = None
        for row_index, row in enumerate(reader):
            timestamp = row["timestamp"]
            asset_id = row["asset_id"]
            if not timestamp:
                raise ValueError(f"CPD row {row_index} missing timestamp: {csv_path}")
            if not asset_id:
                raise ValueError(f"CPD row {row_index} missing asset_id: {csv_path}")
            if expected_asset_id is not None and asset_id != expected_asset_id:
                raise ValueError(
                    f"CPD row {row_index} asset_id mismatch for {csv_path}: {asset_id}"
                )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(
                    f"CPD timestamps must be strictly ascending in {csv_path}: "
                    f"{previous_timestamp} then {timestamp}"
                )
            previous_timestamp = timestamp

            lbw = int(row["lbw"])
            if expected_lbw is not None and lbw != expected_lbw:
                raise ValueError(
                    f"CPD row {row_index} lbw mismatch for {csv_path}: {lbw}"
                )
            if not is_allowed_lbw(lbw):
                raise ValueError(f"CPD row {row_index} has unsupported lbw: {csv_path}")

            status = row["status"]
            if status not in CPD_RESULT_STATUSES:
                raise ValueError(f"CPD row {row_index} has invalid status: {csv_path}")

            retry_used = _parse_bool_text(
                row["retry_used"],
                csv_path=csv_path,
                row_index=row_index,
                column_name="retry_used",
            )
            fallback_used = _parse_bool_text(
                row["fallback_used"],
                csv_path=csv_path,
                row_index=row_index,
                column_name="fallback_used",
            )
            cpd_row = CPDFeatureRow(
                timestamp=timestamp,
                asset_id=asset_id,
                lbw=lbw,
                nu=_parse_optional_float_text(
                    row["nu"],
                    csv_path=csv_path,
                    row_index=row_index,
                    column_name="nu",
                ),
                gamma=_parse_optional_float_text(
                    row["gamma"],
                    csv_path=csv_path,
                    row_index=row_index,
                    column_name="gamma",
                ),
                status=status,
                window_size=int(row["window_size"]),
                nlml_baseline=_parse_optional_float_text(
                    row["nlml_baseline"],
                    csv_path=csv_path,
                    row_index=row_index,
                    column_name="nlml_baseline",
                ),
                nlml_changepoint=_parse_optional_float_text(
                    row["nlml_changepoint"],
                    csv_path=csv_path,
                    row_index=row_index,
                    column_name="nlml_changepoint",
                ),
                retry_used=retry_used,
                fallback_used=fallback_used,
                location_c=_parse_optional_float_text(
                    row["location_c"],
                    csv_path=csv_path,
                    row_index=row_index,
                    column_name="location_c",
                ),
                steepness_s=_parse_optional_float_text(
                    row["steepness_s"],
                    csv_path=csv_path,
                    row_index=row_index,
                    column_name="steepness_s",
                ),
                fallback_source_timestamp=row["fallback_source_timestamp"] or None,
                failure_stage=row["failure_stage"] or None,
                failure_message=row["failure_message"] or None,
            )
            _validate_cpd_row_semantics(cpd_row, csv_path=csv_path, row_index=row_index)
            rows.append(cpd_row)
    return rows


def _validate_cpd_row_semantics(
    row: CPDFeatureRow,
    *,
    csv_path: Path,
    row_index: int,
) -> None:
    if row.window_size < 0:
        raise ValueError(f"CPD row {row_index} has negative window_size: {csv_path}")
    if row.has_outputs:
        if not 0.0 <= row.nu <= 1.0:
            raise ValueError(f"CPD row {row_index} has out-of-range nu: {csv_path}")
        if not 0.0 <= row.gamma <= 1.0:
            raise ValueError(f"CPD row {row_index} has out-of-range gamma: {csv_path}")

    statuses_requiring_outputs = {
        STATUS_SUCCESS,
        STATUS_RETRY_SUCCESS,
        STATUS_FALLBACK_PREVIOUS,
    }
    statuses_without_outputs = {
        STATUS_INVALID_WINDOW,
        STATUS_BASELINE_FAILURE,
        STATUS_CHANGEPOINT_FAILURE,
    }
    if row.status in statuses_requiring_outputs and not row.has_outputs:
        raise ValueError(
            f"CPD row {row_index} must contain outputs for status={row.status}: {csv_path}"
        )
    if row.status in statuses_without_outputs and row.has_outputs:
        raise ValueError(
            f"CPD row {row_index} must not contain outputs for status={row.status}: "
            f"{csv_path}"
        )
    if row.status == STATUS_FALLBACK_PREVIOUS:
        if not row.fallback_used or row.fallback_source_timestamp is None:
            raise ValueError(
                f"CPD row {row_index} fallback status missing fallback fields: {csv_path}"
            )
    elif row.fallback_used or row.fallback_source_timestamp is not None:
        raise ValueError(
            f"CPD row {row_index} has fallback fields outside fallback status: {csv_path}"
        )

    statuses_requiring_retry = {
        STATUS_RETRY_SUCCESS,
        STATUS_FALLBACK_PREVIOUS,
        STATUS_CHANGEPOINT_FAILURE,
    }
    statuses_without_retry = {
        STATUS_SUCCESS,
        STATUS_INVALID_WINDOW,
        STATUS_BASELINE_FAILURE,
    }
    if row.status in statuses_requiring_retry and not row.retry_used:
        raise ValueError(
            f"CPD row {row_index} must set retry_used for status={row.status}: {csv_path}"
        )
    if row.status in statuses_without_retry and row.retry_used:
        raise ValueError(
            f"CPD row {row_index} must not set retry_used for status={row.status}: "
            f"{csv_path}"
        )


def _validate_cpd_timeline_prefix(
    rows: Sequence[CPDFeatureRow],
    returns_rows: Sequence[ReturnsVolatilityRecord],
    *,
    csv_path: Path,
) -> None:
    if len(rows) > len(returns_rows):
        raise ValueError(
            f"CPD rows exceed returns timeline in {csv_path}: {len(rows)} > {len(returns_rows)}"
        )
    for row_index, cpd_row in enumerate(rows):
        expected_timestamp = returns_rows[row_index].timestamp
        if cpd_row.timestamp != expected_timestamp:
            raise ValueError(
                f"CPD timestamp mismatch at row {row_index} in {csv_path}: "
                f"{cpd_row.timestamp} != {expected_timestamp}"
            )


def _summarize_cpd_rows(rows: Sequence[CPDFeatureRow]) -> T14ChainResumeState:
    previous_output_row: CPDFeatureRow | None = None
    for row in reversed(rows):
        if row.has_outputs:
            previous_output_row = row
            break
    previous_outputs = None
    previous_output_timestamp = None
    if previous_output_row is not None:
        previous_outputs = CPDPreviousOutputs(
            nu=previous_output_row.nu,
            gamma=previous_output_row.gamma,
        )
        previous_output_timestamp = previous_output_row.timestamp
    return T14ChainResumeState(
        rows_written=len(rows),
        previous_outputs=previous_outputs,
        previous_output_timestamp=previous_output_timestamp,
        last_timestamp=None if not rows else rows[-1].timestamp,
        retry_count=sum(1 for row in rows if row.retry_used),
        fallback_count=sum(1 for row in rows if row.fallback_used),
    )


def _fsync_handle(handle: object) -> None:
    file_handle = handle
    file_handle.flush()
    os.fsync(file_handle.fileno())


def _repair_partial_output(task: T14ChainTask) -> None:
    partial_path = task.partial_output_path
    if not partial_path.exists():
        return

    with partial_path.open("r", encoding="utf-8", newline="") as handle:
        raw_lines = handle.readlines()
    if not raw_lines:
        raise ValueError(f"Partial output is empty: {partial_path}")

    repaired = False
    kept_lines: list[str] = []
    for line_index, raw_line in enumerate(raw_lines):
        is_last_line = line_index == len(raw_lines) - 1
        has_newline = raw_line.endswith("\n")
        parsed_rows = list(csv.reader([raw_line]))
        if len(parsed_rows) != 1:
            if is_last_line:
                repaired = True
                break
            raise ValueError(f"Malformed partial CSV row {line_index}: {partial_path}")
        fields = parsed_rows[0]
        if line_index == 0:
            if tuple(fields) != CPD_OUTPUT_HEADER:
                raise ValueError(f"Partial output header mismatch: {partial_path}")
            if not has_newline:
                raw_line = raw_line + "\n"
                repaired = True
            kept_lines.append(raw_line)
            continue
        if not has_newline:
            repaired = True
            break
        if len(fields) != len(CPD_OUTPUT_HEADER):
            if is_last_line:
                repaired = True
                break
            raise ValueError(
                f"Malformed partial CSV field count at row {line_index}: {partial_path}"
            )
        kept_lines.append(raw_line)

    if not repaired:
        return
    with partial_path.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(kept_lines)
        _fsync_handle(handle)


def _load_partial_chain_progress(
    task: T14ChainTask,
    *,
    returns_rows: Sequence[ReturnsVolatilityRecord],
) -> T14ChainResumeState:
    if task.checkpoint_path.exists() and not task.partial_output_path.exists():
        if not task.output_path.exists():
            raise ValueError(
                f"Legacy checkpoint exists without partial/final output for "
                f"{task.asset_id} lbw={task.lbw}: {task.checkpoint_path}"
            )

    if not task.partial_output_path.exists():
        return T14ChainResumeState(
            rows_written=0,
            previous_outputs=None,
            previous_output_timestamp=None,
            last_timestamp=None,
            retry_count=0,
            fallback_count=0,
        )

    _repair_partial_output(task)
    partial_rows = load_cpd_feature_csv(
        task.partial_output_path,
        expected_asset_id=task.asset_id,
        expected_lbw=task.lbw,
    )
    _validate_cpd_timeline_prefix(
        partial_rows,
        returns_rows,
        csv_path=task.partial_output_path,
    )
    return _summarize_cpd_rows(partial_rows)


def _load_completed_output_summary(
    task: T14ChainTask,
    *,
    returns_rows: Sequence[ReturnsVolatilityRecord],
) -> T14ChainResumeState:
    completed_rows = load_cpd_feature_csv(
        task.output_path,
        expected_asset_id=task.asset_id,
        expected_lbw=task.lbw,
    )
    if len(completed_rows) != len(returns_rows):
        raise ValueError(
            f"Completed CPD output row count mismatch for {task.asset_id} lbw={task.lbw}: "
            f"{len(completed_rows)} != {len(returns_rows)}"
        )
    _validate_cpd_timeline_prefix(
        completed_rows,
        returns_rows,
        csv_path=task.output_path,
    )
    return _summarize_cpd_rows(completed_rows)


def _remove_if_exists(path: Path | str) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()


def _prepare_partial_writer(
    task: T14ChainTask,
    *,
    use_append: bool,
) -> tuple[csv.DictWriter, object]:
    task.partial_output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = task.partial_output_path.open(
        "a" if use_append else "w",
        encoding="utf-8",
        newline="",
    )
    writer = csv.DictWriter(
        handle,
        fieldnames=list(CPD_OUTPUT_HEADER),
        lineterminator="\n",
    )
    if not use_append:
        writer.writeheader()
        _fsync_handle(handle)
    return writer, handle


def _cleanup_job_memory() -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session(free_memory=False)
    except Exception:
        pass
    gc.collect()


@contextmanager
def _progress_lock(progress_report_path: Path) -> object:
    lock_path = progress_report_path.with_name(f"{progress_report_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _deserialize_progress_row(row: dict[str, str]) -> T14ProgressRow:
    state = row["state"]
    if state not in PROGRESS_STATES:
        raise ValueError(f"Invalid progress state: {state}")
    return T14ProgressRow(
        lbw=int(row["lbw"]),
        asset_id=row["asset_id"],
        state=state,
        rows_written=int(row["rows_written"]),
        last_timestamp=row["last_timestamp"] or None,
        retry_count=int(row["retry_count"]),
        fallback_count=int(row["fallback_count"]),
        started_at=row["started_at"] or None,
        finished_at=row["finished_at"] or None,
        output_path=row["output_path"],
        error_message=row["error_message"] or None,
    )


def _serialize_progress_row(row: T14ProgressRow) -> dict[str, str]:
    return {
        "lbw": str(row.lbw),
        "asset_id": row.asset_id,
        "state": row.state,
        "rows_written": str(row.rows_written),
        "last_timestamp": row.last_timestamp or "",
        "retry_count": str(row.retry_count),
        "fallback_count": str(row.fallback_count),
        "started_at": row.started_at or "",
        "finished_at": row.finished_at or "",
        "output_path": row.output_path,
        "error_message": _serialize_optional_text(row.error_message),
    }


def _read_progress_rows_unlocked(
    progress_report_path: Path,
) -> dict[tuple[str, int], T14ProgressRow]:
    if not progress_report_path.exists():
        return {}
    with progress_report_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PROGRESS_REPORT_HEADER:
            raise ValueError(f"Progress report header mismatch: {progress_report_path}")
        rows: dict[tuple[str, int], T14ProgressRow] = {}
        for raw_row in reader:
            row = _deserialize_progress_row(raw_row)
            key = (row.asset_id, row.lbw)
            rows[key] = row
    return rows


def _write_progress_rows_unlocked(
    progress_report_path: Path,
    rows: Sequence[T14ProgressRow],
) -> None:
    progress_report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = progress_report_path.with_name(f"{progress_report_path.name}.tmp")
    ordered_rows = sorted(rows, key=lambda row: (row.asset_id, row.lbw))
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(PROGRESS_REPORT_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(_serialize_progress_row(row))
        _fsync_handle(handle)
    temp_path.replace(progress_report_path)


def _default_progress_row(
    task: T14ChainTask,
    *,
    project_root: Path,
) -> T14ProgressRow:
    return T14ProgressRow(
        lbw=task.lbw,
        asset_id=task.asset_id,
        state=PROGRESS_STATE_PENDING,
        rows_written=0,
        last_timestamp=None,
        retry_count=0,
        fallback_count=0,
        started_at=None,
        finished_at=None,
        output_path=project_relative_path(task.output_path, project_root),
        error_message=None,
    )


def _update_progress_row(
    progress_report_path: Path | str,
    task: T14ChainTask,
    *,
    project_root: Path,
    state: str | object = _UNSET,
    rows_written: int | object = _UNSET,
    last_timestamp: str | None | object = _UNSET,
    retry_count: int | object = _UNSET,
    fallback_count: int | object = _UNSET,
    started_at: str | None | object = _UNSET,
    finished_at: str | None | object = _UNSET,
    error_message: str | None | object = _UNSET,
) -> T14ProgressRow:
    progress_path = Path(progress_report_path)
    with _progress_lock(progress_path):
        rows = _read_progress_rows_unlocked(progress_path)
        key = (task.asset_id, task.lbw)
        current = rows.get(key) or _default_progress_row(task, project_root=project_root)
        updates: dict[str, object] = {
            "output_path": project_relative_path(task.output_path, project_root),
        }
        if state is not _UNSET:
            if state not in PROGRESS_STATES:
                raise ValueError(f"Invalid progress state: {state}")
            updates["state"] = state
        if rows_written is not _UNSET:
            updates["rows_written"] = rows_written
        if last_timestamp is not _UNSET:
            updates["last_timestamp"] = last_timestamp
        if retry_count is not _UNSET:
            updates["retry_count"] = retry_count
        if fallback_count is not _UNSET:
            updates["fallback_count"] = fallback_count
        if started_at is not _UNSET:
            updates["started_at"] = started_at
        if finished_at is not _UNSET:
            updates["finished_at"] = finished_at
        if error_message is not _UNSET:
            updates["error_message"] = error_message
        updated = replace(current, **updates)
        rows[key] = updated
        _write_progress_rows_unlocked(progress_path, list(rows.values()))
        return updated


def initialize_progress_report(
    tasks: Sequence[T14ChainTask],
    *,
    progress_report_path: Path | str,
    project_root: Path,
) -> Path:
    progress_path = Path(progress_report_path)
    with _progress_lock(progress_path):
        existing_rows = _read_progress_rows_unlocked(progress_path)
        merged_rows: dict[tuple[str, int], T14ProgressRow] = dict(existing_rows)
        for task in tasks:
            key = (task.asset_id, task.lbw)
            existing = existing_rows.get(key)
            if existing is None:
                row = _default_progress_row(task, project_root=project_root)
                if task.output_path.exists():
                    try:
                        summary = _summarize_cpd_rows(
                            load_cpd_feature_csv(
                                task.output_path,
                                expected_asset_id=task.asset_id,
                                expected_lbw=task.lbw,
                            )
                        )
                        row = replace(
                            row,
                            state=PROGRESS_STATE_COMPLETED,
                            rows_written=summary.rows_written,
                            last_timestamp=summary.last_timestamp,
                            retry_count=summary.retry_count,
                            fallback_count=summary.fallback_count,
                        )
                    except Exception as exc:
                        row = replace(
                            row,
                            state=PROGRESS_STATE_FAILED,
                            finished_at=_utcnow_text(),
                            error_message=str(exc),
                        )
                elif task.partial_output_path.exists():
                    try:
                        _repair_partial_output(task)
                        summary = _summarize_cpd_rows(
                            load_cpd_feature_csv(
                                task.partial_output_path,
                                expected_asset_id=task.asset_id,
                                expected_lbw=task.lbw,
                            )
                        )
                        row = replace(
                            row,
                            state=PROGRESS_STATE_RUNNING,
                            rows_written=summary.rows_written,
                            last_timestamp=summary.last_timestamp,
                            retry_count=summary.retry_count,
                            fallback_count=summary.fallback_count,
                        )
                    except Exception as exc:
                        row = replace(
                            row,
                            state=PROGRESS_STATE_FAILED,
                            finished_at=_utcnow_text(),
                            error_message=str(exc),
                        )
                merged_rows[key] = row
                continue
            merged_rows[key] = replace(
                existing,
                output_path=project_relative_path(task.output_path, project_root),
            )
        _write_progress_rows_unlocked(progress_path, list(merged_rows.values()))
    return progress_path


def run_t14_chain_task(
    task: T14ChainTask,
    *,
    fit_window_fn: Callable[[CPDWindowInput], CPDWindowResult] = fit_cpd_window,
    resume: bool = False,
    skip_if_complete: bool = False,
    stop_after_rows: int | None = None,
    progress_report_path: Path | str | None = None,
    project_root: Path | str | None = None,
    flush_rows: int = DEFAULT_FLUSH_ROWS,
) -> Path:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    if flush_rows <= 0:
        raise ValueError(f"flush_rows must be positive: {flush_rows}")
    if stop_after_rows is not None and stop_after_rows <= 0:
        raise ValueError(f"stop_after_rows must be positive: {stop_after_rows}")

    canonical_rows = load_canonical_daily_close_csv(
        task.canonical_csv_path,
        expected_asset_id=task.asset_id,
    )
    returns_rows = load_returns_volatility_csv(
        task.returns_csv_path,
        expected_asset_id=task.asset_id,
    )
    validate_canonical_returns_alignment(
        task.manifest_record,
        canonical_rows,
        returns_rows,
    )

    progress_path = (
        None if progress_report_path is None else Path(progress_report_path)
    )
    if progress_path is not None:
        _update_progress_row(
            progress_path,
            task,
            project_root=project_root_path,
        )

    try:
        if task.output_path.exists():
            if skip_if_complete:
                completed_summary = _load_completed_output_summary(
                    task,
                    returns_rows=returns_rows,
                )
                _remove_if_exists(task.partial_output_path)
                _remove_if_exists(task.checkpoint_path)
                if progress_path is not None:
                    timestamp = _utcnow_text()
                    _update_progress_row(
                        progress_path,
                        task,
                        project_root=project_root_path,
                        state=PROGRESS_STATE_COMPLETED,
                        rows_written=completed_summary.rows_written,
                        last_timestamp=completed_summary.last_timestamp,
                        retry_count=completed_summary.retry_count,
                        fallback_count=completed_summary.fallback_count,
                        started_at=timestamp,
                        finished_at=timestamp,
                        error_message=None,
                    )
                return task.output_path
            _remove_if_exists(task.output_path)

        if not resume:
            _remove_if_exists(task.partial_output_path)
            _remove_if_exists(task.checkpoint_path)
            progress = T14ChainResumeState(
                rows_written=0,
                previous_outputs=None,
                previous_output_timestamp=None,
                last_timestamp=None,
                retry_count=0,
                fallback_count=0,
            )
        else:
            progress = _load_partial_chain_progress(
                task,
                returns_rows=returns_rows,
            )

        if progress.rows_written > len(returns_rows):
            raise ValueError(
                f"Partial progress exceeds available rows for {task.asset_id} lbw={task.lbw}: "
                f"{progress.rows_written} > {len(returns_rows)}"
            )
        if stop_after_rows is not None and progress.rows_written >= stop_after_rows:
            raise ValueError(
                f"stop_after_rows={stop_after_rows} must exceed existing progress "
                f"{progress.rows_written} for {task.asset_id} lbw={task.lbw}"
            )

        if progress.rows_written == len(returns_rows):
            if not task.partial_output_path.exists():
                raise ValueError(
                    f"Missing partial output for completed checkpoint {task.asset_id} lbw={task.lbw}"
                )
            task.partial_output_path.replace(task.output_path)
            _remove_if_exists(task.checkpoint_path)
            if progress_path is not None:
                finished_at = _utcnow_text()
                _update_progress_row(
                    progress_path,
                    task,
                    project_root=project_root_path,
                    state=PROGRESS_STATE_COMPLETED,
                    rows_written=progress.rows_written,
                    last_timestamp=progress.last_timestamp,
                    retry_count=progress.retry_count,
                    fallback_count=progress.fallback_count,
                    started_at=finished_at,
                    finished_at=finished_at,
                    error_message=None,
                )
            return task.output_path

        started_at = _utcnow_text()
        if progress_path is not None:
            _update_progress_row(
                progress_path,
                task,
                project_root=project_root_path,
                state=PROGRESS_STATE_RUNNING,
                rows_written=progress.rows_written,
                last_timestamp=progress.last_timestamp,
                retry_count=progress.retry_count,
                fallback_count=progress.fallback_count,
                started_at=started_at,
                finished_at=None,
                error_message=None,
            )

        previous_outputs = progress.previous_outputs
        previous_output_timestamp = progress.previous_output_timestamp
        rows_written = progress.rows_written
        last_timestamp = progress.last_timestamp
        retry_count = progress.retry_count
        fallback_count = progress.fallback_count
        buffer: list[dict[str, str]] = []

        writer, handle = _prepare_partial_writer(
            task,
            use_append=resume and task.partial_output_path.exists(),
        )

        def flush_buffer() -> None:
            nonlocal rows_written, last_timestamp, retry_count, fallback_count, buffer
            if not buffer:
                return
            writer.writerows(buffer)
            _fsync_handle(handle)
            rows_written += len(buffer)
            last_timestamp = buffer[-1]["timestamp"]
            retry_count += sum(1 for row in buffer if row["retry_used"] == "true")
            fallback_count += sum(1 for row in buffer if row["fallback_used"] == "true")
            if progress_path is not None:
                _update_progress_row(
                    progress_path,
                    task,
                    project_root=project_root_path,
                    state=PROGRESS_STATE_RUNNING,
                    rows_written=rows_written,
                    last_timestamp=last_timestamp,
                    retry_count=retry_count,
                    fallback_count=fallback_count,
                )
            buffer = []

        try:
            for end_index in range(progress.rows_written, len(returns_rows)):
                output_row, previous_outputs, previous_output_timestamp = compute_cpd_feature_row(
                    returns_rows,
                    end_index=end_index,
                    lbw=task.lbw,
                    previous_outputs=previous_outputs,
                    previous_output_timestamp=previous_output_timestamp,
                    fit_window_fn=fit_window_fn,
                )
                buffer.append(output_row)
                pending_rows_written = rows_written + len(buffer)
                if len(buffer) >= flush_rows:
                    flush_buffer()
                if stop_after_rows is not None and pending_rows_written >= stop_after_rows:
                    flush_buffer()
                    raise T14ChainStopRequested(
                        f"Stopped after {pending_rows_written} rows for "
                        f"{task.asset_id} lbw={task.lbw}"
                    )
            flush_buffer()
        except T14ChainStopRequested:
            handle.close()
            raise
        except Exception as exc:
            try:
                flush_buffer()
            finally:
                handle.close()
            if progress_path is not None:
                _update_progress_row(
                    progress_path,
                    task,
                    project_root=project_root_path,
                    state=PROGRESS_STATE_FAILED,
                    rows_written=rows_written,
                    last_timestamp=last_timestamp,
                    retry_count=retry_count,
                    fallback_count=fallback_count,
                    finished_at=_utcnow_text(),
                    error_message=str(exc),
                )
            raise
        else:
            handle.close()

        completed_summary = _load_partial_chain_progress(
            task,
            returns_rows=returns_rows,
        )
        if completed_summary.rows_written != len(returns_rows):
            raise ValueError(
                f"Partial output remained incomplete for {task.asset_id} lbw={task.lbw}: "
                f"{completed_summary.rows_written} != {len(returns_rows)}"
            )
        task.partial_output_path.replace(task.output_path)
        _remove_if_exists(task.checkpoint_path)
        if progress_path is not None:
            _update_progress_row(
                progress_path,
                task,
                project_root=project_root_path,
                state=PROGRESS_STATE_COMPLETED,
                rows_written=completed_summary.rows_written,
                last_timestamp=completed_summary.last_timestamp,
                retry_count=completed_summary.retry_count,
                fallback_count=completed_summary.fallback_count,
                finished_at=_utcnow_text(),
                error_message=None,
            )
        return task.output_path
    finally:
        del canonical_rows
        del returns_rows
        _cleanup_job_memory()


def _deterministic_worker_env() -> dict[str, str]:
    env = dict(DETERMINISTIC_WORKER_ENV)
    env.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "lstm_cpd_mplconfig"),
    )
    return env


def _set_parent_worker_env() -> dict[str, str | None]:
    env_updates = _deterministic_worker_env()
    previous_values = {key: os.environ.get(key) for key in env_updates}
    for key, value in env_updates.items():
        os.environ[key] = value
    Path(env_updates["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return previous_values


def _restore_parent_worker_env(previous_values: dict[str, str | None]) -> None:
    for key, previous_value in previous_values.items():
        if previous_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous_value


def _serialize_t14_chain_task(task: T14ChainTask) -> str:
    payload = {
        "manifest_record": {
            "asset_id": task.manifest_record.asset_id,
            "symbol": task.manifest_record.symbol,
            "category": task.manifest_record.category,
            "path_pattern": task.manifest_record.path_pattern,
            "source_d_file_path": task.manifest_record.source_d_file_path,
            "canonical_csv_path": task.manifest_record.canonical_csv_path,
            "row_count": task.manifest_record.row_count,
            "first_timestamp": task.manifest_record.first_timestamp,
            "last_timestamp": task.manifest_record.last_timestamp,
            "file_hash": task.manifest_record.file_hash,
        },
        "lbw": task.lbw,
        "canonical_csv_path": str(task.canonical_csv_path),
        "returns_csv_path": str(task.returns_csv_path),
        "output_path": str(task.output_path),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _deserialize_t14_chain_task(payload: str) -> T14ChainTask:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Worker task payload must be an object")
    manifest_record_data = data.get("manifest_record")
    if not isinstance(manifest_record_data, dict):
        raise ValueError("Worker task manifest_record must be an object")
    manifest_record = CanonicalDailyCloseManifestRecord(
        asset_id=str(manifest_record_data["asset_id"]),
        symbol=str(manifest_record_data["symbol"]),
        category=str(manifest_record_data["category"]),
        path_pattern=str(manifest_record_data["path_pattern"]),
        source_d_file_path=str(manifest_record_data["source_d_file_path"]),
        canonical_csv_path=str(manifest_record_data["canonical_csv_path"]),
        row_count=int(manifest_record_data["row_count"]),
        first_timestamp=str(manifest_record_data["first_timestamp"]),
        last_timestamp=str(manifest_record_data["last_timestamp"]),
        file_hash=str(manifest_record_data["file_hash"]),
    )
    return T14ChainTask(
        manifest_record=manifest_record,
        lbw=int(data["lbw"]),
        canonical_csv_path=Path(str(data["canonical_csv_path"])),
        returns_csv_path=Path(str(data["returns_csv_path"])),
        output_path=Path(str(data["output_path"])),
    )


def _apply_deterministic_worker_runtime() -> None:
    env_updates = _deterministic_worker_env()
    for key, value in env_updates.items():
        os.environ.setdefault(key, value)
    Path(env_updates["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    except Exception:
        pass


def _worker_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(_deterministic_worker_env())
    src_root = Path(__file__).resolve().parents[2]
    pythonpath_parts = [str(src_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _worker_command(
    task: T14ChainTask,
    *,
    resume: bool,
    progress_report_path: Path,
    project_root: Path,
    flush_rows: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lstm_cpd.cpd.precompute",
        "--worker-task-json",
        _serialize_t14_chain_task(task),
        "--worker-progress-report-path",
        str(progress_report_path),
        "--worker-project-root",
        str(project_root),
        "--worker-flush-rows",
        str(flush_rows),
    ]
    if resume:
        command.append("--worker-resume")
    return command


def _run_t14_chain_task_worker_subprocess(
    task: T14ChainTask,
    *,
    resume: bool,
    progress_report_path: Path,
    project_root: Path,
    flush_rows: int,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        _worker_command(
            task,
            resume=resume,
            progress_report_path=progress_report_path,
            project_root=project_root,
            flush_rows=flush_rows,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def _collect_subprocess_result(
    task: T14ChainTask,
    process: subprocess.Popen[str],
) -> Path:
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"T-14 worker failed for {task.asset_id} lbw={task.lbw} "
            f"with exit code {process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    output_text = stdout.strip()
    if not output_text:
        raise RuntimeError(
            f"T-14 worker returned no output path for {task.asset_id} lbw={task.lbw}"
        )
    return Path(output_text.splitlines()[-1])


def build_t14_outputs_parallel(
    canonical_manifest_input: Path | str = default_canonical_manifest_input(),
    returns_input_dir: Path | str = default_returns_input_dir(),
    output_dir: Path | str = default_output_dir(),
    *,
    project_root: Path | str | None = None,
    lbws: Sequence[int] = ALLOWED_CPD_LBWS,
    asset_ids: Sequence[str] | None = None,
    max_workers: int | None = None,
    resume: bool = True,
    progress_report_path: Path | str | None = None,
    flush_rows: int = DEFAULT_FLUSH_ROWS,
) -> list[Path]:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    progress_path = (
        Path(progress_report_path)
        if progress_report_path is not None
        else default_progress_report_path(project_root_path)
    )
    tasks = build_t14_task_specs(
        canonical_manifest_input=canonical_manifest_input,
        returns_input_dir=returns_input_dir,
        output_dir=output_dir,
        project_root=project_root_path,
        lbws=lbws,
        asset_ids=asset_ids,
    )
    initialize_progress_report(
        tasks,
        progress_report_path=progress_path,
        project_root=project_root_path,
    )

    requested_workers = default_parallel_workers() if max_workers is None else max_workers
    effective_workers = clamp_parallel_workers(requested_workers)

    previous_env = _set_parent_worker_env()
    try:
        _apply_deterministic_worker_runtime()
        if effective_workers == 1:
            output_paths = [
                run_t14_chain_task(
                    task,
                    resume=resume,
                    skip_if_complete=resume,
                    progress_report_path=progress_path,
                    project_root=project_root_path,
                    flush_rows=flush_rows,
                )
                for task in tasks
            ]
        else:
            worker_env = _worker_subprocess_env()
            pending_tasks = list(tasks)
            active_processes: list[tuple[T14ChainTask, subprocess.Popen[str]]] = []
            completed_paths: dict[tuple[str, int], Path] = {}
            while pending_tasks or active_processes:
                while pending_tasks and len(active_processes) < effective_workers:
                    task = pending_tasks.pop(0)
                    active_processes.append(
                        (
                            task,
                            _run_t14_chain_task_worker_subprocess(
                                task,
                                resume=resume,
                                progress_report_path=progress_path,
                                project_root=project_root_path,
                                flush_rows=flush_rows,
                                env=worker_env,
                            ),
                        )
                    )

                next_active: list[tuple[T14ChainTask, subprocess.Popen[str]]] = []
                made_progress = False
                for task, process in active_processes:
                    if process.poll() is None:
                        next_active.append((task, process))
                        continue
                    completed_paths[(task.asset_id, task.lbw)] = _collect_subprocess_result(
                        task,
                        process,
                    )
                    made_progress = True
                active_processes = next_active
                if not made_progress and active_processes:
                    time.sleep(0.05)
            output_paths = [completed_paths[(task.asset_id, task.lbw)] for task in tasks]
    finally:
        _restore_parent_worker_env(previous_env)

    for task in tasks:
        if not task.output_path.exists():
            raise ValueError(f"Missing CPD output for {task.asset_id} lbw={task.lbw}")
    return output_paths


def build_t14_outputs(
    canonical_manifest_input: Path | str = default_canonical_manifest_input(),
    returns_input_dir: Path | str = default_returns_input_dir(),
    output_dir: Path | str = default_output_dir(),
    *,
    project_root: Path | str | None = None,
    lbws: Sequence[int] = ALLOWED_CPD_LBWS,
    asset_ids: Sequence[str] | None = None,
    fit_window_fn: Callable[[CPDWindowInput], CPDWindowResult] = fit_cpd_window,
    resume: bool = False,
    skip_if_complete: bool = False,
    progress_report_path: Path | str | None = None,
    flush_rows: int = DEFAULT_FLUSH_ROWS,
) -> list[Path]:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    progress_path = (
        Path(progress_report_path)
        if progress_report_path is not None
        else default_progress_report_path(project_root_path)
    )
    tasks = build_t14_task_specs(
        canonical_manifest_input=canonical_manifest_input,
        returns_input_dir=returns_input_dir,
        output_dir=output_dir,
        project_root=project_root_path,
        lbws=lbws,
        asset_ids=asset_ids,
    )
    initialize_progress_report(
        tasks,
        progress_report_path=progress_path,
        project_root=project_root_path,
    )

    previous_env = _set_parent_worker_env()
    try:
        _apply_deterministic_worker_runtime()
        output_paths = [
            run_t14_chain_task(
                task,
                fit_window_fn=fit_window_fn,
                resume=resume,
                skip_if_complete=skip_if_complete,
                progress_report_path=progress_path,
                project_root=project_root_path,
                flush_rows=flush_rows,
            )
            for task in tasks
        ]
    finally:
        _restore_parent_worker_env(previous_env)

    for task in tasks:
        if not task.output_path.exists():
            raise ValueError(f"Missing CPD output for {task.asset_id} lbw={task.lbw}")
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute per-LBW CPD feature stores."
    )
    parser.add_argument(
        "--canonical-manifest-input",
        type=Path,
        default=default_canonical_manifest_input(),
    )
    parser.add_argument(
        "--returns-input-dir",
        type=Path,
        default=default_returns_input_dir(),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument(
        "--asset-id",
        dest="asset_ids",
        action="append",
        default=None,
        help="Restrict T-14 to one or more asset_ids.",
    )
    parser.add_argument(
        "--lbw",
        dest="lbws",
        action="append",
        type=int,
        default=None,
        help="Restrict T-14 to one or more LBWs.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from partial CSVs and skip valid completed final outputs.",
    )
    parser.add_argument(
        "--progress-report-path",
        type=Path,
        default=None,
        help="Optional override for the durable T-14 progress report.",
    )
    parser.add_argument(
        "--flush-rows",
        type=int,
        default=DEFAULT_FLUSH_ROWS,
        help="Number of rows to buffer before fsync.",
    )
    parser.add_argument(
        "--worker-task-json",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-resume",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-progress-report-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-project-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-flush-rows",
        type=int,
        default=DEFAULT_FLUSH_ROWS,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_task_json is not None:
        task = _deserialize_t14_chain_task(args.worker_task_json)
        _apply_deterministic_worker_runtime()
        output_path = run_t14_chain_task(
            task,
            resume=args.worker_resume,
            skip_if_complete=args.worker_resume,
            progress_report_path=args.worker_progress_report_path,
            project_root=args.worker_project_root,
            flush_rows=args.worker_flush_rows,
        )
        print(str(output_path))
        return 0

    output_paths = build_t14_outputs_parallel(
        canonical_manifest_input=args.canonical_manifest_input,
        returns_input_dir=args.returns_input_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
        lbws=ALLOWED_CPD_LBWS if args.lbws is None else args.lbws,
        asset_ids=args.asset_ids,
        max_workers=args.workers,
        resume=args.resume,
        progress_report_path=args.progress_report_path,
        flush_rows=args.flush_rows,
    )
    print(f"Wrote {len(output_paths)} CPD feature files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
