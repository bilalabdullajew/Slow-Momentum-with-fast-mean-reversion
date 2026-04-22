from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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
CHECKPOINT_VERSION = 1

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
        return self.output_path.with_name(
            f"{self.output_path.stem}{CPD_PARTIAL_SUFFIX}"
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.output_path.with_name(
            f"{self.output_path.name}{CPD_CHECKPOINT_SUFFIX}"
        )


@dataclass(frozen=True)
class T14ChainCheckpoint:
    version: int
    asset_id: str
    lbw: int
    output_path: str
    partial_output_path: str
    rows_completed: int
    last_completed_timestamp: str | None
    previous_output_timestamp: str | None
    previous_nu: float | None
    previous_gamma: float | None


@dataclass(frozen=True)
class T14ChainProgress:
    rows_completed: int
    previous_outputs: CPDPreviousOutputs | None
    previous_output_timestamp: str | None


class T14ChainStopRequested(RuntimeError):
    pass


def default_canonical_manifest_input() -> Path:
    return default_project_root() / "artifacts/manifests/canonical_daily_close_manifest.json"


def default_returns_input_dir() -> Path:
    return default_project_root() / "artifacts/features/base"


def default_output_dir() -> Path:
    return default_project_root() / "artifacts/features/cpd"


def default_parallel_workers() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 3:
        return 1
    return max(1, min(7, cpu_count - 3))


def project_relative_path(path: Path | str, project_root: Path | str) -> str:
    return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()


def _serialize_optional_text(value: str | None) -> str:
    return "" if value is None else value


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
) -> list[T14ChainTask]:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    output_dir_path = Path(output_dir)
    requested_lbws = _validate_requested_lbws(lbws)
    manifest_records = load_canonical_daily_close_manifest(canonical_manifest_input)
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


def _build_chain_checkpoint(
    task: T14ChainTask,
    *,
    rows_completed: int,
    last_completed_timestamp: str | None,
    previous_outputs: CPDPreviousOutputs | None,
    previous_output_timestamp: str | None,
) -> T14ChainCheckpoint:
    return T14ChainCheckpoint(
        version=CHECKPOINT_VERSION,
        asset_id=task.asset_id,
        lbw=task.lbw,
        output_path=str(task.output_path),
        partial_output_path=str(task.partial_output_path),
        rows_completed=rows_completed,
        last_completed_timestamp=last_completed_timestamp,
        previous_output_timestamp=previous_output_timestamp,
        previous_nu=None if previous_outputs is None else previous_outputs.nu,
        previous_gamma=None if previous_outputs is None else previous_outputs.gamma,
    )


def _write_chain_checkpoint(
    checkpoint_path: Path | str,
    checkpoint: T14ChainCheckpoint,
) -> None:
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": checkpoint.version,
        "asset_id": checkpoint.asset_id,
        "lbw": checkpoint.lbw,
        "output_path": checkpoint.output_path,
        "partial_output_path": checkpoint.partial_output_path,
        "rows_completed": checkpoint.rows_completed,
        "last_completed_timestamp": checkpoint.last_completed_timestamp,
        "previous_output_timestamp": checkpoint.previous_output_timestamp,
        "previous_nu": checkpoint.previous_nu,
        "previous_gamma": checkpoint.previous_gamma,
    }
    temp_path = checkpoint_file.with_name(f"{checkpoint_file.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(checkpoint_file)


def _load_chain_checkpoint(
    task: T14ChainTask,
) -> T14ChainCheckpoint | None:
    if not task.checkpoint_path.exists():
        return None
    payload = json.loads(task.checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be an object: {task.checkpoint_path}")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {task.checkpoint_path}")
    asset_id = str(payload.get("asset_id", ""))
    lbw = int(payload.get("lbw", -1))
    if asset_id != task.asset_id or lbw != task.lbw:
        raise ValueError(f"Checkpoint task mismatch: {task.checkpoint_path}")
    if str(payload.get("output_path", "")) != str(task.output_path):
        raise ValueError(f"Checkpoint output path mismatch: {task.checkpoint_path}")
    if str(payload.get("partial_output_path", "")) != str(task.partial_output_path):
        raise ValueError(f"Checkpoint partial output path mismatch: {task.checkpoint_path}")
    rows_completed = int(payload.get("rows_completed", -1))
    if rows_completed < 0:
        raise ValueError(f"Checkpoint rows_completed must be non-negative: {task.checkpoint_path}")
    previous_nu = payload.get("previous_nu")
    previous_gamma = payload.get("previous_gamma")
    if previous_nu is not None and not math.isfinite(float(previous_nu)):
        raise ValueError(f"Checkpoint previous_nu must be finite: {task.checkpoint_path}")
    if previous_gamma is not None and not math.isfinite(float(previous_gamma)):
        raise ValueError(f"Checkpoint previous_gamma must be finite: {task.checkpoint_path}")
    return T14ChainCheckpoint(
        version=CHECKPOINT_VERSION,
        asset_id=asset_id,
        lbw=lbw,
        output_path=str(payload["output_path"]),
        partial_output_path=str(payload["partial_output_path"]),
        rows_completed=rows_completed,
        last_completed_timestamp=(
            None if payload.get("last_completed_timestamp") is None else str(payload["last_completed_timestamp"])
        ),
        previous_output_timestamp=(
            None if payload.get("previous_output_timestamp") is None else str(payload["previous_output_timestamp"])
        ),
        previous_nu=None if previous_nu is None else float(previous_nu),
        previous_gamma=None if previous_gamma is None else float(previous_gamma),
    )


def _load_partial_chain_progress(task: T14ChainTask) -> T14ChainProgress:
    checkpoint = _load_chain_checkpoint(task)
    if not task.partial_output_path.exists():
        if checkpoint is not None and checkpoint.rows_completed != 0:
            raise ValueError(
                f"Checkpoint exists without partial output for {task.asset_id} lbw={task.lbw}"
            )
        return T14ChainProgress(
            rows_completed=0,
            previous_outputs=None,
            previous_output_timestamp=None,
        )

    partial_rows = load_cpd_feature_csv(
        task.partial_output_path,
        expected_asset_id=task.asset_id,
        expected_lbw=task.lbw,
    )
    if partial_rows and partial_rows[-1].has_outputs:
        previous_outputs = CPDPreviousOutputs(
            nu=partial_rows[-1].nu,
            gamma=partial_rows[-1].gamma,
        )
        previous_output_timestamp = partial_rows[-1].timestamp
    else:
        previous_outputs = None
        previous_output_timestamp = None

    rows_completed = len(partial_rows)
    last_completed_timestamp = partial_rows[-1].timestamp if partial_rows else None
    if checkpoint is not None:
        if checkpoint.rows_completed < 0:
            raise ValueError(f"Invalid checkpoint rows_completed for {task.asset_id} lbw={task.lbw}")
        # The partial file is authoritative on resume because it is the durable row prefix.
        if checkpoint.rows_completed == rows_completed:
            if checkpoint.last_completed_timestamp != last_completed_timestamp:
                raise ValueError(
                    f"Checkpoint timestamp mismatch for {task.asset_id} lbw={task.lbw}"
                )
    return T14ChainProgress(
        rows_completed=rows_completed,
        previous_outputs=previous_outputs,
        previous_output_timestamp=previous_output_timestamp,
    )


def _remove_if_exists(path: Path | str) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()


def _prepare_partial_writer(
    task: T14ChainTask,
    *,
    resume: bool,
) -> tuple[csv.DictWriter, object]:
    partial_exists = task.partial_output_path.exists()
    use_append = resume and partial_exists
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
    return writer, handle


def _validate_completed_output(
    task: T14ChainTask,
    *,
    expected_row_count: int,
) -> None:
    completed_rows = load_cpd_feature_csv(
        task.output_path,
        expected_asset_id=task.asset_id,
        expected_lbw=task.lbw,
    )
    if len(completed_rows) != expected_row_count:
        raise ValueError(
            f"Completed CPD output row count mismatch for {task.asset_id} lbw={task.lbw}: "
            f"{len(completed_rows)} != {expected_row_count}"
        )


def run_t14_chain_task(
    task: T14ChainTask,
    *,
    fit_window_fn: Callable[[CPDWindowInput], CPDWindowResult] = fit_cpd_window,
    resume: bool = False,
    skip_if_complete: bool = False,
    stop_after_rows: int | None = None,
) -> Path:
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

    if stop_after_rows is not None and stop_after_rows <= 0:
        raise ValueError(f"stop_after_rows must be positive: {stop_after_rows}")

    if task.output_path.exists():
        if skip_if_complete:
            _validate_completed_output(
                task,
                expected_row_count=len(returns_rows),
            )
            return task.output_path
        _remove_if_exists(task.output_path)

    if not resume:
        _remove_if_exists(task.partial_output_path)
        _remove_if_exists(task.checkpoint_path)
        progress = T14ChainProgress(
            rows_completed=0,
            previous_outputs=None,
            previous_output_timestamp=None,
        )
    else:
        progress = _load_partial_chain_progress(task)

    if progress.rows_completed > len(returns_rows):
        raise ValueError(
            f"Partial progress exceeds available rows for {task.asset_id} lbw={task.lbw}: "
            f"{progress.rows_completed} > {len(returns_rows)}"
        )

    if stop_after_rows is not None and progress.rows_completed >= stop_after_rows:
        raise ValueError(
            f"stop_after_rows={stop_after_rows} must exceed existing progress "
            f"{progress.rows_completed} for {task.asset_id} lbw={task.lbw}"
        )

    if progress.rows_completed == len(returns_rows):
        if not task.partial_output_path.exists():
            raise ValueError(
                f"Missing partial output for completed checkpoint {task.asset_id} lbw={task.lbw}"
            )
        task.partial_output_path.replace(task.output_path)
        _remove_if_exists(task.checkpoint_path)
        return task.output_path

    previous_outputs = progress.previous_outputs
    previous_output_timestamp = progress.previous_output_timestamp
    writer, handle = _prepare_partial_writer(task, resume=resume)
    try:
        for end_index in range(progress.rows_completed, len(returns_rows)):
            output_row, previous_outputs, previous_output_timestamp = compute_cpd_feature_row(
                returns_rows,
                end_index=end_index,
                lbw=task.lbw,
                previous_outputs=previous_outputs,
                previous_output_timestamp=previous_output_timestamp,
                fit_window_fn=fit_window_fn,
            )
            writer.writerow(output_row)
            rows_completed = end_index + 1
            _write_chain_checkpoint(
                task.checkpoint_path,
                _build_chain_checkpoint(
                    task,
                    rows_completed=rows_completed,
                    last_completed_timestamp=returns_rows[end_index].timestamp,
                    previous_outputs=previous_outputs,
                    previous_output_timestamp=previous_output_timestamp,
                ),
            )
            if stop_after_rows is not None and rows_completed >= stop_after_rows:
                raise T14ChainStopRequested(
                    f"Stopped after {rows_completed} rows for {task.asset_id} lbw={task.lbw}"
                )
    finally:
        handle.close()

    task.partial_output_path.replace(task.output_path)
    _remove_if_exists(task.checkpoint_path)
    return task.output_path


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
        # Thread pools can no longer be mutated if the runtime is already initialized.
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


def _worker_command(task: T14ChainTask, *, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lstm_cpd.cpd.precompute",
        "--worker-task-json",
        _serialize_t14_chain_task(task),
    ]
    if resume:
        command.append("--worker-resume")
    return command


def _run_t14_chain_task_worker_subprocess(
    task: T14ChainTask,
    *,
    resume: bool,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        _worker_command(task, resume=resume),
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
    max_workers: int | None = None,
    resume: bool = True,
) -> list[Path]:
    tasks = build_t14_task_specs(
        canonical_manifest_input=canonical_manifest_input,
        returns_input_dir=returns_input_dir,
        output_dir=output_dir,
        project_root=project_root,
        lbws=lbws,
    )
    if max_workers is None:
        max_workers = default_parallel_workers()
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive: {max_workers}")

    if max_workers == 1:
        output_paths = [
            run_t14_chain_task(
                task,
                resume=resume,
                skip_if_complete=resume,
            )
            for task in tasks
        ]
    else:
        previous_env = _set_parent_worker_env()
        try:
            worker_env = _worker_subprocess_env()
            pending_tasks = list(tasks)
            active_processes: list[tuple[T14ChainTask, subprocess.Popen[str]]] = []
            completed_paths: dict[tuple[str, int], Path] = {}
            while pending_tasks or active_processes:
                while pending_tasks and len(active_processes) < max_workers:
                    task = pending_tasks.pop(0)
                    active_processes.append(
                        (
                            task,
                            _run_t14_chain_task_worker_subprocess(
                                task,
                                resume=resume,
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
            output_paths = [
                completed_paths[(task.asset_id, task.lbw)]
                for task in tasks
            ]
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
    fit_window_fn: Callable[[CPDWindowInput], CPDWindowResult] = fit_cpd_window,
) -> list[Path]:
    tasks = build_t14_task_specs(
        canonical_manifest_input=canonical_manifest_input,
        returns_input_dir=returns_input_dir,
        output_dir=output_dir,
        project_root=project_root,
        lbws=lbws,
    )
    output_paths = [
        run_t14_chain_task(
            task,
            fit_window_fn=fit_window_fn,
            resume=False,
            skip_if_complete=False,
        )
        for task in tasks
    ]
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume from per-chain checkpoints and skip completed final outputs.",
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
        )
        print(str(output_path))
        return 0
    if args.workers > 1 or args.resume:
        output_paths = build_t14_outputs_parallel(
            canonical_manifest_input=args.canonical_manifest_input,
            returns_input_dir=args.returns_input_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
            max_workers=args.workers,
            resume=args.resume,
        )
    else:
        output_paths = build_t14_outputs(
            canonical_manifest_input=args.canonical_manifest_input,
            returns_input_dir=args.returns_input_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    print(f"Wrote {len(output_paths)} CPD feature files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
