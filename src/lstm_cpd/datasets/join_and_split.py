from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lstm_cpd.cpd.precompute_contract import (
    ALLOWED_CPD_LBWS,
    CPD_RESULT_STATUSES,
    STATUS_FALLBACK_PREVIOUS,
    STATUS_RETRY_SUCCESS,
    STATUS_SUCCESS,
    is_allowed_lbw,
)
from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.features.returns import RETURNS_VOLATILITY_HEADER, serialize_optional_float
from lstm_cpd.features.winsorize import BASE_FEATURES_SUFFIX, FEATURE_COLUMNS, T12_OUTPUT_HEADER


RETURNS_VOLATILITY_SUFFIX = "_returns_volatility.csv"
CPD_STATE_PRESENT = "present"
SPLIT_RULE_FLOOR_90_REST_10 = "floor_90_rest_10"

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

MODEL_INPUT_COLUMNS = FEATURE_COLUMNS + ("nu", "gamma")
T16_OUTPUT_HEADER = (
    "timestamp",
    "asset_id",
    "lbw",
    "timeline_index",
    "sigma_t",
    "next_arithmetic_return",
) + MODEL_INPUT_COLUMNS
T16_SPLIT_MANIFEST_HEADER = (
    "asset_id",
    "lbw",
    "usable_row_count",
    "train_row_count",
    "val_row_count",
    "train_start_timestamp",
    "train_end_timestamp",
    "val_start_timestamp",
    "val_end_timestamp",
    "train_start_timeline_index",
    "train_end_timeline_index",
    "val_start_timeline_index",
    "val_end_timeline_index",
    "split_rule",
)


@dataclass(frozen=True)
class CPDFeatureStoreManifestRecord:
    asset_id: str
    lbw: int
    state: str
    missing_reason: str | None
    cpd_csv_path: str
    row_count: int
    canonical_row_count: int
    first_timestamp: str
    last_timestamp: str
    output_row_count: int
    retry_used_count: int
    fallback_used_count: int
    status_counts: dict[str, int]
    matches_canonical_timeline: bool
    file_hash: str | None


@dataclass(frozen=True)
class ReturnsVolatilityJoinRecord:
    timestamp: str
    asset_id: str
    arithmetic_return: float | None
    sigma_t: float | None


@dataclass(frozen=True)
class CPDJoinRecord:
    timestamp: str
    asset_id: str
    lbw: int
    nu: float | None
    gamma: float | None
    status: str

    @property
    def has_outputs(self) -> bool:
        return self.nu is not None and self.gamma is not None


@dataclass(frozen=True)
class T16JoinedFeatureRow:
    timestamp: str
    asset_id: str
    lbw: int
    timeline_index: int
    sigma_t: float
    next_arithmetic_return: float | None
    model_inputs: tuple[float, ...]

    def to_csv_row(self) -> dict[str, str]:
        row = {
            "timestamp": self.timestamp,
            "asset_id": self.asset_id,
            "lbw": str(self.lbw),
            "timeline_index": str(self.timeline_index),
            "sigma_t": serialize_optional_float(self.sigma_t),
            "next_arithmetic_return": serialize_optional_float(self.next_arithmetic_return),
        }
        for column_name, value in zip(MODEL_INPUT_COLUMNS, self.model_inputs):
            row[column_name] = serialize_optional_float(value)
        return row


@dataclass(frozen=True)
class T16SplitManifestRow:
    asset_id: str
    lbw: int
    usable_row_count: int
    train_row_count: int
    val_row_count: int
    train_start_timestamp: str | None
    train_end_timestamp: str | None
    val_start_timestamp: str | None
    val_end_timestamp: str | None
    train_start_timeline_index: int | None
    train_end_timeline_index: int | None
    val_start_timeline_index: int | None
    val_end_timeline_index: int | None
    split_rule: str = SPLIT_RULE_FLOOR_90_REST_10

    def to_csv_row(self) -> dict[str, str]:
        return {
            "asset_id": self.asset_id,
            "lbw": str(self.lbw),
            "usable_row_count": str(self.usable_row_count),
            "train_row_count": str(self.train_row_count),
            "val_row_count": str(self.val_row_count),
            "train_start_timestamp": self.train_start_timestamp or "",
            "train_end_timestamp": self.train_end_timestamp or "",
            "val_start_timestamp": self.val_start_timestamp or "",
            "val_end_timestamp": self.val_end_timestamp or "",
            "train_start_timeline_index": _serialize_optional_int(
                self.train_start_timeline_index
            ),
            "train_end_timeline_index": _serialize_optional_int(
                self.train_end_timeline_index
            ),
            "val_start_timeline_index": _serialize_optional_int(
                self.val_start_timeline_index
            ),
            "val_end_timeline_index": _serialize_optional_int(
                self.val_end_timeline_index
            ),
            "split_rule": self.split_rule,
        }


@dataclass(frozen=True)
class T16OutputArtifacts:
    joined_feature_paths: list[Path]
    split_manifest_paths: list[Path]


def default_base_input_dir() -> Path:
    return default_project_root() / "artifacts/features/base"


def default_returns_input_dir() -> Path:
    return default_project_root() / "artifacts/features/base"


def default_cpd_manifest_input() -> Path:
    return default_project_root() / "artifacts/manifests/cpd_feature_store_manifest.json"


def default_output_dir() -> Path:
    return default_project_root() / "artifacts/datasets"


def project_relative_path(path: Path | str, project_root: Path | str) -> str:
    resolved_path = Path(path).resolve()
    resolved_root = Path(project_root).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _serialize_optional_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


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


def _parse_required_float_text(
    text: str,
    *,
    csv_path: Path,
    row_index: int,
    column_name: str,
) -> float:
    value = _parse_optional_float_text(
        text,
        csv_path=csv_path,
        row_index=row_index,
        column_name=column_name,
    )
    if value is None:
        raise ValueError(
            f"Row {row_index} column {column_name} must not be blank: {csv_path}"
        )
    return value


def _parse_optional_int_text(text: str) -> int | None:
    if text == "":
        return None
    return int(text)


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


def _ordered_asset_ids(
    manifest_records: Sequence[CPDFeatureStoreManifestRecord],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(record.asset_id for record in manifest_records))


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


def load_cpd_feature_store_manifest(
    path: Path | str,
) -> list[CPDFeatureStoreManifestRecord]:
    manifest_path = Path(path)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("CPD feature-store manifest must be a JSON array")

    required_fields = {
        "asset_id",
        "lbw",
        "state",
        "missing_reason",
        "cpd_csv_path",
        "row_count",
        "canonical_row_count",
        "first_timestamp",
        "last_timestamp",
        "output_row_count",
        "retry_used_count",
        "fallback_used_count",
        "status_counts",
        "matches_canonical_timeline",
        "file_hash",
    }
    records: list[CPDFeatureStoreManifestRecord] = []
    seen_keys: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"CPD feature-store manifest row {index} is not an object")
        missing_fields = required_fields - set(row)
        if missing_fields:
            raise ValueError(
                f"CPD feature-store manifest row {index} missing fields: {sorted(missing_fields)}"
            )
        asset_id = str(row["asset_id"])
        lbw = int(row["lbw"])
        key = (asset_id, lbw)
        if key in seen_keys:
            raise ValueError(
                f"CPD feature-store manifest has duplicate asset_id/lbw pair: {key}"
            )
        seen_keys.add(key)
        status_counts_raw = row["status_counts"]
        if not isinstance(status_counts_raw, dict):
            raise ValueError(
                f"CPD feature-store manifest row {index} has invalid status_counts"
            )
        status_counts = {
            status: int(status_counts_raw.get(status, 0))
            for status in CPD_RESULT_STATUSES
        }
        records.append(
            CPDFeatureStoreManifestRecord(
                asset_id=asset_id,
                lbw=lbw,
                state=str(row["state"]),
                missing_reason=(
                    None if row["missing_reason"] in (None, "") else str(row["missing_reason"])
                ),
                cpd_csv_path=str(row["cpd_csv_path"]),
                row_count=int(row["row_count"]),
                canonical_row_count=int(row["canonical_row_count"]),
                first_timestamp=str(row["first_timestamp"]),
                last_timestamp=str(row["last_timestamp"]),
                output_row_count=int(row["output_row_count"]),
                retry_used_count=int(row["retry_used_count"]),
                fallback_used_count=int(row["fallback_used_count"]),
                status_counts=status_counts,
                matches_canonical_timeline=bool(row["matches_canonical_timeline"]),
                file_hash=None if row["file_hash"] in (None, "") else str(row["file_hash"]),
            )
        )
    return records


def _select_manifest_records(
    manifest_records: Sequence[CPDFeatureStoreManifestRecord],
    *,
    lbws: Sequence[int],
    asset_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], dict[tuple[str, int], CPDFeatureStoreManifestRecord]]:
    requested_lbws = _validate_requested_lbws(lbws)
    asset_filter = _normalize_asset_filter(asset_ids)
    expected_assets = asset_filter or _ordered_asset_ids(manifest_records)

    manifest_index: dict[tuple[str, int], CPDFeatureStoreManifestRecord] = {}
    for record in manifest_records:
        if record.asset_id not in expected_assets or record.lbw not in requested_lbws:
            continue
        manifest_index[(record.asset_id, record.lbw)] = record

    missing_pairs = [
        (asset_id, lbw)
        for asset_id in expected_assets
        for lbw in requested_lbws
        if (asset_id, lbw) not in manifest_index
    ]
    if missing_pairs:
        raise ValueError(
            "CPD feature-store manifest is missing asset/lbw coverage: "
            f"{missing_pairs}"
        )

    incomplete_records = [
        record
        for record in manifest_index.values()
        if record.state != CPD_STATE_PRESENT
        or record.cpd_csv_path == ""
        or not record.matches_canonical_timeline
        or record.file_hash is None
        or record.row_count != record.canonical_row_count
    ]
    if incomplete_records:
        rendered = [
            (
                record.asset_id,
                record.lbw,
                record.state,
                record.missing_reason,
            )
            for record in sorted(
                incomplete_records,
                key=lambda item: (item.asset_id, item.lbw),
            )
        ]
        raise ValueError(
            "CPD feature-store manifest is incomplete; T-16 requires finalized T-15 outputs: "
            f"{rendered}"
        )
    return expected_assets, manifest_index


def _validate_base_feature_text(
    *,
    csv_path: Path,
    row_index: int,
    column_name: str,
    text: str,
) -> None:
    if text == "":
        return
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(
            f"Row {row_index} column {column_name} has non-finite value: {csv_path}"
        )


def load_base_feature_csv(
    path: Path | str,
    *,
    expected_asset_id: str | None = None,
) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != T12_OUTPUT_HEADER:
            raise ValueError(f"Base-feature header mismatch: {csv_path}")

        rows: list[dict[str, str]] = []
        previous_timestamp: str | None = None
        for row_index, row in enumerate(reader):
            timestamp = row["timestamp"]
            asset_id = row["asset_id"]
            if not timestamp:
                raise ValueError(f"Base-feature row {row_index} missing timestamp: {csv_path}")
            if not asset_id:
                raise ValueError(f"Base-feature row {row_index} missing asset_id: {csv_path}")
            if expected_asset_id is not None and asset_id != expected_asset_id:
                raise ValueError(
                    f"Base-feature row {row_index} asset_id mismatch for {csv_path}: {asset_id}"
                )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(
                    f"Base-feature timestamps must be strictly ascending in {csv_path}: "
                    f"{previous_timestamp} then {timestamp}"
                )
            previous_timestamp = timestamp
            for column_name in FEATURE_COLUMNS:
                _validate_base_feature_text(
                    csv_path=csv_path,
                    row_index=row_index,
                    column_name=column_name,
                    text=row[column_name],
                )
            rows.append(dict(row))
    return rows


def load_returns_volatility_join_csv(
    path: Path | str,
    *,
    expected_asset_id: str | None = None,
) -> list[ReturnsVolatilityJoinRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RETURNS_VOLATILITY_HEADER:
            raise ValueError(f"Returns-volatility header mismatch: {csv_path}")

        rows: list[ReturnsVolatilityJoinRecord] = []
        previous_timestamp: str | None = None
        for row_index, row in enumerate(reader):
            timestamp = row["timestamp"]
            asset_id = row["asset_id"]
            close_text = row["close"]
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
            if close_text == "":
                raise ValueError(
                    f"Returns-volatility row {row_index} missing close: {csv_path}"
                )
            close_value = float(close_text)
            if not math.isfinite(close_value):
                raise ValueError(
                    f"Returns-volatility row {row_index} has non-finite close: {csv_path}"
                )
            rows.append(
                ReturnsVolatilityJoinRecord(
                    timestamp=timestamp,
                    asset_id=asset_id,
                    arithmetic_return=_parse_optional_float_text(
                        row["arithmetic_return"],
                        csv_path=csv_path,
                        row_index=row_index,
                        column_name="arithmetic_return",
                    ),
                    sigma_t=_parse_optional_float_text(
                        row["sigma_t"],
                        csv_path=csv_path,
                        row_index=row_index,
                        column_name="sigma_t",
                    ),
                )
            )
    return rows


def load_cpd_feature_csv(
    path: Path | str,
    *,
    expected_asset_id: str | None = None,
    expected_lbw: int | None = None,
) -> list[CPDJoinRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CPD_OUTPUT_HEADER:
            raise ValueError(f"CPD feature header mismatch: {csv_path}")

        rows: list[CPDJoinRecord] = []
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
                raise ValueError(f"CPD row {row_index} lbw mismatch for {csv_path}: {lbw}")
            if not is_allowed_lbw(lbw):
                raise ValueError(f"CPD row {row_index} has unsupported lbw: {csv_path}")

            status = row["status"]
            if status not in CPD_RESULT_STATUSES:
                raise ValueError(f"CPD row {row_index} has invalid status: {csv_path}")

            nu = _parse_optional_float_text(
                row["nu"],
                csv_path=csv_path,
                row_index=row_index,
                column_name="nu",
            )
            gamma = _parse_optional_float_text(
                row["gamma"],
                csv_path=csv_path,
                row_index=row_index,
                column_name="gamma",
            )
            has_outputs = nu is not None and gamma is not None
            if (nu is None) != (gamma is None):
                raise ValueError(
                    f"CPD row {row_index} must set nu and gamma together: {csv_path}"
                )
            if status in {STATUS_SUCCESS, STATUS_RETRY_SUCCESS, STATUS_FALLBACK_PREVIOUS}:
                if not has_outputs:
                    raise ValueError(
                        f"CPD row {row_index} must contain outputs for status={status}: "
                        f"{csv_path}"
                    )
            elif has_outputs:
                raise ValueError(
                    f"CPD row {row_index} must not contain outputs for status={status}: "
                    f"{csv_path}"
                )
            if has_outputs and not (0.0 <= nu <= 1.0 and 0.0 <= gamma <= 1.0):
                raise ValueError(
                    f"CPD row {row_index} has out-of-range outputs: {csv_path}"
                )
            rows.append(
                CPDJoinRecord(
                    timestamp=timestamp,
                    asset_id=asset_id,
                    lbw=lbw,
                    nu=nu,
                    gamma=gamma,
                    status=status,
                )
            )
    return rows


def _validate_timeline_alignment(
    asset_id: str,
    *,
    base_rows: Sequence[dict[str, str]],
    returns_rows: Sequence[ReturnsVolatilityJoinRecord],
    cpd_rows: Sequence[CPDJoinRecord],
) -> None:
    if len(base_rows) != len(returns_rows):
        raise ValueError(
            f"Base/returns row-count mismatch for {asset_id}: "
            f"{len(base_rows)} != {len(returns_rows)}"
        )
    if len(base_rows) != len(cpd_rows):
        raise ValueError(
            f"Base/CPD row-count mismatch for {asset_id}: "
            f"{len(base_rows)} != {len(cpd_rows)}"
        )
    for row_index, (base_row, returns_row, cpd_row) in enumerate(
        zip(base_rows, returns_rows, cpd_rows)
    ):
        base_key = (base_row["timestamp"], base_row["asset_id"])
        returns_key = (returns_row.timestamp, returns_row.asset_id)
        cpd_key = (cpd_row.timestamp, cpd_row.asset_id)
        if base_key != returns_key:
            raise ValueError(
                f"Base/returns alignment mismatch for {asset_id} at row {row_index}: "
                f"{base_key} != {returns_key}"
            )
        if base_key != cpd_key:
            raise ValueError(
                f"Base/CPD alignment mismatch for {asset_id} at row {row_index}: "
                f"{base_key} != {cpd_key}"
            )


def _has_complete_model_inputs(
    base_row: dict[str, str],
    returns_row: ReturnsVolatilityJoinRecord,
    cpd_row: CPDJoinRecord,
) -> bool:
    if returns_row.sigma_t is None:
        return False
    if any(base_row[column_name] == "" for column_name in FEATURE_COLUMNS):
        return False
    return cpd_row.has_outputs


def build_joined_feature_rows(
    *,
    base_rows: Sequence[dict[str, str]],
    returns_rows: Sequence[ReturnsVolatilityJoinRecord],
    cpd_rows: Sequence[CPDJoinRecord],
    lbw: int,
) -> list[T16JoinedFeatureRow]:
    if not base_rows:
        return []
    asset_id = base_rows[0]["asset_id"]
    _validate_timeline_alignment(
        asset_id,
        base_rows=base_rows,
        returns_rows=returns_rows,
        cpd_rows=cpd_rows,
    )

    joined_rows: list[T16JoinedFeatureRow] = []
    for row_index, (base_row, returns_row, cpd_row) in enumerate(
        zip(base_rows, returns_rows, cpd_rows)
    ):
        if not _has_complete_model_inputs(base_row, returns_row, cpd_row):
            continue
        next_arithmetic_return = None
        if row_index + 1 < len(returns_rows):
            next_arithmetic_return = returns_rows[row_index + 1].arithmetic_return
        model_inputs = tuple(
            _parse_required_float_text(
                base_row[column_name] if column_name in FEATURE_COLUMNS else "",
                csv_path=Path("<joined_row>"),
                row_index=row_index,
                column_name=column_name,
            )
            for column_name in FEATURE_COLUMNS
        ) + (cpd_row.nu, cpd_row.gamma)
        joined_rows.append(
            T16JoinedFeatureRow(
                timestamp=base_row["timestamp"],
                asset_id=asset_id,
                lbw=lbw,
                timeline_index=row_index,
                sigma_t=returns_row.sigma_t,
                next_arithmetic_return=next_arithmetic_return,
                model_inputs=model_inputs,
            )
        )
    return joined_rows


def build_split_manifest_row(
    joined_rows: Sequence[T16JoinedFeatureRow],
    *,
    asset_id: str,
    lbw: int,
    split_rule: str = SPLIT_RULE_FLOOR_90_REST_10,
) -> T16SplitManifestRow:
    usable_row_count = len(joined_rows)
    train_row_count = math.floor(usable_row_count * 0.9)
    val_row_count = usable_row_count - train_row_count

    train_rows = joined_rows[:train_row_count]
    val_rows = joined_rows[train_row_count:]
    return T16SplitManifestRow(
        asset_id=asset_id,
        lbw=lbw,
        usable_row_count=usable_row_count,
        train_row_count=train_row_count,
        val_row_count=val_row_count,
        train_start_timestamp=None if not train_rows else train_rows[0].timestamp,
        train_end_timestamp=None if not train_rows else train_rows[-1].timestamp,
        val_start_timestamp=None if not val_rows else val_rows[0].timestamp,
        val_end_timestamp=None if not val_rows else val_rows[-1].timestamp,
        train_start_timeline_index=(
            None if not train_rows else train_rows[0].timeline_index
        ),
        train_end_timeline_index=None if not train_rows else train_rows[-1].timeline_index,
        val_start_timeline_index=None if not val_rows else val_rows[0].timeline_index,
        val_end_timeline_index=None if not val_rows else val_rows[-1].timeline_index,
        split_rule=split_rule,
    )


def write_joined_feature_csv(
    rows: Sequence[T16JoinedFeatureRow],
    output_path: Path | str,
) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(T16_OUTPUT_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def write_split_manifest_csv(
    rows: Sequence[T16SplitManifestRow],
    output_path: Path | str,
) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(T16_SPLIT_MANIFEST_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def load_joined_feature_csv(
    path: Path | str,
    *,
    expected_asset_id: str | None = None,
    expected_lbw: int | None = None,
) -> list[T16JoinedFeatureRow]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != T16_OUTPUT_HEADER:
            raise ValueError(f"T-16 joined-feature header mismatch: {csv_path}")

        rows: list[T16JoinedFeatureRow] = []
        previous_timestamp: str | None = None
        previous_timeline_index: int | None = None
        for row_index, row in enumerate(reader):
            timestamp = row["timestamp"]
            asset_id = row["asset_id"]
            lbw = int(row["lbw"])
            timeline_index = int(row["timeline_index"])
            if not timestamp:
                raise ValueError(f"T-16 row {row_index} missing timestamp: {csv_path}")
            if not asset_id:
                raise ValueError(f"T-16 row {row_index} missing asset_id: {csv_path}")
            if expected_asset_id is not None and asset_id != expected_asset_id:
                raise ValueError(
                    f"T-16 row {row_index} asset_id mismatch for {csv_path}: {asset_id}"
                )
            if expected_lbw is not None and lbw != expected_lbw:
                raise ValueError(
                    f"T-16 row {row_index} lbw mismatch for {csv_path}: {lbw}"
                )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(
                    f"T-16 timestamps must be strictly ascending in {csv_path}: "
                    f"{previous_timestamp} then {timestamp}"
                )
            if previous_timeline_index is not None and timeline_index <= previous_timeline_index:
                raise ValueError(
                    f"T-16 timeline indices must be strictly ascending in {csv_path}: "
                    f"{previous_timeline_index} then {timeline_index}"
                )
            previous_timestamp = timestamp
            previous_timeline_index = timeline_index
            model_inputs = tuple(
                _parse_required_float_text(
                    row[column_name],
                    csv_path=csv_path,
                    row_index=row_index,
                    column_name=column_name,
                )
                for column_name in MODEL_INPUT_COLUMNS
            )
            rows.append(
                T16JoinedFeatureRow(
                    timestamp=timestamp,
                    asset_id=asset_id,
                    lbw=lbw,
                    timeline_index=timeline_index,
                    sigma_t=_parse_required_float_text(
                        row["sigma_t"],
                        csv_path=csv_path,
                        row_index=row_index,
                        column_name="sigma_t",
                    ),
                    next_arithmetic_return=_parse_optional_float_text(
                        row["next_arithmetic_return"],
                        csv_path=csv_path,
                        row_index=row_index,
                        column_name="next_arithmetic_return",
                    ),
                    model_inputs=model_inputs,
                )
            )
    return rows


def load_split_manifest_csv(
    path: Path | str,
    *,
    expected_lbw: int | None = None,
) -> list[T16SplitManifestRow]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != T16_SPLIT_MANIFEST_HEADER:
            raise ValueError(f"T-16 split-manifest header mismatch: {csv_path}")

        rows: list[T16SplitManifestRow] = []
        seen_assets: set[str] = set()
        for row_index, row in enumerate(reader):
            asset_id = row["asset_id"]
            lbw = int(row["lbw"])
            if asset_id in seen_assets:
                raise ValueError(f"Duplicate split-manifest asset_id in {csv_path}: {asset_id}")
            seen_assets.add(asset_id)
            if expected_lbw is not None and lbw != expected_lbw:
                raise ValueError(
                    f"T-16 split-manifest row {row_index} lbw mismatch for {csv_path}: {lbw}"
                )
            usable_row_count = int(row["usable_row_count"])
            train_row_count = int(row["train_row_count"])
            val_row_count = int(row["val_row_count"])
            if usable_row_count != train_row_count + val_row_count:
                raise ValueError(
                    f"T-16 split-manifest row {row_index} count mismatch in {csv_path}"
                )
            rows.append(
                T16SplitManifestRow(
                    asset_id=asset_id,
                    lbw=lbw,
                    usable_row_count=usable_row_count,
                    train_row_count=train_row_count,
                    val_row_count=val_row_count,
                    train_start_timestamp=row["train_start_timestamp"] or None,
                    train_end_timestamp=row["train_end_timestamp"] or None,
                    val_start_timestamp=row["val_start_timestamp"] or None,
                    val_end_timestamp=row["val_end_timestamp"] or None,
                    train_start_timeline_index=_parse_optional_int_text(
                        row["train_start_timeline_index"]
                    ),
                    train_end_timeline_index=_parse_optional_int_text(
                        row["train_end_timeline_index"]
                    ),
                    val_start_timeline_index=_parse_optional_int_text(
                        row["val_start_timeline_index"]
                    ),
                    val_end_timeline_index=_parse_optional_int_text(
                        row["val_end_timeline_index"]
                    ),
                    split_rule=row["split_rule"],
                )
            )
    return rows


def build_t16_outputs(
    *,
    base_input_dir: Path | str = default_base_input_dir(),
    returns_input_dir: Path | str = default_returns_input_dir(),
    cpd_manifest_input: Path | str = default_cpd_manifest_input(),
    output_dir: Path | str = default_output_dir(),
    project_root: Path | str | None = None,
    lbws: Sequence[int] = ALLOWED_CPD_LBWS,
    asset_ids: Sequence[str] | None = None,
) -> T16OutputArtifacts:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    output_dir_path = Path(output_dir)
    manifest_records = load_cpd_feature_store_manifest(cpd_manifest_input)
    asset_order, manifest_index = _select_manifest_records(
        manifest_records,
        lbws=lbws,
        asset_ids=asset_ids,
    )

    base_paths = _build_asset_path_map(base_input_dir, suffix=BASE_FEATURES_SUFFIX)
    returns_paths = _build_asset_path_map(
        returns_input_dir,
        suffix=RETURNS_VOLATILITY_SUFFIX,
    )
    missing_base_assets = [asset_id for asset_id in asset_order if asset_id not in base_paths]
    if missing_base_assets:
        raise ValueError(f"Missing base-feature inputs for assets: {missing_base_assets}")
    missing_returns_assets = [
        asset_id for asset_id in asset_order if asset_id not in returns_paths
    ]
    if missing_returns_assets:
        raise ValueError(
            f"Missing returns-volatility inputs for assets: {missing_returns_assets}"
        )

    joined_feature_paths: list[Path] = []
    split_manifest_paths: list[Path] = []
    requested_lbws = _validate_requested_lbws(lbws)
    for lbw in requested_lbws:
        split_rows: list[T16SplitManifestRow] = []
        for asset_id in asset_order:
            base_rows = load_base_feature_csv(base_paths[asset_id], expected_asset_id=asset_id)
            returns_rows = load_returns_volatility_join_csv(
                returns_paths[asset_id],
                expected_asset_id=asset_id,
            )
            manifest_record = manifest_index[(asset_id, lbw)]
            cpd_csv_path = project_root_path / manifest_record.cpd_csv_path
            cpd_rows = load_cpd_feature_csv(
                cpd_csv_path,
                expected_asset_id=asset_id,
                expected_lbw=lbw,
            )
            joined_rows = build_joined_feature_rows(
                base_rows=base_rows,
                returns_rows=returns_rows,
                cpd_rows=cpd_rows,
                lbw=lbw,
            )
            joined_path = (
                output_dir_path / f"lbw_{lbw}" / "joined_features" / f"{asset_id}.csv"
            )
            write_joined_feature_csv(joined_rows, joined_path)
            joined_feature_paths.append(joined_path)
            split_rows.append(build_split_manifest_row(joined_rows, asset_id=asset_id, lbw=lbw))

        split_manifest_path = output_dir_path / f"lbw_{lbw}" / "split_manifest.csv"
        write_split_manifest_csv(split_rows, split_manifest_path)
        split_manifest_paths.append(split_manifest_path)

    return T16OutputArtifacts(
        joined_feature_paths=joined_feature_paths,
        split_manifest_paths=split_manifest_paths,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build T-16 joined feature tables and chronological split manifests."
    )
    parser.add_argument("--base-input-dir", type=Path, default=default_base_input_dir())
    parser.add_argument(
        "--returns-input-dir",
        type=Path,
        default=default_returns_input_dir(),
    )
    parser.add_argument(
        "--cpd-manifest-input",
        type=Path,
        default=default_cpd_manifest_input(),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument(
        "--lbw",
        type=int,
        action="append",
        dest="lbws",
        default=None,
        help="Limit output generation to one or more allowed LBWs.",
    )
    parser.add_argument(
        "--asset-id",
        action="append",
        dest="asset_ids",
        default=None,
        help="Limit output generation to one or more asset IDs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = build_t16_outputs(
        base_input_dir=args.base_input_dir,
        returns_input_dir=args.returns_input_dir,
        cpd_manifest_input=args.cpd_manifest_input,
        output_dir=args.output_dir,
        project_root=args.project_root,
        lbws=ALLOWED_CPD_LBWS if args.lbws is None else tuple(args.lbws),
        asset_ids=args.asset_ids,
    )
    print(
        "Wrote "
        f"{len(artifacts.joined_feature_paths)} joined feature files and "
        f"{len(artifacts.split_manifest_paths)} split manifests."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
