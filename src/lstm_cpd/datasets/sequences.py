from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from lstm_cpd.cpd.precompute_contract import ALLOWED_CPD_LBWS, is_allowed_lbw
from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.datasets.join_and_split import (
    MODEL_INPUT_COLUMNS,
    T16JoinedFeatureRow,
    T16SplitManifestRow,
    default_output_dir as default_datasets_dir,
    load_joined_feature_csv,
    load_split_manifest_csv,
    project_relative_path,
)
from lstm_cpd.features.returns import serialize_optional_float


SEQUENCE_LENGTH = 63
TARGET_VOLATILITY = 0.15
SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"

REASON_GAP_IN_TIMELINE = "GAP_IN_TIMELINE"
REASON_MISSING_NEXT_RETURN = "MISSING_NEXT_RETURN"
REASON_TERMINAL_FRAGMENT_LT_63 = "TERMINAL_FRAGMENT_LT_63"
REASON_NO_FULL_VALIDATION_SEQUENCE = "NO_FULL_VALIDATION_SEQUENCE"

T17_SEQUENCE_MANIFEST_HEADER = (
    "sequence_id",
    "asset_id",
    "lbw",
    "split",
    "sequence_index",
    "start_timestamp",
    "end_timestamp",
    "start_timeline_index",
    "end_timeline_index",
    "row_count",
)
T17_TARGET_ALIGNMENT_HEADER = (
    "sequence_id",
    "asset_id",
    "lbw",
    "split",
    "step_index",
    "timestamp",
    "timeline_index",
    "sigma_t",
    "next_arithmetic_return",
    "target_scale",
) + MODEL_INPUT_COLUMNS
T17_DISCARDED_FRAGMENTS_HEADER = (
    "asset_id",
    "lbw",
    "split",
    "fragment_start_timestamp",
    "fragment_end_timestamp",
    "fragment_start_timeline_index",
    "fragment_end_timeline_index",
    "fragment_length",
    "reason_code",
    "dropped_sequence_count",
)
T17_GAP_EXCLUSION_HEADER = (
    "asset_id",
    "lbw",
    "split",
    "previous_timestamp",
    "current_timestamp",
    "previous_timeline_index",
    "current_timeline_index",
    "missing_steps",
    "reason_code",
)


@dataclass(frozen=True)
class T17SequenceManifestRow:
    sequence_id: str
    asset_id: str
    lbw: int
    split: str
    sequence_index: int
    start_timestamp: str
    end_timestamp: str
    start_timeline_index: int
    end_timeline_index: int
    row_count: int = SEQUENCE_LENGTH

    def to_csv_row(self) -> dict[str, str]:
        return {
            "sequence_id": self.sequence_id,
            "asset_id": self.asset_id,
            "lbw": str(self.lbw),
            "split": self.split,
            "sequence_index": str(self.sequence_index),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "start_timeline_index": str(self.start_timeline_index),
            "end_timeline_index": str(self.end_timeline_index),
            "row_count": str(self.row_count),
        }


@dataclass(frozen=True)
class T17TargetAlignmentRow:
    sequence_id: str
    asset_id: str
    lbw: int
    split: str
    step_index: int
    timestamp: str
    timeline_index: int
    sigma_t: float
    next_arithmetic_return: float
    target_scale: float
    model_inputs: tuple[float, ...]

    def to_csv_row(self) -> dict[str, str]:
        row = {
            "sequence_id": self.sequence_id,
            "asset_id": self.asset_id,
            "lbw": str(self.lbw),
            "split": self.split,
            "step_index": str(self.step_index),
            "timestamp": self.timestamp,
            "timeline_index": str(self.timeline_index),
            "sigma_t": serialize_optional_float(self.sigma_t),
            "next_arithmetic_return": serialize_optional_float(self.next_arithmetic_return),
            "target_scale": serialize_optional_float(self.target_scale),
        }
        for column_name, value in zip(MODEL_INPUT_COLUMNS, self.model_inputs):
            row[column_name] = serialize_optional_float(value)
        return row


@dataclass(frozen=True)
class T17DiscardedFragmentRow:
    asset_id: str
    lbw: int
    split: str
    fragment_start_timestamp: str | None
    fragment_end_timestamp: str | None
    fragment_start_timeline_index: int | None
    fragment_end_timeline_index: int | None
    fragment_length: int
    reason_code: str
    dropped_sequence_count: int

    def to_csv_row(self) -> dict[str, str]:
        return {
            "asset_id": self.asset_id,
            "lbw": str(self.lbw),
            "split": self.split,
            "fragment_start_timestamp": self.fragment_start_timestamp or "",
            "fragment_end_timestamp": self.fragment_end_timestamp or "",
            "fragment_start_timeline_index": _serialize_optional_int(
                self.fragment_start_timeline_index
            ),
            "fragment_end_timeline_index": _serialize_optional_int(
                self.fragment_end_timeline_index
            ),
            "fragment_length": str(self.fragment_length),
            "reason_code": self.reason_code,
            "dropped_sequence_count": str(self.dropped_sequence_count),
        }


@dataclass(frozen=True)
class T17GapExclusionRow:
    asset_id: str
    lbw: int
    split: str
    previous_timestamp: str
    current_timestamp: str
    previous_timeline_index: int
    current_timeline_index: int
    missing_steps: int
    reason_code: str = REASON_GAP_IN_TIMELINE

    def to_csv_row(self) -> dict[str, str]:
        return {
            "asset_id": self.asset_id,
            "lbw": str(self.lbw),
            "split": self.split,
            "previous_timestamp": self.previous_timestamp,
            "current_timestamp": self.current_timestamp,
            "previous_timeline_index": str(self.previous_timeline_index),
            "current_timeline_index": str(self.current_timeline_index),
            "missing_steps": str(self.missing_steps),
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class T17OutputArtifacts:
    sequence_manifest_paths: list[Path]
    target_alignment_registry_paths: list[Path]
    discarded_fragment_report_paths: list[Path]
    gap_exclusion_report_paths: list[Path]


def default_input_dir() -> Path:
    return default_datasets_dir()


def default_output_dir() -> Path:
    return default_datasets_dir()


def _serialize_optional_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def _validate_requested_lbws(lbws: Sequence[int]) -> tuple[int, ...]:
    unique_lbws = tuple(dict.fromkeys(int(lbw) for lbw in lbws))
    if not unique_lbws:
        raise ValueError("At least one lbw is required")
    invalid_lbws = [lbw for lbw in unique_lbws if not is_allowed_lbw(lbw)]
    if invalid_lbws:
        raise ValueError(f"Unsupported lbws requested: {invalid_lbws}")
    return unique_lbws


def build_sequence_id(
    *,
    asset_id: str,
    lbw: int,
    split: str,
    start_timeline_index: int,
) -> str:
    return f"{asset_id}__lbw_{lbw}__{split}__{start_timeline_index:08d}"


def _write_csv_rows(
    rows: Sequence[dict[str, str]],
    *,
    header: Sequence[str],
    output_path: Path | str,
) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _validate_split_row_against_joined_rows(
    split_row: T16SplitManifestRow,
    joined_rows: Sequence[T16JoinedFeatureRow],
) -> None:
    if split_row.usable_row_count != len(joined_rows):
        raise ValueError(
            f"T-17 usable-row mismatch for {split_row.asset_id}, lbw={split_row.lbw}: "
            f"{split_row.usable_row_count} != {len(joined_rows)}"
        )
    if joined_rows:
        train_rows = joined_rows[: split_row.train_row_count]
        val_rows = joined_rows[split_row.train_row_count :]
        if bool(train_rows) != bool(split_row.train_start_timestamp):
            raise ValueError(
                f"T-17 train split metadata mismatch for {split_row.asset_id}, "
                f"lbw={split_row.lbw}"
            )
        if bool(val_rows) != bool(split_row.val_start_timestamp):
            raise ValueError(
                f"T-17 validation split metadata mismatch for {split_row.asset_id}, "
                f"lbw={split_row.lbw}"
            )
        if train_rows:
            if train_rows[0].timestamp != split_row.train_start_timestamp:
                raise ValueError("T-17 train start timestamp mismatch")
            if train_rows[-1].timestamp != split_row.train_end_timestamp:
                raise ValueError("T-17 train end timestamp mismatch")
        if val_rows:
            if val_rows[0].timestamp != split_row.val_start_timestamp:
                raise ValueError("T-17 validation start timestamp mismatch")
            if val_rows[-1].timestamp != split_row.val_end_timestamp:
                raise ValueError("T-17 validation end timestamp mismatch")


def _segment_rows(
    rows: Sequence[T16JoinedFeatureRow],
    *,
    split: str,
    asset_id: str,
    lbw: int,
) -> tuple[list[list[T16JoinedFeatureRow]], list[T17GapExclusionRow], list[T17DiscardedFragmentRow]]:
    runs: list[list[T16JoinedFeatureRow]] = []
    gap_rows: list[T17GapExclusionRow] = []
    discarded_rows: list[T17DiscardedFragmentRow] = []
    current_run: list[T16JoinedFeatureRow] = []

    for row in rows:
        if row.next_arithmetic_return is None:
            if current_run:
                runs.append(current_run)
                current_run = []
            discarded_rows.append(
                T17DiscardedFragmentRow(
                    asset_id=asset_id,
                    lbw=lbw,
                    split=split,
                    fragment_start_timestamp=row.timestamp,
                    fragment_end_timestamp=row.timestamp,
                    fragment_start_timeline_index=row.timeline_index,
                    fragment_end_timeline_index=row.timeline_index,
                    fragment_length=1,
                    reason_code=REASON_MISSING_NEXT_RETURN,
                    dropped_sequence_count=0,
                )
            )
            continue

        if not current_run:
            current_run = [row]
            continue

        previous_row = current_run[-1]
        if row.timeline_index == previous_row.timeline_index + 1:
            current_run.append(row)
            continue

        runs.append(current_run)
        gap_rows.append(
            T17GapExclusionRow(
                asset_id=asset_id,
                lbw=lbw,
                split=split,
                previous_timestamp=previous_row.timestamp,
                current_timestamp=row.timestamp,
                previous_timeline_index=previous_row.timeline_index,
                current_timeline_index=row.timeline_index,
                missing_steps=row.timeline_index - previous_row.timeline_index - 1,
            )
        )
        current_run = [row]

    if current_run:
        runs.append(current_run)
    return runs, gap_rows, discarded_rows


def _remove_split_terminal_target(
    rows: Sequence[T16JoinedFeatureRow],
) -> list[T16JoinedFeatureRow]:
    if not rows:
        return []
    sanitized_rows = list(rows)
    terminal_row = sanitized_rows[-1]
    if terminal_row.next_arithmetic_return is not None:
        sanitized_rows[-1] = replace(terminal_row, next_arithmetic_return=None)
    return sanitized_rows


def _build_sequences_for_split(
    rows: Sequence[T16JoinedFeatureRow],
    *,
    split: str,
) -> tuple[
    list[T17SequenceManifestRow],
    list[T17TargetAlignmentRow],
    list[T17GapExclusionRow],
    list[T17DiscardedFragmentRow],
]:
    if not rows:
        return [], [], [], []

    split_rows = _remove_split_terminal_target(rows)
    asset_id = split_rows[0].asset_id
    lbw = split_rows[0].lbw
    runs, gap_rows, discarded_rows = _segment_rows(
        split_rows,
        split=split,
        asset_id=asset_id,
        lbw=lbw,
    )

    sequence_rows: list[T17SequenceManifestRow] = []
    target_rows: list[T17TargetAlignmentRow] = []
    sequence_index = 0
    for run in runs:
        full_sequence_count = len(run) // SEQUENCE_LENGTH
        for block_index in range(full_sequence_count):
            start = block_index * SEQUENCE_LENGTH
            sequence_rows_slice = run[start : start + SEQUENCE_LENGTH]
            first_row = sequence_rows_slice[0]
            last_row = sequence_rows_slice[-1]
            sequence_id = build_sequence_id(
                asset_id=asset_id,
                lbw=lbw,
                split=split,
                start_timeline_index=first_row.timeline_index,
            )
            sequence_rows.append(
                T17SequenceManifestRow(
                    sequence_id=sequence_id,
                    asset_id=asset_id,
                    lbw=lbw,
                    split=split,
                    sequence_index=sequence_index,
                    start_timestamp=first_row.timestamp,
                    end_timestamp=last_row.timestamp,
                    start_timeline_index=first_row.timeline_index,
                    end_timeline_index=last_row.timeline_index,
                )
            )
            sequence_index += 1
            for step_index, row in enumerate(sequence_rows_slice):
                if row.sigma_t <= 0.0 or not math.isfinite(row.sigma_t):
                    raise ValueError(
                        f"T-17 encountered non-positive sigma_t for {asset_id}, lbw={lbw}, "
                        f"timestamp={row.timestamp}"
                    )
                target_scale = TARGET_VOLATILITY / row.sigma_t * row.next_arithmetic_return
                target_rows.append(
                    T17TargetAlignmentRow(
                        sequence_id=sequence_id,
                        asset_id=asset_id,
                        lbw=lbw,
                        split=split,
                        step_index=step_index,
                        timestamp=row.timestamp,
                        timeline_index=row.timeline_index,
                        sigma_t=row.sigma_t,
                        next_arithmetic_return=row.next_arithmetic_return,
                        target_scale=target_scale,
                        model_inputs=row.model_inputs,
                    )
                )

        remainder = len(run) % SEQUENCE_LENGTH
        if remainder:
            fragment_rows = run[-remainder:]
            discarded_rows.append(
                T17DiscardedFragmentRow(
                    asset_id=asset_id,
                    lbw=lbw,
                    split=split,
                    fragment_start_timestamp=fragment_rows[0].timestamp,
                    fragment_end_timestamp=fragment_rows[-1].timestamp,
                    fragment_start_timeline_index=fragment_rows[0].timeline_index,
                    fragment_end_timeline_index=fragment_rows[-1].timeline_index,
                    fragment_length=remainder,
                    reason_code=REASON_TERMINAL_FRAGMENT_LT_63,
                    dropped_sequence_count=0,
                )
            )
    return sequence_rows, target_rows, gap_rows, discarded_rows


def write_sequence_manifest_csv(
    rows: Sequence[T17SequenceManifestRow],
    output_path: Path | str,
) -> None:
    _write_csv_rows(
        [row.to_csv_row() for row in rows],
        header=T17_SEQUENCE_MANIFEST_HEADER,
        output_path=output_path,
    )


def write_target_alignment_registry_csv(
    rows: Sequence[T17TargetAlignmentRow],
    output_path: Path | str,
) -> None:
    _write_csv_rows(
        [row.to_csv_row() for row in rows],
        header=T17_TARGET_ALIGNMENT_HEADER,
        output_path=output_path,
    )


def write_discarded_fragment_report_csv(
    rows: Sequence[T17DiscardedFragmentRow],
    output_path: Path | str,
) -> None:
    _write_csv_rows(
        [row.to_csv_row() for row in rows],
        header=T17_DISCARDED_FRAGMENTS_HEADER,
        output_path=output_path,
    )


def write_gap_exclusion_report_csv(
    rows: Sequence[T17GapExclusionRow],
    output_path: Path | str,
) -> None:
    _write_csv_rows(
        [row.to_csv_row() for row in rows],
        header=T17_GAP_EXCLUSION_HEADER,
        output_path=output_path,
    )


def _parse_required_float_text(
    text: str,
    *,
    csv_path: Path,
    row_index: int,
    column_name: str,
) -> float:
    if text == "":
        raise ValueError(
            f"T-17 row {row_index} column {column_name} must not be blank: {csv_path}"
        )
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(
            f"T-17 row {row_index} column {column_name} has non-finite value: {csv_path}"
        )
    return value


def load_sequence_manifest_csv(
    path: Path | str,
    *,
    expected_lbw: int | None = None,
) -> list[T17SequenceManifestRow]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != T17_SEQUENCE_MANIFEST_HEADER:
            raise ValueError(f"T-17 sequence-manifest header mismatch: {csv_path}")

        rows: list[T17SequenceManifestRow] = []
        seen_sequence_ids: set[str] = set()
        for row_index, row in enumerate(reader):
            sequence_id = row["sequence_id"]
            if sequence_id in seen_sequence_ids:
                raise ValueError(f"Duplicate T-17 sequence_id in {csv_path}: {sequence_id}")
            seen_sequence_ids.add(sequence_id)
            lbw = int(row["lbw"])
            if expected_lbw is not None and lbw != expected_lbw:
                raise ValueError(
                    f"T-17 sequence-manifest row {row_index} lbw mismatch for "
                    f"{csv_path}: {lbw}"
                )
            row_count = int(row["row_count"])
            if row_count != SEQUENCE_LENGTH:
                raise ValueError(
                    f"T-17 sequence-manifest row {row_index} has invalid row_count "
                    f"{row_count}: {csv_path}"
                )
            rows.append(
                T17SequenceManifestRow(
                    sequence_id=sequence_id,
                    asset_id=row["asset_id"],
                    lbw=lbw,
                    split=row["split"],
                    sequence_index=int(row["sequence_index"]),
                    start_timestamp=row["start_timestamp"],
                    end_timestamp=row["end_timestamp"],
                    start_timeline_index=int(row["start_timeline_index"]),
                    end_timeline_index=int(row["end_timeline_index"]),
                    row_count=row_count,
                )
            )
    return rows


def load_target_alignment_registry_csv(
    path: Path | str,
    *,
    expected_lbw: int | None = None,
) -> list[T17TargetAlignmentRow]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != T17_TARGET_ALIGNMENT_HEADER:
            raise ValueError(
                f"T-17 target-alignment registry header mismatch: {csv_path}"
            )

        rows: list[T17TargetAlignmentRow] = []
        for row_index, row in enumerate(reader):
            lbw = int(row["lbw"])
            if expected_lbw is not None and lbw != expected_lbw:
                raise ValueError(
                    f"T-17 target-alignment row {row_index} lbw mismatch for "
                    f"{csv_path}: {lbw}"
                )
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
                T17TargetAlignmentRow(
                    sequence_id=row["sequence_id"],
                    asset_id=row["asset_id"],
                    lbw=lbw,
                    split=row["split"],
                    step_index=int(row["step_index"]),
                    timestamp=row["timestamp"],
                    timeline_index=int(row["timeline_index"]),
                    sigma_t=_parse_required_float_text(
                        row["sigma_t"],
                        csv_path=csv_path,
                        row_index=row_index,
                        column_name="sigma_t",
                    ),
                    next_arithmetic_return=_parse_required_float_text(
                        row["next_arithmetic_return"],
                        csv_path=csv_path,
                        row_index=row_index,
                        column_name="next_arithmetic_return",
                    ),
                    target_scale=_parse_required_float_text(
                        row["target_scale"],
                        csv_path=csv_path,
                        row_index=row_index,
                        column_name="target_scale",
                    ),
                    model_inputs=model_inputs,
                )
            )
    return rows


def build_t17_outputs(
    *,
    input_dir: Path | str = default_input_dir(),
    output_dir: Path | str = default_output_dir(),
    project_root: Path | str | None = None,
    lbws: Sequence[int] = ALLOWED_CPD_LBWS,
) -> T17OutputArtifacts:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    requested_lbws = _validate_requested_lbws(lbws)

    sequence_manifest_paths: list[Path] = []
    target_alignment_registry_paths: list[Path] = []
    discarded_fragment_report_paths: list[Path] = []
    gap_exclusion_report_paths: list[Path] = []

    for lbw in requested_lbws:
        split_manifest_path = input_dir_path / f"lbw_{lbw}" / "split_manifest.csv"
        split_rows = load_split_manifest_csv(split_manifest_path, expected_lbw=lbw)

        sequence_rows: list[T17SequenceManifestRow] = []
        target_rows: list[T17TargetAlignmentRow] = []
        discarded_rows: list[T17DiscardedFragmentRow] = []
        gap_rows: list[T17GapExclusionRow] = []

        for split_row in split_rows:
            joined_path = (
                input_dir_path / f"lbw_{lbw}" / "joined_features" / f"{split_row.asset_id}.csv"
            )
            joined_rows = load_joined_feature_csv(
                joined_path,
                expected_asset_id=split_row.asset_id,
                expected_lbw=lbw,
            )
            _validate_split_row_against_joined_rows(split_row, joined_rows)

            train_rows = joined_rows[: split_row.train_row_count]
            val_rows = joined_rows[split_row.train_row_count :]
            (
                asset_train_sequences,
                asset_train_targets,
                asset_train_gaps,
                asset_train_discards,
            ) = _build_sequences_for_split(train_rows, split=SPLIT_TRAIN)
            (
                asset_val_sequences,
                asset_val_targets,
                asset_val_gaps,
                asset_val_discards,
            ) = _build_sequences_for_split(val_rows, split=SPLIT_VALIDATION)

            gap_rows.extend(asset_train_gaps)
            gap_rows.extend(asset_val_gaps)
            discarded_rows.extend(asset_train_discards)
            discarded_rows.extend(asset_val_discards)

            if not asset_val_sequences:
                discarded_rows.append(
                    T17DiscardedFragmentRow(
                        asset_id=split_row.asset_id,
                        lbw=lbw,
                        split=SPLIT_VALIDATION,
                        fragment_start_timestamp=split_row.val_start_timestamp,
                        fragment_end_timestamp=split_row.val_end_timestamp,
                        fragment_start_timeline_index=split_row.val_start_timeline_index,
                        fragment_end_timeline_index=split_row.val_end_timeline_index,
                        fragment_length=split_row.val_row_count,
                        reason_code=REASON_NO_FULL_VALIDATION_SEQUENCE,
                        dropped_sequence_count=len(asset_train_sequences),
                    )
                )
                continue

            sequence_rows.extend(asset_train_sequences)
            sequence_rows.extend(asset_val_sequences)
            target_rows.extend(asset_train_targets)
            target_rows.extend(asset_val_targets)

        lbw_dir = output_dir_path / f"lbw_{lbw}"
        sequence_manifest_output = lbw_dir / "sequence_manifest.csv"
        target_alignment_output = lbw_dir / "target_alignment_registry.csv"
        discarded_output = lbw_dir / "discarded_fragments_report.csv"
        gap_output = lbw_dir / "gap_exclusion_report.csv"

        write_sequence_manifest_csv(sequence_rows, sequence_manifest_output)
        write_target_alignment_registry_csv(target_rows, target_alignment_output)
        write_discarded_fragment_report_csv(discarded_rows, discarded_output)
        write_gap_exclusion_report_csv(gap_rows, gap_output)

        sequence_manifest_paths.append(sequence_manifest_output)
        target_alignment_registry_paths.append(target_alignment_output)
        discarded_fragment_report_paths.append(discarded_output)
        gap_exclusion_report_paths.append(gap_output)

        # Validate that the written paths stay project-local when using the project root.
        project_relative_path(sequence_manifest_output, project_root_path)
        project_relative_path(target_alignment_output, project_root_path)
        project_relative_path(discarded_output, project_root_path)
        project_relative_path(gap_output, project_root_path)

    return T17OutputArtifacts(
        sequence_manifest_paths=sequence_manifest_paths,
        target_alignment_registry_paths=target_alignment_registry_paths,
        discarded_fragment_report_paths=discarded_fragment_report_paths,
        gap_exclusion_report_paths=gap_exclusion_report_paths,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build T-17 sequence manifests and target-alignment registries."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = build_t17_outputs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
        lbws=ALLOWED_CPD_LBWS if args.lbws is None else tuple(args.lbws),
    )
    print(
        "Wrote "
        f"{len(artifacts.sequence_manifest_paths)} sequence manifests, "
        f"{len(artifacts.target_alignment_registry_paths)} target-alignment registries, "
        f"{len(artifacts.discarded_fragment_report_paths)} discarded-fragment reports, "
        f"and {len(artifacts.gap_exclusion_report_paths)} gap reports."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
