from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from lstm_cpd.cpd.precompute_contract import ALLOWED_CPD_LBWS, is_allowed_lbw
from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.datasets.join_and_split import MODEL_INPUT_COLUMNS, project_relative_path
from lstm_cpd.datasets.sequences import (
    SEQUENCE_LENGTH,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    T17SequenceManifestRow,
    T17TargetAlignmentRow,
    default_input_dir as default_datasets_dir,
    load_sequence_manifest_csv,
    load_target_alignment_registry_csv,
)


T18_SEQUENCE_INDEX_HEADER = (
    "array_row_index",
    "sequence_id",
    "asset_id",
    "lbw",
    "split",
    "start_timestamp",
    "end_timestamp",
    "start_timeline_index",
    "end_timeline_index",
)


@dataclass(frozen=True)
class T18SequenceIndexRow:
    array_row_index: int
    sequence_id: str
    asset_id: str
    lbw: int
    split: str
    start_timestamp: str
    end_timestamp: str
    start_timeline_index: int
    end_timeline_index: int

    def to_csv_row(self) -> dict[str, str]:
        return {
            "array_row_index": str(self.array_row_index),
            "sequence_id": self.sequence_id,
            "asset_id": self.asset_id,
            "lbw": str(self.lbw),
            "split": self.split,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "start_timeline_index": str(self.start_timeline_index),
            "end_timeline_index": str(self.end_timeline_index),
        }


@dataclass(frozen=True)
class T18OutputArtifacts:
    dataset_registry_path: Path
    train_input_paths: list[Path]
    val_input_paths: list[Path]


def default_input_dir() -> Path:
    return default_datasets_dir()


def default_output_dir() -> Path:
    return default_datasets_dir()


def default_dataset_registry_output() -> Path:
    return default_project_root() / "artifacts/manifests/dataset_registry.json"


def _validate_requested_lbws(lbws: Sequence[int]) -> tuple[int, ...]:
    unique_lbws = tuple(dict.fromkeys(int(lbw) for lbw in lbws))
    if not unique_lbws:
        raise ValueError("At least one lbw is required")
    invalid_lbws = [lbw for lbw in unique_lbws if not is_allowed_lbw(lbw)]
    if invalid_lbws:
        raise ValueError(f"Unsupported lbws requested: {invalid_lbws}")
    return unique_lbws


def _write_sequence_index_csv(
    rows: Sequence[T18SequenceIndexRow],
    output_path: Path | str,
) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(T18_SEQUENCE_INDEX_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def _sort_sequence_rows(
    rows: Sequence[T17SequenceManifestRow],
) -> list[T17SequenceManifestRow]:
    return sorted(
        rows,
        key=lambda row: (row.asset_id, row.start_timeline_index, row.sequence_id),
    )


def _group_target_rows(
    rows: Sequence[T17TargetAlignmentRow],
) -> dict[str, list[T17TargetAlignmentRow]]:
    grouped: dict[str, list[T17TargetAlignmentRow]] = {}
    for row in rows:
        grouped.setdefault(row.sequence_id, []).append(row)
    return grouped


def _validate_target_rows_for_sequence(
    sequence_row: T17SequenceManifestRow,
    rows: Sequence[T17TargetAlignmentRow],
) -> list[T17TargetAlignmentRow]:
    ordered_rows = sorted(rows, key=lambda row: row.step_index)
    if len(ordered_rows) != SEQUENCE_LENGTH:
        raise ValueError(
            f"T-18 target registry length mismatch for {sequence_row.sequence_id}: "
            f"{len(ordered_rows)} != {SEQUENCE_LENGTH}"
        )
    expected_step_indices = list(range(SEQUENCE_LENGTH))
    actual_step_indices = [row.step_index for row in ordered_rows]
    if actual_step_indices != expected_step_indices:
        raise ValueError(
            f"T-18 target registry step mismatch for {sequence_row.sequence_id}: "
            f"{actual_step_indices[:5]} ..."
        )
    for row in ordered_rows:
        if row.asset_id != sequence_row.asset_id:
            raise ValueError(
                f"T-18 target registry asset mismatch for {sequence_row.sequence_id}"
            )
        if row.lbw != sequence_row.lbw:
            raise ValueError(
                f"T-18 target registry lbw mismatch for {sequence_row.sequence_id}"
            )
        if row.split != sequence_row.split:
            raise ValueError(
                f"T-18 target registry split mismatch for {sequence_row.sequence_id}"
            )
    if ordered_rows[0].timestamp != sequence_row.start_timestamp:
        raise ValueError(
            f"T-18 target registry start timestamp mismatch for {sequence_row.sequence_id}"
        )
    if ordered_rows[-1].timestamp != sequence_row.end_timestamp:
        raise ValueError(
            f"T-18 target registry end timestamp mismatch for {sequence_row.sequence_id}"
        )
    if ordered_rows[0].timeline_index != sequence_row.start_timeline_index:
        raise ValueError(
            f"T-18 target registry start timeline mismatch for {sequence_row.sequence_id}"
        )
    if ordered_rows[-1].timeline_index != sequence_row.end_timeline_index:
        raise ValueError(
            f"T-18 target registry end timeline mismatch for {sequence_row.sequence_id}"
        )
    return ordered_rows


def _build_split_arrays(
    sequence_rows: Sequence[T17SequenceManifestRow],
    grouped_target_rows: dict[str, list[T17TargetAlignmentRow]],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[T18SequenceIndexRow]]:
    split_sequence_rows = _sort_sequence_rows(
        [row for row in sequence_rows if row.split == split]
    )
    input_count = len(split_sequence_rows)
    inputs = np.empty(
        (input_count, SEQUENCE_LENGTH, len(MODEL_INPUT_COLUMNS)),
        dtype=np.float32,
    )
    targets = np.empty((input_count, SEQUENCE_LENGTH), dtype=np.float32)
    index_rows: list[T18SequenceIndexRow] = []

    for array_row_index, sequence_row in enumerate(split_sequence_rows):
        target_rows = grouped_target_rows.get(sequence_row.sequence_id)
        if target_rows is None:
            raise ValueError(
                f"T-18 missing target-alignment rows for {sequence_row.sequence_id}"
            )
        ordered_rows = _validate_target_rows_for_sequence(sequence_row, target_rows)
        for step_index, row in enumerate(ordered_rows):
            inputs[array_row_index, step_index, :] = np.asarray(
                row.model_inputs,
                dtype=np.float32,
            )
            targets[array_row_index, step_index] = np.float32(row.target_scale)
        index_rows.append(
            T18SequenceIndexRow(
                array_row_index=array_row_index,
                sequence_id=sequence_row.sequence_id,
                asset_id=sequence_row.asset_id,
                lbw=sequence_row.lbw,
                split=sequence_row.split,
                start_timestamp=sequence_row.start_timestamp,
                end_timestamp=sequence_row.end_timestamp,
                start_timeline_index=sequence_row.start_timeline_index,
                end_timeline_index=sequence_row.end_timeline_index,
            )
        )
    return inputs, targets, index_rows


def build_t18_outputs(
    *,
    input_dir: Path | str = default_input_dir(),
    output_dir: Path | str = default_output_dir(),
    dataset_registry_output: Path | str = default_dataset_registry_output(),
    project_root: Path | str | None = None,
    lbws: Sequence[int] = ALLOWED_CPD_LBWS,
) -> T18OutputArtifacts:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    dataset_registry_path = Path(dataset_registry_output)
    requested_lbws = _validate_requested_lbws(lbws)

    registry_entries: list[dict[str, object]] = []
    train_input_paths: list[Path] = []
    val_input_paths: list[Path] = []
    for lbw in requested_lbws:
        lbw_dir = input_dir_path / f"lbw_{lbw}"
        sequence_manifest_path = lbw_dir / "sequence_manifest.csv"
        target_alignment_path = lbw_dir / "target_alignment_registry.csv"
        split_manifest_path = lbw_dir / "split_manifest.csv"

        sequence_rows = load_sequence_manifest_csv(
            sequence_manifest_path,
            expected_lbw=lbw,
        )
        target_rows = load_target_alignment_registry_csv(
            target_alignment_path,
            expected_lbw=lbw,
        )
        grouped_target_rows = _group_target_rows(target_rows)

        extra_sequence_ids = sorted(
            set(grouped_target_rows) - {row.sequence_id for row in sequence_rows}
        )
        if extra_sequence_ids:
            raise ValueError(
                f"T-18 found target-alignment rows without sequence-manifest entries "
                f"for lbw={lbw}: {extra_sequence_ids}"
            )

        train_inputs, train_targets, train_index_rows = _build_split_arrays(
            sequence_rows,
            grouped_target_rows,
            split=SPLIT_TRAIN,
        )
        val_inputs, val_targets, val_index_rows = _build_split_arrays(
            sequence_rows,
            grouped_target_rows,
            split=SPLIT_VALIDATION,
        )

        output_lbw_dir = output_dir_path / f"lbw_{lbw}"
        output_lbw_dir.mkdir(parents=True, exist_ok=True)
        train_inputs_path = output_lbw_dir / "train_inputs.npy"
        train_targets_path = output_lbw_dir / "train_target_scale.npy"
        val_inputs_path = output_lbw_dir / "val_inputs.npy"
        val_targets_path = output_lbw_dir / "val_target_scale.npy"
        train_index_path = output_lbw_dir / "train_sequence_index.csv"
        val_index_path = output_lbw_dir / "val_sequence_index.csv"

        np.save(train_inputs_path, train_inputs)
        np.save(train_targets_path, train_targets)
        np.save(val_inputs_path, val_inputs)
        np.save(val_targets_path, val_targets)
        _write_sequence_index_csv(train_index_rows, train_index_path)
        _write_sequence_index_csv(val_index_rows, val_index_path)

        train_input_paths.append(train_inputs_path)
        val_input_paths.append(val_inputs_path)
        registry_entries.append(
            {
                "lbw": lbw,
                "feature_columns": list(MODEL_INPUT_COLUMNS),
                "sequence_length": SEQUENCE_LENGTH,
                "train_sequence_count": int(train_inputs.shape[0]),
                "val_sequence_count": int(val_inputs.shape[0]),
                "train_input_shape": list(train_inputs.shape),
                "train_target_shape": list(train_targets.shape),
                "val_input_shape": list(val_inputs.shape),
                "val_target_shape": list(val_targets.shape),
                "artifacts": {
                    "train_inputs_path": project_relative_path(
                        train_inputs_path, project_root_path
                    ),
                    "train_target_scale_path": project_relative_path(
                        train_targets_path, project_root_path
                    ),
                    "val_inputs_path": project_relative_path(
                        val_inputs_path, project_root_path
                    ),
                    "val_target_scale_path": project_relative_path(
                        val_targets_path, project_root_path
                    ),
                    "train_sequence_index_path": project_relative_path(
                        train_index_path, project_root_path
                    ),
                    "val_sequence_index_path": project_relative_path(
                        val_index_path, project_root_path
                    ),
                },
                "source_artifacts": {
                    "split_manifest_path": project_relative_path(
                        split_manifest_path, project_root_path
                    ),
                    "sequence_manifest_path": project_relative_path(
                        sequence_manifest_path, project_root_path
                    ),
                    "target_alignment_registry_path": project_relative_path(
                        target_alignment_path, project_root_path
                    ),
                },
            }
        )

    dataset_registry_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_registry_path.write_text(
        json.dumps(registry_entries, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return T18OutputArtifacts(
        dataset_registry_path=dataset_registry_path,
        train_input_paths=train_input_paths,
        val_input_paths=val_input_paths,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build T-18 NumPy dataset arrays and the dataset registry."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--dataset-registry-output",
        type=Path,
        default=default_dataset_registry_output(),
    )
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
    artifacts = build_t18_outputs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_registry_output=args.dataset_registry_output,
        project_root=args.project_root,
        lbws=ALLOWED_CPD_LBWS if args.lbws is None else tuple(args.lbws),
    )
    print(
        "Wrote dataset registry and "
        f"{len(artifacts.train_input_paths)} train-input arrays plus "
        f"{len(artifacts.val_input_paths)} validation-input arrays."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
