from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import tensorflow as tf

from lstm_cpd.cpd.precompute import (
    ReturnsVolatilityRecord as CPDReturnsVolatilityRecord,
    build_cpd_feature_rows,
    fit_cpd_window,
)
from lstm_cpd.cpd.precompute_contract import CPDWindowInput, CPDWindowResult
from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.datasets.join_and_split import (
    CPDJoinRecord,
    MODEL_INPUT_COLUMNS,
    ReturnsVolatilityJoinRecord,
    T16JoinedFeatureRow,
    build_joined_feature_rows,
    project_relative_path,
)
from lstm_cpd.features.macd import build_macd_feature_rows
from lstm_cpd.features.normalized_returns import (
    ReturnsVolatilityRecord as NormalizedReturnsRecord,
    compute_normalized_return_features,
)
from lstm_cpd.features.returns import load_canonical_daily_close_csv, serialize_optional_float
from lstm_cpd.features.volatility import (
    build_returns_volatility_rows,
    default_canonical_manifest_input,
    load_canonical_daily_close_manifest,
    validate_canonical_alignment,
)
from lstm_cpd.features.winsorize import build_base_feature_rows, join_feature_rows
from lstm_cpd.model_source import (
    ResolvedModelSource,
    default_best_candidate_path,
    default_best_config_path,
    resolve_selected_model_source,
)


DEFAULT_LATEST_POSITIONS_OUTPUT = "artifacts/inference/latest_positions.csv"
DEFAULT_LATEST_SEQUENCE_MANIFEST_OUTPUT = "artifacts/inference/latest_sequence_manifest.csv"
LATEST_POSITIONS_HEADER = (
    "asset_id",
    "lbw",
    "signal_timestamp",
    "next_day_position",
    "candidate_id",
    "model_path",
)
LATEST_SEQUENCE_MANIFEST_HEADER = (
    "asset_id",
    "lbw",
    "sequence_start_timestamp",
    "sequence_end_timestamp",
    "row_count",
    "start_timeline_index",
    "end_timeline_index",
    "candidate_id",
    "model_path",
)


@dataclass(frozen=True)
class LatestPositionRow:
    asset_id: str
    lbw: int
    signal_timestamp: str
    next_day_position: float
    candidate_id: str
    model_path: str

    def to_csv_row(self) -> dict[str, str]:
        return {
            "asset_id": self.asset_id,
            "lbw": str(self.lbw),
            "signal_timestamp": self.signal_timestamp,
            "next_day_position": serialize_optional_float(self.next_day_position),
            "candidate_id": self.candidate_id,
            "model_path": self.model_path,
        }


@dataclass(frozen=True)
class LatestSequenceManifestRow:
    asset_id: str
    lbw: int
    sequence_start_timestamp: str
    sequence_end_timestamp: str
    row_count: int
    start_timeline_index: int
    end_timeline_index: int
    candidate_id: str
    model_path: str

    def to_csv_row(self) -> dict[str, str]:
        return {
            "asset_id": self.asset_id,
            "lbw": str(self.lbw),
            "sequence_start_timestamp": self.sequence_start_timestamp,
            "sequence_end_timestamp": self.sequence_end_timestamp,
            "row_count": str(self.row_count),
            "start_timeline_index": str(self.start_timeline_index),
            "end_timeline_index": str(self.end_timeline_index),
            "candidate_id": self.candidate_id,
            "model_path": self.model_path,
        }


@dataclass(frozen=True)
class OnlineInferenceArtifacts:
    latest_positions_path: Path
    latest_sequence_manifest_path: Path
    model_source: ResolvedModelSource
    asset_count: int


def default_latest_positions_output() -> Path:
    return default_project_root() / DEFAULT_LATEST_POSITIONS_OUTPUT


def default_latest_sequence_manifest_output() -> Path:
    return default_project_root() / DEFAULT_LATEST_SEQUENCE_MANIFEST_OUTPUT


def _returns_rows_to_normalized_records(
    returns_rows: Sequence[dict[str, str]],
) -> list[NormalizedReturnsRecord]:
    records: list[NormalizedReturnsRecord] = []
    for row in returns_rows:
        sigma_t_text = row["sigma_t"]
        records.append(
            NormalizedReturnsRecord(
                timestamp=row["timestamp"],
                asset_id=row["asset_id"],
                close_text=row["close"],
                close_value=float(row["close"]),
                sigma_t_text=sigma_t_text,
                sigma_t_value=None if sigma_t_text == "" else float(sigma_t_text),
            )
        )
    return records


def _returns_rows_to_join_records(
    returns_rows: Sequence[dict[str, str]],
) -> list[ReturnsVolatilityJoinRecord]:
    return [
        ReturnsVolatilityJoinRecord(
            timestamp=row["timestamp"],
            asset_id=row["asset_id"],
            arithmetic_return=(
                None if row["arithmetic_return"] == "" else float(row["arithmetic_return"])
            ),
            sigma_t=None if row["sigma_t"] == "" else float(row["sigma_t"]),
        )
        for row in returns_rows
    ]


def _returns_rows_to_cpd_records(
    returns_rows: Sequence[dict[str, str]],
) -> list[CPDReturnsVolatilityRecord]:
    return [
        CPDReturnsVolatilityRecord(
            timestamp=row["timestamp"],
            asset_id=row["asset_id"],
            arithmetic_return=(
                None if row["arithmetic_return"] == "" else float(row["arithmetic_return"])
            ),
        )
        for row in returns_rows
    ]


def _cpd_rows_to_join_records(
    cpd_rows: Sequence[dict[str, str]],
    *,
    expected_asset_id: str,
    expected_lbw: int,
) -> list[CPDJoinRecord]:
    join_rows: list[CPDJoinRecord] = []
    for row in cpd_rows:
        asset_id = row["asset_id"]
        lbw = int(row["lbw"])
        if asset_id != expected_asset_id:
            raise ValueError(f"CPD asset_id mismatch: {asset_id} != {expected_asset_id}")
        if lbw != expected_lbw:
            raise ValueError(f"CPD lbw mismatch: {lbw} != {expected_lbw}")
        join_rows.append(
            CPDJoinRecord(
                timestamp=row["timestamp"],
                asset_id=asset_id,
                lbw=lbw,
                nu=None if row["nu"] == "" else float(row["nu"]),
                gamma=None if row["gamma"] == "" else float(row["gamma"]),
                status=row["status"],
            )
        )
    return join_rows


def _select_latest_causal_sequence(
    joined_rows: Sequence[T16JoinedFeatureRow],
) -> list[T16JoinedFeatureRow]:
    if not joined_rows:
        return []
    trailing_start_index = len(joined_rows) - 1
    for row_index in range(len(joined_rows) - 2, -1, -1):
        current_row = joined_rows[row_index]
        next_row = joined_rows[row_index + 1]
        if current_row.timeline_index + 1 != next_row.timeline_index:
            break
        trailing_start_index = row_index
    trailing_rows = list(joined_rows[trailing_start_index:])
    if len(trailing_rows) < 63:
        return []
    return trailing_rows[-63:]


def _write_csv(
    output_path: Path | str,
    *,
    header: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> Path:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def run_online_inference(
    *,
    best_candidate_path: Path | str = default_best_candidate_path(),
    best_config_path: Path | str = default_best_config_path(),
    model_path: Path | str | None = None,
    candidate_config_path: Path | str | None = None,
    dataset_registry_path: Path | str | None = None,
    canonical_manifest_input: Path | str = default_canonical_manifest_input(),
    latest_positions_output: Path | str = default_latest_positions_output(),
    latest_sequence_manifest_output: Path | str = default_latest_sequence_manifest_output(),
    project_root: Path | str | None = None,
    fit_window_fn: Callable[[CPDWindowInput], CPDWindowResult] | None = None,
) -> OnlineInferenceArtifacts:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    resolved_model_source = resolve_selected_model_source(
        best_candidate_path=best_candidate_path,
        best_config_path=best_config_path,
        model_path=model_path,
        candidate_config_path=candidate_config_path,
        dataset_registry_path=dataset_registry_path,
        project_root=project_root_path,
    )
    fit_window = fit_window_fn if fit_window_fn is not None else fit_cpd_window

    manifest_records = load_canonical_daily_close_manifest(canonical_manifest_input)
    model_path_text = project_relative_path(
        resolved_model_source.model_path,
        project_root_path,
    )
    inference_inputs: list[np.ndarray] = []
    sequence_rows: list[LatestSequenceManifestRow] = []
    coverage_failures: list[str] = []

    for manifest_record in manifest_records:
        canonical_csv_path = project_root_path / manifest_record.canonical_csv_path
        canonical_rows = load_canonical_daily_close_csv(
            canonical_csv_path,
            expected_asset_id=manifest_record.asset_id,
        )
        validate_canonical_alignment(manifest_record, canonical_rows)

        returns_rows = build_returns_volatility_rows(canonical_rows)
        normalized_rows = compute_normalized_return_features(
            _returns_rows_to_normalized_records(returns_rows)
        )
        macd_rows = build_macd_feature_rows(canonical_rows)
        base_rows = build_base_feature_rows(join_feature_rows(normalized_rows, macd_rows))
        returns_join_rows = _returns_rows_to_join_records(returns_rows)
        cpd_rows = build_cpd_feature_rows(
            _returns_rows_to_cpd_records(returns_rows),
            lbw=resolved_model_source.lbw,
            fit_window_fn=fit_window,
        )
        joined_rows = build_joined_feature_rows(
            base_rows=base_rows,
            returns_rows=returns_join_rows,
            cpd_rows=_cpd_rows_to_join_records(
                cpd_rows,
                expected_asset_id=manifest_record.asset_id,
                expected_lbw=resolved_model_source.lbw,
            ),
            lbw=resolved_model_source.lbw,
        )
        latest_sequence = _select_latest_causal_sequence(joined_rows)
        if len(latest_sequence) != 63:
            coverage_failures.append(
                f"{manifest_record.asset_id}: trailing_causal_sequence_length={len(latest_sequence)}"
            )
            continue
        inference_inputs.append(
            np.asarray([row.model_inputs for row in latest_sequence], dtype=np.float32)
        )
        sequence_rows.append(
            LatestSequenceManifestRow(
                asset_id=manifest_record.asset_id,
                lbw=resolved_model_source.lbw,
                sequence_start_timestamp=latest_sequence[0].timestamp,
                sequence_end_timestamp=latest_sequence[-1].timestamp,
                row_count=len(latest_sequence),
                start_timeline_index=latest_sequence[0].timeline_index,
                end_timeline_index=latest_sequence[-1].timeline_index,
                candidate_id=resolved_model_source.candidate_id,
                model_path=model_path_text,
            )
        )

    if coverage_failures:
        raise ValueError(
            "Online inference coverage failure for admitted assets: "
            + "; ".join(coverage_failures)
        )

    model = tf.keras.models.load_model(
        resolved_model_source.model_path,
        compile=False,
    )
    model_outputs = model(
        tf.convert_to_tensor(np.stack(inference_inputs, axis=0), dtype=tf.float32),
        training=False,
    )
    output_array = np.asarray(model_outputs, dtype=np.float32)
    expected_shape = (len(inference_inputs), 63, 1)
    if output_array.shape != expected_shape:
        raise ValueError(
            f"Model output shape mismatch: {output_array.shape} != {expected_shape}"
        )
    if not np.all(np.isfinite(output_array)):
        raise ValueError("Model outputs must be finite during online inference")

    position_rows = [
        LatestPositionRow(
            asset_id=sequence_row.asset_id,
            lbw=sequence_row.lbw,
            signal_timestamp=sequence_row.sequence_end_timestamp,
            next_day_position=float(output_array[row_index, -1, 0]),
            candidate_id=sequence_row.candidate_id,
            model_path=sequence_row.model_path,
        )
        for row_index, sequence_row in enumerate(sequence_rows)
    ]

    positions_path = _write_csv(
        latest_positions_output,
        header=LATEST_POSITIONS_HEADER,
        rows=[row.to_csv_row() for row in position_rows],
    )
    sequence_manifest_path = _write_csv(
        latest_sequence_manifest_output,
        header=LATEST_SEQUENCE_MANIFEST_HEADER,
        rows=[row.to_csv_row() for row in sequence_rows],
    )
    return OnlineInferenceArtifacts(
        latest_positions_path=positions_path,
        latest_sequence_manifest_path=sequence_manifest_path,
        model_source=resolved_model_source,
        asset_count=len(position_rows),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run causal online inference for the selected LBW model."
    )
    parser.add_argument("--best-candidate", type=Path, default=default_best_candidate_path())
    parser.add_argument("--best-config", type=Path, default=default_best_config_path())
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--candidate-config-path", type=Path)
    parser.add_argument("--dataset-registry-path", type=Path)
    parser.add_argument(
        "--canonical-manifest-input",
        type=Path,
        default=default_canonical_manifest_input(),
    )
    parser.add_argument(
        "--latest-positions-output",
        type=Path,
        default=default_latest_positions_output(),
    )
    parser.add_argument(
        "--latest-sequence-manifest-output",
        type=Path,
        default=default_latest_sequence_manifest_output(),
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = run_online_inference(
        best_candidate_path=args.best_candidate,
        best_config_path=args.best_config,
        model_path=args.model_path,
        candidate_config_path=args.candidate_config_path,
        dataset_registry_path=args.dataset_registry_path,
        canonical_manifest_input=args.canonical_manifest_input,
        latest_positions_output=args.latest_positions_output,
        latest_sequence_manifest_output=args.latest_sequence_manifest_output,
        project_root=args.project_root,
    )
    print(
        "Wrote online inference artifacts for "
        f"{artifacts.asset_count} assets to {artifacts.latest_positions_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
