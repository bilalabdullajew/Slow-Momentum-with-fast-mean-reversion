from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.datasets.join_and_split import project_relative_path
from lstm_cpd.datasets.registry import T18_SEQUENCE_INDEX_HEADER
from lstm_cpd.datasets.sequences import (
    SEQUENCE_LENGTH,
    SPLIT_VALIDATION,
    T17TargetAlignmentRow,
    load_sequence_manifest_csv,
    load_target_alignment_registry_csv,
)
from lstm_cpd.features.returns import load_canonical_daily_close_csv, serialize_optional_float
from lstm_cpd.features.volatility import (
    default_canonical_manifest_input,
    load_canonical_daily_close_manifest,
)
from lstm_cpd.model_source import (
    ResolvedModelSource,
    default_best_candidate_path,
    default_best_config_path,
    resolve_selected_model_source,
)
from lstm_cpd.training.train_candidate import load_dataset_registry_entry


TRADING_DAYS_PER_YEAR = 252
DEFAULT_RAW_VALIDATION_RETURNS_OUTPUT = "artifacts/evaluation/raw_validation_returns.csv"
DEFAULT_RAW_VALIDATION_METRICS_OUTPUT = "artifacts/evaluation/raw_validation_metrics.json"
DEFAULT_RESCALED_VALIDATION_RETURNS_OUTPUT = (
    "artifacts/evaluation/rescaled_validation_returns.csv"
)
DEFAULT_RESCALED_VALIDATION_METRICS_OUTPUT = (
    "artifacts/evaluation/rescaled_validation_metrics.json"
)
DEFAULT_EVALUATION_REPORT_OUTPUT = "artifacts/evaluation/evaluation_report.md"
VALIDATION_RETURNS_HEADER = ("return_timestamp", "asset_count", "portfolio_return")
_VALID_RUN_TYPES = {"candidate", "smoke", "search_summary", "inference", "evaluation"}


@dataclass(frozen=True)
class SequenceIndexRow:
    array_row_index: int
    sequence_id: str
    asset_id: str
    lbw: int
    split: str
    start_timestamp: str
    end_timestamp: str
    start_timeline_index: int
    end_timeline_index: int


@dataclass(frozen=True)
class ValidationReturnRow:
    return_timestamp: str
    asset_count: int
    portfolio_return: float

    def to_csv_row(self) -> dict[str, str]:
        return {
            "return_timestamp": self.return_timestamp,
            "asset_count": str(self.asset_count),
            "portfolio_return": serialize_optional_float(self.portfolio_return),
        }


@dataclass(frozen=True)
class ValidationEvaluationArtifacts:
    raw_validation_returns_path: Path
    raw_validation_metrics_path: Path
    rescaled_validation_returns_path: Path
    rescaled_validation_metrics_path: Path
    evaluation_report_path: Path
    model_source: ResolvedModelSource
    daily_observation_count: int


def default_raw_validation_returns_output() -> Path:
    return default_project_root() / DEFAULT_RAW_VALIDATION_RETURNS_OUTPUT


def default_raw_validation_metrics_output() -> Path:
    return default_project_root() / DEFAULT_RAW_VALIDATION_METRICS_OUTPUT


def default_rescaled_validation_returns_output() -> Path:
    return default_project_root() / DEFAULT_RESCALED_VALIDATION_RETURNS_OUTPUT


def default_rescaled_validation_metrics_output() -> Path:
    return default_project_root() / DEFAULT_RESCALED_VALIDATION_METRICS_OUTPUT


def default_evaluation_report_output() -> Path:
    return default_project_root() / DEFAULT_EVALUATION_REPORT_OUTPUT


def _resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _load_sequence_index_csv(path: Path | str) -> list[SequenceIndexRow]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != T18_SEQUENCE_INDEX_HEADER:
            raise ValueError(f"Sequence-index header mismatch: {csv_path}")
        rows: list[SequenceIndexRow] = []
        seen_indices: set[int] = set()
        for row in reader:
            array_row_index = int(row["array_row_index"])
            if array_row_index in seen_indices:
                raise ValueError(f"Duplicate array_row_index in {csv_path}: {array_row_index}")
            seen_indices.add(array_row_index)
            rows.append(
                SequenceIndexRow(
                    array_row_index=array_row_index,
                    sequence_id=row["sequence_id"],
                    asset_id=row["asset_id"],
                    lbw=int(row["lbw"]),
                    split=row["split"],
                    start_timestamp=row["start_timestamp"],
                    end_timestamp=row["end_timestamp"],
                    start_timeline_index=int(row["start_timeline_index"]),
                    end_timeline_index=int(row["end_timeline_index"]),
                )
            )
    return sorted(rows, key=lambda row: row.array_row_index)


def _group_target_alignment_rows(
    rows: Sequence[T17TargetAlignmentRow],
) -> dict[str, list[T17TargetAlignmentRow]]:
    grouped: dict[str, list[T17TargetAlignmentRow]] = {}
    for row in rows:
        grouped.setdefault(row.sequence_id, []).append(row)
    for sequence_id, grouped_rows in grouped.items():
        grouped[sequence_id] = sorted(grouped_rows, key=lambda row: row.step_index)
        if len(grouped[sequence_id]) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Target-alignment sequence {sequence_id} has {len(grouped[sequence_id])} "
                f"rows, expected {SEQUENCE_LENGTH}"
            )
    return grouped


def _build_canonical_timestamp_index(
    *,
    canonical_manifest_input: Path | str,
    project_root: Path,
) -> dict[str, tuple[str, ...]]:
    timestamp_index: dict[str, tuple[str, ...]] = {}
    for manifest_record in load_canonical_daily_close_manifest(canonical_manifest_input):
        canonical_rows = load_canonical_daily_close_csv(
            project_root / manifest_record.canonical_csv_path,
            expected_asset_id=manifest_record.asset_id,
        )
        timestamp_index[manifest_record.asset_id] = tuple(row.timestamp for row in canonical_rows)
    return timestamp_index


def _write_csv(
    path: Path | str,
    *,
    header: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _write_json(path: Path | str, payload: object) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return json_path


def build_validation_metrics(
    daily_returns: Sequence[float],
) -> dict[str, float | None]:
    if not daily_returns:
        raise ValueError("Validation return series must not be empty")
    returns_array = np.asarray(daily_returns, dtype=np.float64)
    if returns_array.ndim != 1:
        raise ValueError("Validation return series must be one-dimensional")
    if not np.all(np.isfinite(returns_array)):
        raise ValueError("Validation return series must contain only finite values")

    annualized_return = float(np.mean(returns_array) * TRADING_DAYS_PER_YEAR)
    annualized_volatility = float(
        np.std(returns_array, ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    downside_returns = np.minimum(returns_array, 0.0)
    annualized_downside_deviation = float(
        np.sqrt(np.mean(np.square(downside_returns))) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    cumulative_returns = np.cumprod(1.0 + returns_array)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns / running_max - 1.0
    maximum_drawdown = float(np.min(drawdowns))
    sharpe_ratio = (
        None
        if annualized_volatility == 0.0
        else float(annualized_return / annualized_volatility)
    )
    sortino_ratio = (
        None
        if annualized_downside_deviation == 0.0
        else float(annualized_return / annualized_downside_deviation)
    )
    calmar_ratio = (
        None if maximum_drawdown == 0.0 else float(annualized_return / abs(maximum_drawdown))
    )
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "annualized_downside_deviation": annualized_downside_deviation,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": calmar_ratio,
        "percentage_positive_daily_returns": float(np.mean(returns_array > 0.0) * 100.0),
    }


def _render_evaluation_report(
    *,
    model_source: ResolvedModelSource,
    dataset_registry_path: Path,
    raw_returns_path: Path,
    raw_metrics_path: Path,
    raw_metrics: dict[str, object],
    rescaled_returns_path: Path,
    rescaled_metrics_path: Path,
    rescaled_metrics: dict[str, object],
    project_root: Path,
) -> str:
    raw_returns_text = project_relative_path(raw_returns_path, project_root)
    raw_metrics_text = project_relative_path(raw_metrics_path, project_root)
    rescaled_returns_text = project_relative_path(rescaled_returns_path, project_root)
    rescaled_metrics_text = project_relative_path(rescaled_metrics_path, project_root)
    model_path_text = project_relative_path(model_source.model_path, project_root)
    dataset_registry_text = project_relative_path(dataset_registry_path, project_root)
    return "\n".join(
        [
            "# Validation Evaluation Report",
            "",
            f"- candidate_id: `{model_source.candidate_id}`",
            f"- lbw: `{model_source.lbw}`",
            f"- model_path: `{model_path_text}`",
            f"- dataset_registry: `{dataset_registry_text}`",
            "",
            "## Raw Validation Metrics",
            "",
            f"- annualized_return: `{raw_metrics['annualized_return']}`",
            f"- annualized_volatility: `{raw_metrics['annualized_volatility']}`",
            f"- annualized_downside_deviation: `{raw_metrics['annualized_downside_deviation']}`",
            f"- sharpe_ratio: `{raw_metrics['sharpe_ratio']}`",
            f"- sortino_ratio: `{raw_metrics['sortino_ratio']}`",
            f"- maximum_drawdown: `{raw_metrics['maximum_drawdown']}`",
            f"- calmar_ratio: `{raw_metrics['calmar_ratio']}`",
            f"- percentage_positive_daily_returns: `{raw_metrics['percentage_positive_daily_returns']}`",
            "",
            "## Rescaled Validation Metrics",
            "",
            f"- annualized_return: `{rescaled_metrics['annualized_return']}`",
            f"- annualized_volatility: `{rescaled_metrics['annualized_volatility']}`",
            f"- annualized_downside_deviation: `{rescaled_metrics['annualized_downside_deviation']}`",
            f"- sharpe_ratio: `{rescaled_metrics['sharpe_ratio']}`",
            f"- sortino_ratio: `{rescaled_metrics['sortino_ratio']}`",
            f"- maximum_drawdown: `{rescaled_metrics['maximum_drawdown']}`",
            f"- calmar_ratio: `{rescaled_metrics['calmar_ratio']}`",
            f"- percentage_positive_daily_returns: `{rescaled_metrics['percentage_positive_daily_returns']}`",
            f"- rescaling_factor: `{rescaled_metrics['rescaling_factor']}`",
            "",
            "## Artifacts",
            "",
            f"- raw_validation_returns: `{raw_returns_text}`",
            f"- raw_validation_metrics: `{raw_metrics_text}`",
            f"- rescaled_validation_returns: `{rescaled_returns_text}`",
            f"- rescaled_validation_metrics: `{rescaled_metrics_text}`",
            "",
            "## Scope Note",
            "",
            "FTMO evaluation is model-faithful but not a literal reproduction of the paper's "
            "50-futures universe.",
            "",
        ]
    )


def run_validation_evaluation(
    *,
    best_candidate_path: Path | str = default_best_candidate_path(),
    best_config_path: Path | str = default_best_config_path(),
    model_path: Path | str | None = None,
    candidate_config_path: Path | str | None = None,
    dataset_registry_path: Path | str | None = None,
    canonical_manifest_input: Path | str = default_canonical_manifest_input(),
    raw_validation_returns_output: Path | str = default_raw_validation_returns_output(),
    raw_validation_metrics_output: Path | str = default_raw_validation_metrics_output(),
    rescaled_validation_returns_output: Path | str = default_rescaled_validation_returns_output(),
    rescaled_validation_metrics_output: Path | str = default_rescaled_validation_metrics_output(),
    evaluation_report_output: Path | str = default_evaluation_report_output(),
    project_root: Path | str | None = None,
) -> ValidationEvaluationArtifacts:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    model_source = resolve_selected_model_source(
        best_candidate_path=best_candidate_path,
        best_config_path=best_config_path,
        model_path=model_path,
        candidate_config_path=candidate_config_path,
        dataset_registry_path=dataset_registry_path,
        project_root=project_root_path,
    )
    dataset_entry = load_dataset_registry_entry(
        model_source.dataset_registry_path,
        lbw=model_source.lbw,
    )

    val_inputs = np.load(project_root_path / dataset_entry.val_inputs_path).astype(np.float32)
    val_targets = np.load(project_root_path / dataset_entry.val_target_scale_path).astype(np.float32)
    if val_inputs.shape[0] != dataset_entry.val_sequence_count:
        raise ValueError(
            f"Validation input count mismatch: {val_inputs.shape[0]} != "
            f"{dataset_entry.val_sequence_count}"
        )
    if val_targets.shape != tuple(dataset_entry.val_target_shape):
        raise ValueError(
            f"Validation target shape mismatch: {val_targets.shape} != "
            f"{dataset_entry.val_target_shape}"
        )

    sequence_index_rows = _load_sequence_index_csv(
        project_root_path / dataset_entry.val_sequence_index_path
    )
    if len(sequence_index_rows) != val_inputs.shape[0]:
        raise ValueError(
            "Validation sequence-index count does not match validation tensor rows"
        )

    sequence_manifest_rows = {
        row.sequence_id: row
        for row in load_sequence_manifest_csv(
            project_root_path / dataset_entry.sequence_manifest_path,
            expected_lbw=model_source.lbw,
        )
        if row.split == SPLIT_VALIDATION
    }
    target_alignment_rows = _group_target_alignment_rows(
        load_target_alignment_registry_csv(
            project_root_path / dataset_entry.target_alignment_registry_path,
            expected_lbw=model_source.lbw,
        )
    )
    canonical_timestamp_index = _build_canonical_timestamp_index(
        canonical_manifest_input=canonical_manifest_input,
        project_root=project_root_path,
    )

    model = tf.keras.models.load_model(model_source.model_path, compile=False)
    model_outputs = model(tf.convert_to_tensor(val_inputs), training=False)
    positions = np.asarray(model_outputs, dtype=np.float32)
    expected_output_shape = (val_inputs.shape[0], SEQUENCE_LENGTH, 1)
    if positions.shape != expected_output_shape:
        raise ValueError(
            f"Validation model output shape mismatch: {positions.shape} != "
            f"{expected_output_shape}"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("Validation model outputs must be finite")

    portfolio_return_map: dict[str, list[float]] = {}
    for sequence_index_row in sequence_index_rows:
        if sequence_index_row.split != SPLIT_VALIDATION:
            raise ValueError(
                "Validation sequence index contains a non-validation split row: "
                f"{sequence_index_row.sequence_id}"
            )
        manifest_row = sequence_manifest_rows.get(sequence_index_row.sequence_id)
        if manifest_row is None:
            raise ValueError(
                "Validation sequence manifest is missing sequence_id: "
                f"{sequence_index_row.sequence_id}"
            )
        if (
            manifest_row.start_timestamp != sequence_index_row.start_timestamp
            or manifest_row.end_timestamp != sequence_index_row.end_timestamp
            or manifest_row.start_timeline_index != sequence_index_row.start_timeline_index
            or manifest_row.end_timeline_index != sequence_index_row.end_timeline_index
        ):
            raise ValueError(
                "Validation sequence index does not match the sequence manifest for "
                f"{sequence_index_row.sequence_id}"
            )
        target_rows = target_alignment_rows.get(sequence_index_row.sequence_id)
        if target_rows is None:
            raise ValueError(
                "Target-alignment registry is missing sequence_id: "
                f"{sequence_index_row.sequence_id}"
            )
        expected_target_scale = np.asarray(
            [row.target_scale for row in target_rows],
            dtype=np.float32,
        )
        if not np.allclose(
            val_targets[sequence_index_row.array_row_index],
            expected_target_scale,
            atol=1e-7,
        ):
            raise ValueError(
                "Validation target-scale tensor does not match target_alignment_registry "
                f"for {sequence_index_row.sequence_id}"
            )
        asset_timestamps = canonical_timestamp_index.get(sequence_index_row.asset_id)
        if asset_timestamps is None:
            raise ValueError(
                f"Canonical manifest is missing validation asset {sequence_index_row.asset_id}"
            )
        sequence_positions = positions[sequence_index_row.array_row_index, :, 0]
        sequence_realized_returns = sequence_positions * val_targets[
            sequence_index_row.array_row_index
        ]
        for step_index, target_row in enumerate(target_rows):
            if target_row.timeline_index + 1 >= len(asset_timestamps):
                raise ValueError(
                    f"Return timestamp index out of range for {target_row.asset_id} "
                    f"timeline_index={target_row.timeline_index}"
                )
            return_timestamp = asset_timestamps[target_row.timeline_index + 1]
            portfolio_return_map.setdefault(return_timestamp, []).append(
                float(sequence_realized_returns[step_index])
            )

    raw_return_rows = [
        ValidationReturnRow(
            return_timestamp=return_timestamp,
            asset_count=len(values),
            portfolio_return=float(np.mean(values)),
        )
        for return_timestamp, values in sorted(portfolio_return_map.items())
    ]
    raw_daily_returns = [row.portfolio_return for row in raw_return_rows]
    raw_metrics = build_validation_metrics(raw_daily_returns)
    raw_metrics_payload = {
        **raw_metrics,
        "selected_candidate_id": model_source.candidate_id,
        "selected_lbw": model_source.lbw,
        "daily_observation_count": len(raw_return_rows),
    }

    raw_returns_path = _write_csv(
        raw_validation_returns_output,
        header=VALIDATION_RETURNS_HEADER,
        rows=[row.to_csv_row() for row in raw_return_rows],
    )
    raw_metrics_path = _write_json(raw_validation_metrics_output, raw_metrics_payload)

    raw_annualized_volatility = float(raw_metrics["annualized_volatility"])
    if raw_annualized_volatility == 0.0:
        raise ValueError(
            "Cannot apply validation rescaling because raw annualized volatility is zero"
        )
    rescaling_factor = 0.15 / raw_annualized_volatility
    rescaled_return_rows = [
        ValidationReturnRow(
            return_timestamp=row.return_timestamp,
            asset_count=row.asset_count,
            portfolio_return=row.portfolio_return * rescaling_factor,
        )
        for row in raw_return_rows
    ]
    rescaled_daily_returns = [row.portfolio_return for row in rescaled_return_rows]
    rescaled_metrics = build_validation_metrics(rescaled_daily_returns)
    rescaled_metrics_payload = {
        **rescaled_metrics,
        "selected_candidate_id": model_source.candidate_id,
        "selected_lbw": model_source.lbw,
        "daily_observation_count": len(rescaled_return_rows),
        "rescaling_factor": rescaling_factor,
    }

    rescaled_returns_path = _write_csv(
        rescaled_validation_returns_output,
        header=VALIDATION_RETURNS_HEADER,
        rows=[row.to_csv_row() for row in rescaled_return_rows],
    )
    rescaled_metrics_path = _write_json(
        rescaled_validation_metrics_output,
        rescaled_metrics_payload,
    )
    report_path = Path(evaluation_report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_evaluation_report(
            model_source=model_source,
            dataset_registry_path=model_source.dataset_registry_path,
            raw_returns_path=raw_returns_path,
            raw_metrics_path=raw_metrics_path,
            raw_metrics=raw_metrics_payload,
            rescaled_returns_path=rescaled_returns_path,
            rescaled_metrics_path=rescaled_metrics_path,
            rescaled_metrics=rescaled_metrics_payload,
            project_root=project_root_path,
        ),
        encoding="utf-8",
    )

    return ValidationEvaluationArtifacts(
        raw_validation_returns_path=raw_returns_path,
        raw_validation_metrics_path=raw_metrics_path,
        rescaled_validation_returns_path=rescaled_returns_path,
        rescaled_validation_metrics_path=rescaled_metrics_path,
        evaluation_report_path=report_path,
        model_source=model_source,
        daily_observation_count=len(raw_return_rows),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate validation performance for the selected LBW model."
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
        "--raw-validation-returns-output",
        type=Path,
        default=default_raw_validation_returns_output(),
    )
    parser.add_argument(
        "--raw-validation-metrics-output",
        type=Path,
        default=default_raw_validation_metrics_output(),
    )
    parser.add_argument(
        "--rescaled-validation-returns-output",
        type=Path,
        default=default_rescaled_validation_returns_output(),
    )
    parser.add_argument(
        "--rescaled-validation-metrics-output",
        type=Path,
        default=default_rescaled_validation_metrics_output(),
    )
    parser.add_argument(
        "--evaluation-report-output",
        type=Path,
        default=default_evaluation_report_output(),
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = run_validation_evaluation(
        best_candidate_path=args.best_candidate,
        best_config_path=args.best_config,
        model_path=args.model_path,
        candidate_config_path=args.candidate_config_path,
        dataset_registry_path=args.dataset_registry_path,
        canonical_manifest_input=args.canonical_manifest_input,
        raw_validation_returns_output=args.raw_validation_returns_output,
        raw_validation_metrics_output=args.raw_validation_metrics_output,
        rescaled_validation_returns_output=args.rescaled_validation_returns_output,
        rescaled_validation_metrics_output=args.rescaled_validation_metrics_output,
        evaluation_report_output=args.evaluation_report_output,
        project_root=args.project_root,
    )
    print(
        "Wrote validation evaluation artifacts to "
        f"{artifacts.evaluation_report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
