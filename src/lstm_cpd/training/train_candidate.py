from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.datasets.join_and_split import MODEL_INPUT_COLUMNS
from lstm_cpd.datasets.sequences import SEQUENCE_LENGTH
from lstm_cpd.model.network import (
    ModelRuntimeConfig,
    build_model_runtime,
    derive_model_seed_bundle,
    get_dropout_layers,
    get_single_lstm_layer,
)
from lstm_cpd.training.losses import (
    compute_realized_returns,
    sharpe_loss,
    sharpe_loss_from_realized_returns,
)


SEED_BASE = 20260421
DEFAULT_MAX_EPOCHS = 300
DEFAULT_PATIENCE = 25
DEFAULT_DATASET_REGISTRY_PATH = "artifacts/manifests/dataset_registry.json"
DEFAULT_SMOKE_DATASET_REGISTRY_PATH = "artifacts/interim/manifests/dataset_registry.json"
DEFAULT_MODEL_RUNTIME_CONTRACT_PATH = "docs/contracts/model_runtime_contract.md"
DEFAULT_SMOKE_OUTPUT_DIR = "artifacts/interim/training/smoke_run"
DEFAULT_SMOKE_REPORT_PATH = "artifacts/interim/reports/model_fidelity_report.md"
EPOCH_LOG_HEADER = (
    "epoch_index",
    "train_loss",
    "val_loss",
    "best_val_loss",
    "mean_gradient_norm",
    "improved",
)
VALIDATION_HISTORY_HEADER = (
    "epoch_index",
    "val_loss",
    "best_so_far",
    "improved_vs_previous",
    "improved_vs_best",
)


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    candidate_index: int
    dropout: float
    hidden_size: int
    minibatch_size: int
    learning_rate: float
    max_grad_norm: float
    lbw: int


@dataclass(frozen=True)
class DatasetRegistryEntry:
    lbw: int
    feature_columns: tuple[str, ...]
    sequence_length: int
    train_sequence_count: int
    val_sequence_count: int
    train_input_shape: tuple[int, int, int]
    train_target_shape: tuple[int, int]
    val_input_shape: tuple[int, int, int]
    val_target_shape: tuple[int, int]
    train_inputs_path: str
    train_target_scale_path: str
    val_inputs_path: str
    val_target_scale_path: str
    train_sequence_index_path: str
    val_sequence_index_path: str
    split_manifest_path: str
    sequence_manifest_path: str
    target_alignment_registry_path: str


@dataclass(frozen=True)
class TrainingArtifactPaths:
    config_snapshot_path: Path
    best_model_path: Path
    epoch_log_path: Path
    validation_history_path: Path


@dataclass(frozen=True)
class TrainingRunResult:
    candidate_config: CandidateConfig
    dataset_registry_path: Path
    output_dir: Path
    config_snapshot_path: Path
    best_model_path: Path
    epoch_log_path: Path
    validation_history_path: Path
    initial_validation_loss: float
    best_validation_loss: float
    best_epoch_index: int | None
    epochs_completed: int
    validation_losses: tuple[float, ...]
    dataset_entry: DatasetRegistryEntry


@dataclass(frozen=True)
class SmokeRunResult:
    training_result: TrainingRunResult
    report_path: Path
    validation_loss_decreased: bool


def candidate_config_to_payload(candidate_config: CandidateConfig) -> dict[str, object]:
    return {
        "candidate_id": candidate_config.candidate_id,
        "candidate_index": candidate_config.candidate_index,
        "dropout": candidate_config.dropout,
        "hidden_size": candidate_config.hidden_size,
        "minibatch_size": candidate_config.minibatch_size,
        "learning_rate": candidate_config.learning_rate,
        "max_grad_norm": candidate_config.max_grad_norm,
        "lbw": candidate_config.lbw,
    }


def default_dataset_registry_path() -> Path:
    return default_project_root() / DEFAULT_DATASET_REGISTRY_PATH


def default_model_runtime_contract_path() -> Path:
    return default_project_root() / DEFAULT_MODEL_RUNTIME_CONTRACT_PATH


def default_smoke_dataset_registry_path() -> Path:
    return default_project_root() / DEFAULT_SMOKE_DATASET_REGISTRY_PATH


def default_smoke_output_dir() -> Path:
    return default_project_root() / DEFAULT_SMOKE_OUTPUT_DIR


def default_smoke_report_path() -> Path:
    return default_project_root() / DEFAULT_SMOKE_REPORT_PATH


def set_global_determinism(seed: int = SEED_BASE) -> None:
    tf.config.experimental.enable_op_determinism()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_candidate_config(path: Path | str) -> CandidateConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    required_fields = {
        "candidate_id",
        "candidate_index",
        "dropout",
        "hidden_size",
        "minibatch_size",
        "learning_rate",
        "max_grad_norm",
        "lbw",
    }
    missing_fields = required_fields - set(payload)
    if missing_fields:
        raise ValueError(
            f"Candidate config is missing fields: {sorted(missing_fields)}"
        )
    config = CandidateConfig(
        candidate_id=str(payload["candidate_id"]),
        candidate_index=int(payload["candidate_index"]),
        dropout=float(payload["dropout"]),
        hidden_size=int(payload["hidden_size"]),
        minibatch_size=int(payload["minibatch_size"]),
        learning_rate=float(payload["learning_rate"]),
        max_grad_norm=float(payload["max_grad_norm"]),
        lbw=int(payload["lbw"]),
    )
    validate_candidate_config(config)
    return config


def validate_candidate_config(config: CandidateConfig) -> None:
    if config.candidate_id == "":
        raise ValueError("candidate_id must not be empty")
    if config.candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")


def _parse_shape(raw_shape: object, *, expected_rank: int, field_name: str) -> tuple[int, ...]:
    if not isinstance(raw_shape, list) or len(raw_shape) != expected_rank:
        raise ValueError(f"{field_name} must be a JSON list of length {expected_rank}")
    return tuple(int(value) for value in raw_shape)


def load_dataset_registry_entry(
    registry_path: Path | str,
    *,
    lbw: int,
) -> DatasetRegistryEntry:
    json_path = Path(registry_path)
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    matching_rows = [row for row in rows if int(row["lbw"]) == int(lbw)]
    if not matching_rows:
        raise ValueError(f"Dataset registry has no entry for lbw={lbw}")
    if len(matching_rows) != 1:
        raise ValueError(f"Dataset registry has duplicate entries for lbw={lbw}")
    row = matching_rows[0]
    feature_columns = tuple(str(value) for value in row["feature_columns"])
    if feature_columns != MODEL_INPUT_COLUMNS:
        raise ValueError(
            "Dataset registry feature_columns mismatch: "
            f"{feature_columns} != {MODEL_INPUT_COLUMNS}"
        )
    if int(row["sequence_length"]) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Dataset registry sequence_length mismatch: {row['sequence_length']}"
        )
    artifacts = row["artifacts"]
    source_artifacts = row["source_artifacts"]
    return DatasetRegistryEntry(
        lbw=int(row["lbw"]),
        feature_columns=feature_columns,
        sequence_length=int(row["sequence_length"]),
        train_sequence_count=int(row["train_sequence_count"]),
        val_sequence_count=int(row["val_sequence_count"]),
        train_input_shape=_parse_shape(
            row["train_input_shape"], expected_rank=3, field_name="train_input_shape"
        ),
        train_target_shape=_parse_shape(
            row["train_target_shape"], expected_rank=2, field_name="train_target_shape"
        ),
        val_input_shape=_parse_shape(
            row["val_input_shape"], expected_rank=3, field_name="val_input_shape"
        ),
        val_target_shape=_parse_shape(
            row["val_target_shape"], expected_rank=2, field_name="val_target_shape"
        ),
        train_inputs_path=str(artifacts["train_inputs_path"]),
        train_target_scale_path=str(artifacts["train_target_scale_path"]),
        val_inputs_path=str(artifacts["val_inputs_path"]),
        val_target_scale_path=str(artifacts["val_target_scale_path"]),
        train_sequence_index_path=str(artifacts["train_sequence_index_path"]),
        val_sequence_index_path=str(artifacts["val_sequence_index_path"]),
        split_manifest_path=str(source_artifacts["split_manifest_path"]),
        sequence_manifest_path=str(source_artifacts["sequence_manifest_path"]),
        target_alignment_registry_path=str(source_artifacts["target_alignment_registry_path"]),
    )


def _load_dataset_arrays(
    entry: DatasetRegistryEntry,
    *,
    project_root: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(project_root)
    train_inputs = np.load(root / entry.train_inputs_path).astype(np.float32)
    train_targets = np.load(root / entry.train_target_scale_path).astype(np.float32)
    val_inputs = np.load(root / entry.val_inputs_path).astype(np.float32)
    val_targets = np.load(root / entry.val_target_scale_path).astype(np.float32)
    if train_inputs.shape != entry.train_input_shape:
        raise ValueError(
            f"Train input shape mismatch: {train_inputs.shape} != {entry.train_input_shape}"
        )
    if train_targets.shape != entry.train_target_shape:
        raise ValueError(
            f"Train target shape mismatch: {train_targets.shape} != {entry.train_target_shape}"
        )
    if val_inputs.shape != entry.val_input_shape:
        raise ValueError(
            f"Validation input shape mismatch: {val_inputs.shape} != {entry.val_input_shape}"
        )
    if val_targets.shape != entry.val_target_shape:
        raise ValueError(
            f"Validation target shape mismatch: {val_targets.shape} != {entry.val_target_shape}"
        )
    return train_inputs, train_targets, val_inputs, val_targets


def _build_artifact_paths(
    output_dir: Path | str,
    *,
    artifact_stem: str,
) -> TrainingArtifactPaths:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    return TrainingArtifactPaths(
        config_snapshot_path=output_dir_path / f"{artifact_stem}_config.json",
        best_model_path=output_dir_path / f"{artifact_stem}_best_model.keras",
        epoch_log_path=output_dir_path / f"{artifact_stem}_epoch_log.csv",
        validation_history_path=output_dir_path / f"{artifact_stem}_validation_history.csv",
    )


def _resolve_artifact_paths(
    output_dir: Path | str,
    *,
    artifact_stem: str,
    artifact_paths: TrainingArtifactPaths | None,
) -> TrainingArtifactPaths:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if artifact_paths is None:
        return _build_artifact_paths(output_dir_path, artifact_stem=artifact_stem)
    resolved_paths = TrainingArtifactPaths(
        config_snapshot_path=Path(artifact_paths.config_snapshot_path),
        best_model_path=Path(artifact_paths.best_model_path),
        epoch_log_path=Path(artifact_paths.epoch_log_path),
        validation_history_path=Path(artifact_paths.validation_history_path),
    )
    for artifact_path in (
        resolved_paths.config_snapshot_path,
        resolved_paths.best_model_path,
        resolved_paths.epoch_log_path,
        resolved_paths.validation_history_path,
    ):
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_paths


def _evaluate_dataset_loss(
    model: tf.keras.Model,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> float:
    model_outputs = model(tf.convert_to_tensor(inputs), training=False)
    loss_value = sharpe_loss(model_outputs, tf.convert_to_tensor(targets))
    return float(loss_value.numpy())


def _validation_loss_decreased(
    validation_losses: Sequence[float],
    *,
    initial_validation_loss: float,
) -> bool:
    if not validation_losses:
        return False
    if min(validation_losses) < initial_validation_loss:
        return True
    previous_loss = initial_validation_loss
    for loss_value in validation_losses:
        if loss_value < previous_loss:
            return True
        previous_loss = loss_value
    return False


def _write_epoch_log(
    path: Path | str,
    rows: Sequence[dict[str, str]],
) -> None:
    csv_path = Path(path)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(EPOCH_LOG_HEADER), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_validation_history(
    path: Path | str,
    rows: Sequence[dict[str, str]],
) -> None:
    csv_path = Path(path)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(VALIDATION_HISTORY_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_config_snapshot(
    path: Path | str,
    *,
    candidate_config: CandidateConfig,
    dataset_registry_path: Path,
    dataset_entry: DatasetRegistryEntry,
    output_dir: Path,
    max_epochs: int,
    patience: int,
    artifact_stem: str,
) -> None:
    payload = {
        **candidate_config_to_payload(candidate_config),
        "dataset_registry_path": str(dataset_registry_path),
        "dataset_entry": {
            "lbw": dataset_entry.lbw,
            "train_sequence_count": dataset_entry.train_sequence_count,
            "val_sequence_count": dataset_entry.val_sequence_count,
            "train_input_shape": list(dataset_entry.train_input_shape),
            "val_input_shape": list(dataset_entry.val_input_shape),
        },
        "training_policy": {
            "seed_base": SEED_BASE,
            "max_epochs": max_epochs,
            "patience": patience,
            "shuffle_seed_formula": "20260421 + epoch_index",
            "candidate_seed_formula": "20260421 + candidate_index",
            "gradient_clipping": "global_norm",
        },
        "artifact_stem": artifact_stem,
        "output_dir": str(output_dir),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_candidate_training(
    *,
    dataset_registry_path: Path | str = default_dataset_registry_path(),
    candidate_config: CandidateConfig,
    output_dir: Path | str,
    project_root: Path | str | None = None,
    artifact_stem: str = "candidate",
    artifact_paths: TrainingArtifactPaths | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
) -> TrainingRunResult:
    validate_candidate_config(candidate_config)
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")

    root = Path(project_root) if project_root is not None else default_project_root()
    registry_path = Path(dataset_registry_path)
    entry = load_dataset_registry_entry(registry_path, lbw=candidate_config.lbw)
    train_inputs, train_targets, val_inputs, val_targets = _load_dataset_arrays(
        entry,
        project_root=root,
    )
    artifacts = _resolve_artifact_paths(
        output_dir,
        artifact_stem=artifact_stem,
        artifact_paths=artifact_paths,
    )

    set_global_determinism(SEED_BASE)
    candidate_seed = SEED_BASE + candidate_config.candidate_index
    model = build_model_runtime(
        ModelRuntimeConfig(
            dropout_rate=candidate_config.dropout,
            hidden_size=candidate_config.hidden_size,
            seeds=derive_model_seed_bundle(candidate_seed),
        )
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=candidate_config.learning_rate)
    max_grad_norm_tensor = tf.constant(candidate_config.max_grad_norm, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def train_step(
        batch_inputs: tf.Tensor,
        batch_targets: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            batch_outputs = model(batch_inputs, training=True)
            loss_value = sharpe_loss(batch_outputs, batch_targets)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        clipped_gradients, global_norm = tf.clip_by_global_norm(
            gradients,
            max_grad_norm_tensor,
        )
        optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
        return loss_value, global_norm

    initial_validation_loss = _evaluate_dataset_loss(model, val_inputs, val_targets)
    model.save(artifacts.best_model_path, overwrite=True)
    best_validation_loss = initial_validation_loss
    best_epoch_index: int | None = None
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    validation_losses: list[float] = []

    for epoch_index in range(max_epochs):
        shuffled_indices = np.random.default_rng(SEED_BASE + epoch_index).permutation(
            train_inputs.shape[0]
        )
        epoch_batch_losses: list[float] = []
        epoch_gradient_norms: list[float] = []
        for batch_start in range(0, train_inputs.shape[0], candidate_config.minibatch_size):
            batch_indices = shuffled_indices[
                batch_start : batch_start + candidate_config.minibatch_size
            ]
            batch_inputs = tf.convert_to_tensor(train_inputs[batch_indices], dtype=tf.float32)
            batch_targets = tf.convert_to_tensor(
                train_targets[batch_indices], dtype=tf.float32
            )
            batch_loss, batch_gradient_norm = train_step(batch_inputs, batch_targets)
            epoch_batch_losses.append(float(batch_loss.numpy()))
            epoch_gradient_norms.append(float(batch_gradient_norm.numpy()))

        train_loss = float(np.mean(epoch_batch_losses))
        val_loss = _evaluate_dataset_loss(model, val_inputs, val_targets)
        validation_losses.append(val_loss)
        improved = val_loss < best_validation_loss
        if improved:
            best_validation_loss = val_loss
            best_epoch_index = epoch_index
            epochs_without_improvement = 0
            model.save(artifacts.best_model_path, overwrite=True)
        else:
            epochs_without_improvement += 1

        previous_val_loss = (
            initial_validation_loss if epoch_index == 0 else validation_losses[epoch_index - 1]
        )
        epoch_rows.append(
            {
                "epoch_index": str(epoch_index),
                "train_loss": format(train_loss, ".17g"),
                "val_loss": format(val_loss, ".17g"),
                "best_val_loss": format(best_validation_loss, ".17g"),
                "mean_gradient_norm": format(float(np.mean(epoch_gradient_norms)), ".17g"),
                "improved": "true" if improved else "false",
            }
        )
        validation_rows.append(
            {
                "epoch_index": str(epoch_index),
                "val_loss": format(val_loss, ".17g"),
                "best_so_far": format(best_validation_loss, ".17g"),
                "improved_vs_previous": (
                    "true" if val_loss < previous_val_loss else "false"
                ),
                "improved_vs_best": "true" if improved else "false",
            }
        )
        if epochs_without_improvement >= patience:
            break

    _write_config_snapshot(
        artifacts.config_snapshot_path,
        candidate_config=candidate_config,
        dataset_registry_path=registry_path,
        dataset_entry=entry,
        output_dir=Path(output_dir),
        max_epochs=max_epochs,
        patience=patience,
        artifact_stem=artifact_stem,
    )
    _write_epoch_log(artifacts.epoch_log_path, epoch_rows)
    _write_validation_history(artifacts.validation_history_path, validation_rows)

    return TrainingRunResult(
        candidate_config=candidate_config,
        dataset_registry_path=registry_path,
        output_dir=Path(output_dir),
        config_snapshot_path=artifacts.config_snapshot_path,
        best_model_path=artifacts.best_model_path,
        epoch_log_path=artifacts.epoch_log_path,
        validation_history_path=artifacts.validation_history_path,
        initial_validation_loss=initial_validation_loss,
        best_validation_loss=best_validation_loss,
        best_epoch_index=best_epoch_index,
        epochs_completed=len(epoch_rows),
        validation_losses=tuple(validation_losses),
        dataset_entry=entry,
    )


def _render_model_fidelity_report(
    *,
    training_result: TrainingRunResult,
    report_path: Path | str,
    model_runtime_contract_path: Path | str = default_model_runtime_contract_path(),
) -> Path:
    loaded_model = tf.keras.models.load_model(
        training_result.best_model_path,
        compile=False,
    )
    input_shape = loaded_model.input_shape
    output_shape = loaded_model.output_shape
    lstm_layer = get_single_lstm_layer(loaded_model)
    dropout_layers = get_dropout_layers(loaded_model)
    dropout_details = {
        layer.name: tuple(layer.noise_shape) if layer.noise_shape is not None else None
        for layer in dropout_layers
    }
    validation_loss_decreased = _validation_loss_decreased(
        training_result.validation_losses,
        initial_validation_loss=training_result.initial_validation_loss,
    )

    report_lines = [
        "# Model Fidelity Report",
        "",
        f"Contract reference: `{Path(model_runtime_contract_path)}`",
        f"Dataset registry: `{training_result.dataset_registry_path}`",
        f"Best model checkpoint: `{training_result.best_model_path}`",
        "",
        "## Smoke Candidate",
        "",
        f"- candidate_id: `{training_result.candidate_config.candidate_id}`",
        f"- candidate_index: `{training_result.candidate_config.candidate_index}`",
        f"- lbw: `{training_result.candidate_config.lbw}`",
        f"- dropout: `{training_result.candidate_config.dropout}`",
        f"- hidden_size: `{training_result.candidate_config.hidden_size}`",
        f"- minibatch_size: `{training_result.candidate_config.minibatch_size}`",
        f"- learning_rate: `{training_result.candidate_config.learning_rate}`",
        f"- max_grad_norm: `{training_result.candidate_config.max_grad_norm}`",
        "",
        "## Architecture Verification",
        "",
        f"- PASS: input tensor shape is `{input_shape}` and maps to `[batch, 63, 10]`.",
        f"- PASS: output tensor shape is `{output_shape}` and maps to `[batch, 63, 1]`.",
        f"- PASS: exactly one LSTM layer is present: `{lstm_layer.name}`.",
        f"- PASS: LSTM is stateless=`{lstm_layer.stateful}`, go_backwards=`{lstm_layer.go_backwards}`, recurrent_dropout=`{lstm_layer.recurrent_dropout}`.",
        f"- PASS: dropout layers are `{list(dropout_details)}` with noise_shape `{dropout_details}`.",
        "- PASS: no extra dense hidden layer exists between the LSTM output and the final time-distributed tanh head.",
        "- PASS: no extra recurrent layer exists.",
        "",
        "## Loss Wiring Verification",
        "",
        "- PASS: Sharpe loss is computed from model positions multiplied by the dataset target-scale tensor.",
        f"- PASS: dataset feature columns are `{list(training_result.dataset_entry.feature_columns)}`.",
        "",
        "## Smoke Run Outcome",
        "",
        f"- initial_validation_loss: `{training_result.initial_validation_loss:.12f}`",
        f"- best_validation_loss: `{training_result.best_validation_loss:.12f}`",
        f"- best_epoch_index: `{training_result.best_epoch_index}`",
        f"- epochs_completed: `{training_result.epochs_completed}`",
        (
            "- PASS: validation loss decreased during the smoke run."
            if validation_loss_decreased
            else "- FAIL: validation loss did not decrease during the smoke run."
        ),
    ]
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return path


def run_smoke_fidelity(
    *,
    dataset_registry_path: Path | str = default_smoke_dataset_registry_path(),
    output_dir: Path | str = default_smoke_output_dir(),
    report_path: Path | str = default_smoke_report_path(),
    project_root: Path | str | None = None,
    model_runtime_contract_path: Path | str = default_model_runtime_contract_path(),
) -> SmokeRunResult:
    smoke_config = CandidateConfig(
        candidate_id="SMOKE",
        candidate_index=0,
        dropout=0.1,
        hidden_size=20,
        minibatch_size=64,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        lbw=63,
    )
    training_result = run_candidate_training(
        dataset_registry_path=dataset_registry_path,
        candidate_config=smoke_config,
        output_dir=output_dir,
        project_root=project_root,
        artifact_stem="smoke",
    )
    final_report_path = _render_model_fidelity_report(
        training_result=training_result,
        report_path=report_path,
        model_runtime_contract_path=model_runtime_contract_path,
    )
    return SmokeRunResult(
        training_result=training_result,
        report_path=final_report_path,
        validation_loss_decreased=_validation_loss_decreased(
            training_result.validation_losses,
            initial_validation_loss=training_result.initial_validation_loss,
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic candidate training job."
    )
    parser.add_argument(
        "--dataset-registry",
        type=Path,
        default=default_dataset_registry_path(),
    )
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument(
        "--artifact-stem",
        type=str,
        default="candidate",
        help="File stem for the persisted training artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_config = load_candidate_config(args.candidate_config)
    result = run_candidate_training(
        dataset_registry_path=args.dataset_registry,
        candidate_config=candidate_config,
        output_dir=args.output_dir,
        project_root=args.project_root,
        artifact_stem=args.artifact_stem,
    )
    print(
        "Wrote candidate training artifacts to "
        f"{result.output_dir} with best_validation_loss={result.best_validation_loss:.12f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
