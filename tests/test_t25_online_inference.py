from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.cpd.precompute import (  # noqa: E402
    ReturnsVolatilityRecord as CPDReturnsVolatilityRecord,
    build_cpd_feature_rows,
)
from lstm_cpd.cpd.precompute_contract import (  # noqa: E402
    CPDWindowInput,
    CPDWindowResult,
    STATUS_INVALID_WINDOW,
    STATUS_SUCCESS,
)
from lstm_cpd.datasets.join_and_split import (  # noqa: E402
    build_joined_feature_rows,
)
from lstm_cpd.features.macd import build_macd_feature_rows  # noqa: E402
from lstm_cpd.features.normalized_returns import (  # noqa: E402
    ReturnsVolatilityRecord as NormalizedReturnsRecord,
    compute_normalized_return_features,
)
from lstm_cpd.features.returns import CANONICAL_DAILY_CLOSE_HEADER  # noqa: E402
from lstm_cpd.features.volatility import (  # noqa: E402
    build_returns_volatility_rows,
)
from lstm_cpd.features.winsorize import build_base_feature_rows, join_feature_rows  # noqa: E402
from lstm_cpd.inference.online_inference import run_online_inference  # noqa: E402
from lstm_cpd.model_source import resolve_selected_model_source  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def make_timestamp(day_offset: int) -> str:
    return (datetime(2024, 1, 1) + timedelta(days=day_offset)).isoformat()


def write_canonical_manifest(
    project_root: Path,
    *,
    asset_rows: list[tuple[str, int, float]],
) -> None:
    manifest = []
    for asset_id, row_count, base_price in asset_rows:
        rows = [
            [
                make_timestamp(day_offset),
                asset_id,
                f"{base_price + (0.15 * day_offset) + (((day_offset % 7) - 3) * 0.07):.6f}",
            ]
            for day_offset in range(row_count)
        ]
        canonical_path = project_root / f"artifacts/canonical_daily_close/{asset_id}.csv"
        write_csv(canonical_path, CANONICAL_DAILY_CLOSE_HEADER, rows)
        manifest.append(
            {
                "asset_id": asset_id,
                "symbol": asset_id,
                "category": "equity",
                "path_pattern": "unused",
                "source_d_file_path": "unused",
                "canonical_csv_path": f"artifacts/canonical_daily_close/{asset_id}.csv",
                "row_count": row_count,
                "first_timestamp": make_timestamp(0),
                "last_timestamp": make_timestamp(row_count - 1),
                "file_hash": f"sha256:{asset_id}",
            }
        )
    write_json(
        project_root / "artifacts/manifests/canonical_daily_close_manifest.json",
        manifest,
    )


def write_candidate_artifacts(
    project_root: Path,
    *,
    candidate_id: str,
    lbw: int,
    dataset_registry_path: str,
    config_path: Path,
    model_path: Path,
) -> None:
    write_json(
        config_path,
        {
            "candidate_id": candidate_id,
            "candidate_index": 7,
            "dropout": 0.1,
            "hidden_size": 20,
            "minibatch_size": 64,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "lbw": lbw,
            "dataset_registry_path": dataset_registry_path,
        },
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("stub checkpoint\n", encoding="utf-8")
    write_json(
        project_root / "artifacts/training/best_config.json",
        {
            "candidate_id": "BEST",
            "candidate_index": 0,
            "dropout": 0.5,
            "hidden_size": 80,
            "minibatch_size": 128,
            "learning_rate": 0.1,
            "max_grad_norm": 100.0,
            "lbw": 252,
        },
    )
    write_json(
        project_root / "artifacts/training/best_candidate.json",
        {
            "candidate_id": "BEST",
            "candidate_index": 0,
            "dropout": 0.5,
            "hidden_size": 80,
            "minibatch_size": 128,
            "learning_rate": 0.1,
            "max_grad_norm": 100.0,
            "lbw": 252,
            "best_model_path": "artifacts/training/best_model.keras",
            "dataset_registry_path": "artifacts/manifests/dataset_registry.json",
        },
    )
    write_json(project_root / dataset_registry_path, [{"lbw": lbw}])


def successful_stub_fit_window(window_input: CPDWindowInput) -> CPDWindowResult:
    finite_returns = [value for value in window_input.window_returns if np.isfinite(value)]
    if len(finite_returns) != window_input.lbw + 1:
        return CPDWindowResult(
            status=STATUS_INVALID_WINDOW,
            lbw=window_input.lbw,
            window_size=len(window_input.window_returns),
            nu=None,
            gamma=None,
            nlml_baseline=None,
            nlml_changepoint=None,
            retry_used=False,
            fallback_used=False,
            location_c=None,
            steepness_s=None,
            failure_stage="window_length",
            failure_message="incomplete_window",
        )
    return CPDWindowResult(
        status=STATUS_SUCCESS,
        lbw=window_input.lbw,
        window_size=len(window_input.window_returns),
        nu=0.25,
        gamma=0.75,
        nlml_baseline=1.0,
        nlml_changepoint=0.5,
        retry_used=False,
        fallback_used=False,
        location_c=float(window_input.lbw) / 2.0,
        steepness_s=2.0,
        failure_stage=None,
        failure_message=None,
    )


def build_expected_latest_sequence(project_root: Path, *, asset_id: str, lbw: int) -> np.ndarray:
    canonical_rows = []
    canonical_csv_path = project_root / f"artifacts/canonical_daily_close/{asset_id}.csv"
    with canonical_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        canonical_rows = [
            type(
                "CanonicalRow",
                (),
                {
                    "timestamp": row["timestamp"],
                    "asset_id": row["asset_id"],
                    "close_text": row["close"],
                    "close_value": float(row["close"]),
                },
            )()
            for row in reader
        ]
    returns_rows = build_returns_volatility_rows(canonical_rows)
    normalized_rows = compute_normalized_return_features(
        [
            NormalizedReturnsRecord(
                timestamp=row["timestamp"],
                asset_id=row["asset_id"],
                close_text=row["close"],
                close_value=float(row["close"]),
                sigma_t_text=row["sigma_t"],
                sigma_t_value=None if row["sigma_t"] == "" else float(row["sigma_t"]),
            )
            for row in returns_rows
        ]
    )
    macd_rows = build_macd_feature_rows(canonical_rows)
    base_rows = build_base_feature_rows(join_feature_rows(normalized_rows, macd_rows))
    cpd_rows = build_cpd_feature_rows(
        [
            CPDReturnsVolatilityRecord(
                timestamp=row["timestamp"],
                asset_id=row["asset_id"],
                arithmetic_return=(
                    None if row["arithmetic_return"] == "" else float(row["arithmetic_return"])
                ),
            )
            for row in returns_rows
        ],
        lbw=lbw,
        fit_window_fn=successful_stub_fit_window,
    )
    joined_rows = build_joined_feature_rows(
        base_rows=base_rows,
        returns_rows=[
            type(
                "ReturnsRow",
                (),
                {
                    "timestamp": row["timestamp"],
                    "asset_id": row["asset_id"],
                    "arithmetic_return": (
                        None if row["arithmetic_return"] == "" else float(row["arithmetic_return"])
                    ),
                    "sigma_t": None if row["sigma_t"] == "" else float(row["sigma_t"]),
                },
            )()
            for row in returns_rows
        ],
        cpd_rows=[
            type(
                "CPDRow",
                (),
                {
                    "timestamp": row["timestamp"],
                    "asset_id": row["asset_id"],
                    "lbw": int(row["lbw"]),
                    "nu": None if row["nu"] == "" else float(row["nu"]),
                    "gamma": None if row["gamma"] == "" else float(row["gamma"]),
                    "status": row["status"],
                    "has_outputs": row["nu"] != "" and row["gamma"] != "",
                },
            )()
            for row in cpd_rows
        ],
        lbw=lbw,
    )
    trailing_rows = [joined_rows[-1]]
    for row in reversed(joined_rows[:-1]):
        if row.timeline_index + 1 != trailing_rows[0].timeline_index:
            break
        trailing_rows.insert(0, row)
    latest_sequence = trailing_rows[-63:]
    return np.asarray([row.model_inputs for row in latest_sequence], dtype=np.float32)


class RecordingModel:
    def __init__(self) -> None:
        self.training_flags: list[bool] = []
        self.recorded_inputs: list[np.ndarray] = []

    def __call__(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        array = np.asarray(inputs, dtype=np.float32)
        self.training_flags.append(training)
        self.recorded_inputs.append(array)
        batch_size = array.shape[0]
        output = np.tile(np.arange(array.shape[1], dtype=np.float32), (batch_size, 1))
        return tf.convert_to_tensor(output[:, :, None], dtype=tf.float32)


class T25OnlineInferenceTests(unittest.TestCase):
    def test_model_source_resolver_prefers_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            override_config_path = (
                tmp_path / "artifacts/interim/training/smoke_run/smoke_config.json"
            )
            override_model_path = (
                tmp_path / "artifacts/interim/training/smoke_run/smoke_best_model.keras"
            )
            write_candidate_artifacts(
                tmp_path,
                candidate_id="SMOKE",
                lbw=21,
                dataset_registry_path="artifacts/interim/manifests/dataset_registry.json",
                config_path=override_config_path,
                model_path=override_model_path,
            )

            resolved = resolve_selected_model_source(
                best_candidate_path=tmp_path / "artifacts/training/best_candidate.json",
                best_config_path=tmp_path / "artifacts/training/best_config.json",
                model_path=override_model_path,
                candidate_config_path=override_config_path,
                dataset_registry_path=tmp_path / "artifacts/interim/manifests/dataset_registry.json",
                project_root=tmp_path,
            )

            self.assertEqual(resolved.candidate_id, "SMOKE")
            self.assertEqual(resolved.lbw, 21)
            self.assertEqual(resolved.model_path, override_model_path)
            self.assertEqual(
                resolved.dataset_registry_path,
                tmp_path / "artifacts/interim/manifests/dataset_registry.json",
            )

    def test_online_inference_uses_training_false_last_timestep_and_selected_lbw_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            write_canonical_manifest(
                tmp_path,
                asset_rows=[("TEST", 390, 100.0)],
            )
            config_path = tmp_path / "artifacts/interim/training/smoke_run/smoke_config.json"
            model_path = tmp_path / "artifacts/interim/training/smoke_run/smoke_best_model.keras"
            write_candidate_artifacts(
                tmp_path,
                candidate_id="SMOKE",
                lbw=21,
                dataset_registry_path="artifacts/interim/manifests/dataset_registry.json",
                config_path=config_path,
                model_path=model_path,
            )
            recording_model = RecordingModel()
            seen_lbws: list[int] = []

            def recording_fit_window(window_input: CPDWindowInput) -> CPDWindowResult:
                seen_lbws.append(window_input.lbw)
                return successful_stub_fit_window(window_input)

            with patch(
                "lstm_cpd.inference.online_inference.tf.keras.models.load_model",
                return_value=recording_model,
            ):
                artifacts = run_online_inference(
                    model_path=model_path,
                    candidate_config_path=config_path,
                    dataset_registry_path=tmp_path / "artifacts/interim/manifests/dataset_registry.json",
                    canonical_manifest_input=(
                        tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"
                    ),
                    latest_positions_output=tmp_path / "artifacts/inference/latest_positions.csv",
                    latest_sequence_manifest_output=(
                        tmp_path / "artifacts/inference/latest_sequence_manifest.csv"
                    ),
                    project_root=tmp_path,
                    fit_window_fn=recording_fit_window,
                )

            self.assertEqual(artifacts.asset_count, 1)
            self.assertEqual(recording_model.training_flags, [False])
            self.assertTrue(all(lbw == 21 for lbw in seen_lbws))
            recorded_inputs = recording_model.recorded_inputs[0]
            self.assertEqual(recorded_inputs.shape, (1, 63, 10))
            expected_inputs = build_expected_latest_sequence(tmp_path, asset_id="TEST", lbw=21)
            np.testing.assert_allclose(recorded_inputs[0], expected_inputs, atol=1e-6)

            with (tmp_path / "artifacts/inference/latest_positions.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                position_rows = list(csv.DictReader(handle))
            self.assertEqual(len(position_rows), 1)
            self.assertEqual(position_rows[0]["asset_id"], "TEST")
            self.assertEqual(float(position_rows[0]["next_day_position"]), 62.0)

            with (tmp_path / "artifacts/inference/latest_sequence_manifest.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                manifest_rows = list(csv.DictReader(handle))
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(manifest_rows[0]["row_count"], "63")
            self.assertEqual(manifest_rows[0]["candidate_id"], "SMOKE")

    def test_online_inference_hard_fails_if_any_admitted_asset_lacks_full_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            write_canonical_manifest(
                tmp_path,
                asset_rows=[("ENOUGH", 340, 100.0), ("SHORT", 300, 50.0)],
            )
            config_path = tmp_path / "artifacts/interim/training/smoke_run/smoke_config.json"
            model_path = tmp_path / "artifacts/interim/training/smoke_run/smoke_best_model.keras"
            write_candidate_artifacts(
                tmp_path,
                candidate_id="SMOKE",
                lbw=10,
                dataset_registry_path="artifacts/interim/manifests/dataset_registry.json",
                config_path=config_path,
                model_path=model_path,
            )

            with patch(
                "lstm_cpd.inference.online_inference.tf.keras.models.load_model",
                return_value=RecordingModel(),
            ):
                with self.assertRaisesRegex(ValueError, "SHORT"):
                    run_online_inference(
                        model_path=model_path,
                        candidate_config_path=config_path,
                        dataset_registry_path=(
                            tmp_path / "artifacts/interim/manifests/dataset_registry.json"
                        ),
                        canonical_manifest_input=(
                            tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"
                        ),
                        latest_positions_output=(
                            tmp_path / "artifacts/inference/latest_positions.csv"
                        ),
                        latest_sequence_manifest_output=(
                            tmp_path / "artifacts/inference/latest_sequence_manifest.csv"
                        ),
                        project_root=tmp_path,
                        fit_window_fn=successful_stub_fit_window,
                    )


if __name__ == "__main__":
    unittest.main()
