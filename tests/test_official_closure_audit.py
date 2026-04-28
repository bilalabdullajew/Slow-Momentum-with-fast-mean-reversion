from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.closure.audit import (  # noqa: E402
    NOTEBOOK_ARTIFACT_MAP_HEADER,
    SEARCH_COMPLETION_LOG_HEADER,
    SEARCH_SUMMARY_REPORT_HEADER,
    STAGE_G04,
    STAGE_G05,
    STAGE_G09,
    T17_GAP_EXCLUSION_HEADER,
    VALIDATION_RETURNS_HEADER,
    audit_official_closure,
)
from lstm_cpd.canonical_daily_close_store import sha256_prefixed  # noqa: E402
from lstm_cpd.cpd.precompute_contract import ALLOWED_CPD_LBWS  # noqa: E402
from lstm_cpd.cpd.telemetry import (  # noqa: E402
    FALLBACK_LEDGER_HEADER,
    FAILURE_LEDGER_HEADER,
    TELEMETRY_HEADER,
)
from lstm_cpd.datasets.join_and_split import (  # noqa: E402
    MODEL_INPUT_COLUMNS,
    T16_SPLIT_MANIFEST_HEADER,
)
from lstm_cpd.datasets.registry import T18_SEQUENCE_INDEX_HEADER  # noqa: E402
from lstm_cpd.datasets.sequences import (  # noqa: E402
    T17_SEQUENCE_MANIFEST_HEADER,
    T17_TARGET_ALIGNMENT_HEADER,
)
from lstm_cpd.notebook.assemble import build_replication_notebook  # noqa: E402
from lstm_cpd.notebook.catalog import (  # noqa: E402
    build_replication_section_catalog,
    notebook_section_id_order,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, header: tuple[str, ...] | list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def make_timestamp(day_offset: int) -> str:
    return (datetime(2024, 1, 1) + timedelta(days=day_offset)).isoformat()


def _write_fixture_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        write_json(path, {"path": str(path)})
        return
    if suffix == ".csv":
        path.write_text("column_a,column_b\nvalue_a,value_b\n", encoding="utf-8")
        return
    if suffix == ".md":
        path.write_text(f"# {path.name}\nfixture\n", encoding="utf-8")
        return
    if suffix == ".ipynb":
        path.write_text('{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}\n', encoding="utf-8")
        return
    path.write_text("fixture\n", encoding="utf-8")


def _materialize_section_fixture_refs(project_root: Path) -> None:
    sections = build_replication_section_catalog(project_root=project_root)
    for section in sections:
        for artifact_ref in section.artifact_refs:
            _write_fixture_file(project_root / artifact_ref)


def _write_canonical_layer(project_root: Path, *, asset_id: str = "TEST") -> None:
    canonical_csv_path = project_root / "artifacts/canonical_daily_close/TEST.csv"
    write_csv(
        canonical_csv_path,
        ("timestamp", "asset_id", "close"),
        [
            [make_timestamp(0), asset_id, "100.0"],
            [make_timestamp(1), asset_id, "101.0"],
            [make_timestamp(2), asset_id, "102.0"],
        ],
    )
    write_json(
        project_root / "artifacts/manifests/ftmo_asset_universe.json",
        [{"asset_id": asset_id, "symbol": asset_id, "category": "Equities"}],
    )
    write_json(
        project_root / "artifacts/manifests/canonical_daily_close_manifest.json",
        [
            {
                "asset_id": asset_id,
                "symbol": asset_id,
                "category": "Equities",
                "path_pattern": "category_symbol_d",
                "source_d_file_path": "data/FTMO Data/Equities/TEST/D/TEST_data.csv",
                "canonical_csv_path": "artifacts/canonical_daily_close/TEST.csv",
                "row_count": 3,
                "first_timestamp": make_timestamp(0),
                "last_timestamp": make_timestamp(2),
                "file_hash": sha256_prefixed(canonical_csv_path.read_bytes()),
            }
        ],
    )


def _write_g04_outputs(project_root: Path, *, asset_id: str = "TEST") -> None:
    progress_rows: list[list[str]] = []
    telemetry_rows: list[list[str]] = []
    manifest_rows: list[dict[str, object]] = []
    for lbw in ALLOWED_CPD_LBWS:
        cpd_csv_rel = f"artifacts/features/cpd/lbw_{lbw}/{asset_id}_cpd.csv"
        cpd_csv_path = project_root / cpd_csv_rel
        write_csv(
            cpd_csv_path,
            ("timestamp", "asset_id", "lbw", "nu", "gamma"),
            [
                [make_timestamp(0), asset_id, str(lbw), "0.1", "0.2"],
                [make_timestamp(1), asset_id, str(lbw), "0.1", "0.2"],
                [make_timestamp(2), asset_id, str(lbw), "0.1", "0.2"],
            ],
        )
        progress_rows.append(
            [
                str(lbw),
                asset_id,
                "completed",
                "3",
                make_timestamp(2),
                "0",
                "0",
                "2026-04-23T00:00:00Z",
                "2026-04-23T00:00:01Z",
                cpd_csv_rel,
                "",
            ]
        )
        telemetry_rows.append(
            [
                asset_id,
                str(lbw),
                "present",
                "",
                cpd_csv_rel.replace("artifacts/features/cpd/", ""),
                "3",
                "3",
                make_timestamp(0),
                make_timestamp(2),
                "3",
                "0",
                "0",
                "0",
                "0",
                "0",
                "3",
                "0",
                "0",
            ]
        )
        manifest_rows.append(
            {
                "asset_id": asset_id,
                "lbw": lbw,
                "state": "present",
                "missing_reason": None,
                "cpd_csv_path": cpd_csv_rel,
                "row_count": 3,
                "canonical_row_count": 3,
                "first_timestamp": make_timestamp(0),
                "last_timestamp": make_timestamp(2),
                "output_row_count": 3,
                "retry_used_count": 0,
                "fallback_used_count": 0,
                "status_counts": {
                    "success": 3,
                    "retry_success": 0,
                    "fallback_previous": 0,
                    "baseline_failure": 0,
                    "changepoint_failure": 0,
                },
                "matches_canonical_timeline": True,
                "file_hash": sha256_prefixed(cpd_csv_path.read_bytes()),
            }
        )
    write_csv(
        project_root / "artifacts/reports/cpd_precompute_progress.csv",
        (
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
        ),
        progress_rows,
    )
    write_csv(
        project_root / "artifacts/reports/cpd_fit_telemetry.csv",
        TELEMETRY_HEADER,
        telemetry_rows,
    )
    write_csv(
        project_root / "artifacts/reports/cpd_failure_ledger.csv",
        FAILURE_LEDGER_HEADER,
        [],
    )
    write_csv(
        project_root / "artifacts/reports/cpd_fallback_ledger.csv",
        FALLBACK_LEDGER_HEADER,
        [],
    )
    write_json(
        project_root / "artifacts/manifests/cpd_feature_store_manifest.json",
        manifest_rows,
    )


def _write_dataset_outputs(project_root: Path, *, asset_id: str = "TEST") -> None:
    dataset_registry_rows: list[dict[str, object]] = []
    for lbw in ALLOWED_CPD_LBWS:
        lbw_dir = project_root / f"artifacts/datasets/lbw_{lbw}"
        train_sequence_id = f"{asset_id}__lbw_{lbw}__train__00000000"
        val_sequence_id = f"{asset_id}__lbw_{lbw}__validation__00000100"
        write_csv(
            lbw_dir / "split_manifest.csv",
            T16_SPLIT_MANIFEST_HEADER,
            [
                [
                    asset_id,
                    str(lbw),
                    "126",
                    "113",
                    "13",
                    make_timestamp(0),
                    make_timestamp(112),
                    make_timestamp(113),
                    make_timestamp(125),
                    "0",
                    "112",
                    "113",
                    "125",
                    "floor_90_rest_10",
                ]
            ],
        )
        write_csv(
            lbw_dir / "sequence_manifest.csv",
            T17_SEQUENCE_MANIFEST_HEADER,
            [
                [
                    train_sequence_id,
                    asset_id,
                    str(lbw),
                    "train",
                    "0",
                    make_timestamp(0),
                    make_timestamp(62),
                    "0",
                    "62",
                    "63",
                ],
                [
                    val_sequence_id,
                    asset_id,
                    str(lbw),
                    "validation",
                    "0",
                    make_timestamp(100),
                    make_timestamp(162),
                    "100",
                    "162",
                    "63",
                ],
            ],
        )
        target_rows: list[list[str]] = []
        for step_index in range(63):
            feature_values = [
                str(step_index + feature_index + 1)
                for feature_index in range(len(MODEL_INPUT_COLUMNS))
            ]
            target_scale = 0.15 / 0.5 * 0.01
            target_rows.append(
                [
                    train_sequence_id,
                    asset_id,
                    str(lbw),
                    "train",
                    str(step_index),
                    make_timestamp(step_index),
                    str(step_index),
                    "0.5",
                    "0.01",
                    str(target_scale),
                    *feature_values,
                ]
            )
            target_rows.append(
                [
                    val_sequence_id,
                    asset_id,
                    str(lbw),
                    "validation",
                    str(step_index),
                    make_timestamp(100 + step_index),
                    str(100 + step_index),
                    "0.5",
                    "0.01",
                    str(target_scale),
                    *feature_values,
                ]
            )
        write_csv(
            lbw_dir / "target_alignment_registry.csv",
            T17_TARGET_ALIGNMENT_HEADER,
            target_rows,
        )
        write_csv(
            lbw_dir / "discarded_fragments_report.csv",
            (
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
            ),
            [],
        )
        write_csv(
            lbw_dir / "gap_exclusion_report.csv",
            T17_GAP_EXCLUSION_HEADER,
            [],
        )
        train_inputs = np.ones((1, 63, len(MODEL_INPUT_COLUMNS)), dtype=np.float32)
        val_inputs = np.full((1, 63, len(MODEL_INPUT_COLUMNS)), 2.0, dtype=np.float32)
        train_targets = np.full((1, 63), 0.003, dtype=np.float32)
        val_targets = np.full((1, 63), 0.003, dtype=np.float32)
        np.save(lbw_dir / "train_inputs.npy", train_inputs)
        np.save(lbw_dir / "train_target_scale.npy", train_targets)
        np.save(lbw_dir / "val_inputs.npy", val_inputs)
        np.save(lbw_dir / "val_target_scale.npy", val_targets)
        write_csv(
            lbw_dir / "train_sequence_index.csv",
            T18_SEQUENCE_INDEX_HEADER,
            [[
                "0",
                train_sequence_id,
                asset_id,
                str(lbw),
                "train",
                make_timestamp(0),
                make_timestamp(62),
                "0",
                "62",
            ]],
        )
        write_csv(
            lbw_dir / "val_sequence_index.csv",
            T18_SEQUENCE_INDEX_HEADER,
            [[
                "0",
                val_sequence_id,
                asset_id,
                str(lbw),
                "validation",
                make_timestamp(100),
                make_timestamp(162),
                "100",
                "162",
            ]],
        )
        dataset_registry_rows.append(
            {
                "lbw": lbw,
                "feature_columns": list(MODEL_INPUT_COLUMNS),
                "sequence_length": 63,
                "train_sequence_count": 1,
                "val_sequence_count": 1,
                "train_input_shape": [1, 63, len(MODEL_INPUT_COLUMNS)],
                "train_target_shape": [1, 63],
                "val_input_shape": [1, 63, len(MODEL_INPUT_COLUMNS)],
                "val_target_shape": [1, 63],
                "artifacts": {
                    "train_inputs_path": f"artifacts/datasets/lbw_{lbw}/train_inputs.npy",
                    "train_target_scale_path": (
                        f"artifacts/datasets/lbw_{lbw}/train_target_scale.npy"
                    ),
                    "val_inputs_path": f"artifacts/datasets/lbw_{lbw}/val_inputs.npy",
                    "val_target_scale_path": (
                        f"artifacts/datasets/lbw_{lbw}/val_target_scale.npy"
                    ),
                    "train_sequence_index_path": (
                        f"artifacts/datasets/lbw_{lbw}/train_sequence_index.csv"
                    ),
                    "val_sequence_index_path": (
                        f"artifacts/datasets/lbw_{lbw}/val_sequence_index.csv"
                    ),
                },
                "source_artifacts": {
                    "split_manifest_path": f"artifacts/datasets/lbw_{lbw}/split_manifest.csv",
                    "sequence_manifest_path": (
                        f"artifacts/datasets/lbw_{lbw}/sequence_manifest.csv"
                    ),
                    "target_alignment_registry_path": (
                        f"artifacts/datasets/lbw_{lbw}/target_alignment_registry.csv"
                    ),
                },
            }
        )
    write_json(
        project_root / "artifacts/manifests/dataset_registry.json",
        dataset_registry_rows,
    )


def _write_smoke_outputs(project_root: Path) -> None:
    smoke_dir = project_root / "artifacts/training/smoke_run"
    write_json(
        smoke_dir / "smoke_config.json",
        {
            "candidate_id": "SMOKE",
            "candidate_index": 0,
            "dropout": 0.1,
            "hidden_size": 20,
            "minibatch_size": 64,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "lbw": 63,
            "dataset_registry_path": "artifacts/manifests/dataset_registry.json",
        },
    )
    (smoke_dir / "smoke_best_model.keras").write_text("checkpoint\n", encoding="utf-8")
    write_csv(
        smoke_dir / "smoke_epoch_log.csv",
        (
            "epoch_index",
            "train_loss",
            "val_loss",
            "best_val_loss",
            "mean_gradient_norm",
            "improved",
        ),
        [["0", "-0.10", "-0.20", "-0.20", "0.40", "true"]],
    )
    write_csv(
        smoke_dir / "smoke_validation_history.csv",
        (
            "epoch_index",
            "val_loss",
            "best_so_far",
            "improved_vs_previous",
            "improved_vs_best",
        ),
        [["0", "-0.20", "-0.20", "true", "true"]],
    )
    (project_root / "artifacts/reports/model_fidelity_report.md").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        project_root / "artifacts/reports/model_fidelity_report.md"
    ).write_text(
        "# Model Fidelity Report\n\n- PASS: validation loss decreased during the smoke run.\n",
        encoding="utf-8",
    )


def _candidate_payload(candidate_index: int) -> dict[str, object]:
    lbws = list(ALLOWED_CPD_LBWS)
    return {
        "candidate_index": candidate_index,
        "candidate_id": f"C-{candidate_index + 1:03d}",
        "dropout": round(0.1 + 0.1 * (candidate_index % 5), 1),
        "hidden_size": [5, 10, 20, 40, 80, 160][candidate_index % 6],
        "minibatch_size": [64, 128, 256][candidate_index % 3],
        "learning_rate": [0.0001, 0.001, 0.01, 0.1][candidate_index % 4],
        "max_grad_norm": [0.01, 1.0, 100.0][candidate_index % 3],
        "lbw": lbws[candidate_index % len(lbws)],
    }


def _write_search_outputs(project_root: Path) -> None:
    schedule_rows = [_candidate_payload(candidate_index) for candidate_index in range(50)]
    write_json(project_root / "artifacts/training/search_schedule.json", schedule_rows)
    write_csv(
        project_root / "artifacts/training/search_schedule.csv",
        (
            "candidate_index",
            "candidate_id",
            "dropout",
            "hidden_size",
            "minibatch_size",
            "learning_rate",
            "max_grad_norm",
            "lbw",
        ),
        [
            [
                str(row["candidate_index"]),
                str(row["candidate_id"]),
                str(row["dropout"]),
                str(row["hidden_size"]),
                str(row["minibatch_size"]),
                str(row["learning_rate"]),
                str(row["max_grad_norm"]),
                str(row["lbw"]),
            ]
            for row in schedule_rows
        ],
    )
    completion_rows: list[list[str]] = []
    summary_rows: list[list[str]] = []
    for row in schedule_rows:
        candidate_index = int(row["candidate_index"])
        candidate_dir = project_root / "artifacts/training/candidates" / f"candidate_{candidate_index:03d}"
        config_path = candidate_dir / "config.json"
        best_model_path = candidate_dir / "best_model.keras"
        epoch_log_path = candidate_dir / "epoch_log.csv"
        validation_history_path = candidate_dir / "validation_history.csv"
        final_metadata_path = candidate_dir / "final_metadata.json"
        write_json(
            config_path,
            {
                **row,
                "dataset_registry_path": "artifacts/manifests/dataset_registry.json",
            },
        )
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        best_model_path.write_text("checkpoint\n", encoding="utf-8")
        epoch_log_path.write_text("epoch_index,val_loss\n0,-0.1\n", encoding="utf-8")
        validation_history_path.write_text("epoch_index,val_loss\n0,-0.1\n", encoding="utf-8")
        write_json(
            final_metadata_path,
            {
                **row,
                "status": "completed",
                "dataset_registry_path": "artifacts/manifests/dataset_registry.json",
                "best_model_path": str(best_model_path),
            },
        )
        best_validation_loss = f"{-0.5000 + candidate_index * 0.001:.16f}"
        completion_rows.append(
            [
                str(candidate_index),
                str(row["candidate_id"]),
                "completed",
                "",
                str(row["dropout"]),
                str(row["hidden_size"]),
                str(row["minibatch_size"]),
                str(row["learning_rate"]),
                str(row["max_grad_norm"]),
                str(row["lbw"]),
                best_validation_loss,
                "0",
                "1",
                "artifacts/manifests/dataset_registry.json",
                str(candidate_dir),
                str(config_path),
                str(best_model_path),
                str(epoch_log_path),
                str(validation_history_path),
                str(final_metadata_path),
            ]
        )
        summary_rows.append(
            [
                str(candidate_index),
                str(row["candidate_id"]),
                "completed",
                "true" if candidate_index == 0 else "false",
                best_validation_loss,
                "",
                str(row["dropout"]),
                str(row["hidden_size"]),
                str(row["minibatch_size"]),
                str(row["learning_rate"]),
                str(row["max_grad_norm"]),
                str(row["lbw"]),
                str(best_model_path),
                str(final_metadata_path),
            ]
        )
    write_csv(
        project_root / "artifacts/training/search_completion_log.csv",
        SEARCH_COMPLETION_LOG_HEADER,
        completion_rows,
    )
    write_json(
        project_root / "artifacts/training/best_candidate.json",
        {
            **schedule_rows[0],
            "dataset_registry_path": "artifacts/manifests/dataset_registry.json",
            "best_model_path": str(
                project_root
                / "artifacts/training/candidates/candidate_000/best_model.keras"
            ),
        },
    )
    write_json(
        project_root / "artifacts/training/best_config.json",
        schedule_rows[0],
    )
    write_csv(
        project_root / "artifacts/reports/search_summary_report.csv",
        SEARCH_SUMMARY_REPORT_HEADER,
        summary_rows,
    )


def _write_g08_outputs(project_root: Path) -> None:
    write_csv(
        project_root / "artifacts/inference/latest_positions.csv",
        (
            "asset_id",
            "lbw",
            "signal_timestamp",
            "next_day_position",
            "candidate_id",
            "model_path",
        ),
        [[
            "TEST",
            "63",
            make_timestamp(162),
            "0.25",
            "C-001",
            str(project_root / "artifacts/training/candidates/candidate_000/best_model.keras"),
        ]],
    )
    write_csv(
        project_root / "artifacts/inference/latest_sequence_manifest.csv",
        (
            "asset_id",
            "lbw",
            "sequence_start_timestamp",
            "sequence_end_timestamp",
            "row_count",
            "start_timeline_index",
            "end_timeline_index",
            "candidate_id",
            "model_path",
        ),
        [[
            "TEST",
            "63",
            make_timestamp(100),
            make_timestamp(162),
            "63",
            "100",
            "162",
            "C-001",
            str(project_root / "artifacts/training/candidates/candidate_000/best_model.keras"),
        ]],
    )
    write_csv(
        project_root / "artifacts/evaluation/raw_validation_returns.csv",
        VALIDATION_RETURNS_HEADER,
        [[make_timestamp(162), "1", "0.001"]],
    )
    write_json(
        project_root / "artifacts/evaluation/raw_validation_metrics.json",
        {
            "annualized_return": 0.10,
            "annualized_volatility": 0.12,
            "annualized_downside_deviation": 0.08,
            "sharpe_ratio": 0.83,
            "sortino_ratio": 1.25,
            "maximum_drawdown": -0.05,
            "calmar_ratio": 2.0,
            "percentage_positive_daily_returns": 55.0,
        },
    )
    write_csv(
        project_root / "artifacts/evaluation/rescaled_validation_returns.csv",
        VALIDATION_RETURNS_HEADER,
        [[make_timestamp(162), "1", "0.00125"]],
    )
    write_json(
        project_root / "artifacts/evaluation/rescaled_validation_metrics.json",
        {
            "annualized_return": 0.12,
            "annualized_volatility": 0.15,
            "annualized_downside_deviation": 0.10,
            "sharpe_ratio": 0.80,
            "sortino_ratio": 1.20,
            "maximum_drawdown": -0.05,
            "calmar_ratio": 2.4,
            "percentage_positive_daily_returns": 55.0,
        },
    )
    (
        project_root / "artifacts/evaluation/evaluation_report.md"
    ).parent.mkdir(parents=True, exist_ok=True)
    (
        project_root / "artifacts/evaluation/evaluation_report.md"
    ).write_text(
        "# Evaluation Report\n\nFTMO universe differs from the 50-futures paper universe.\n",
        encoding="utf-8",
    )
    write_json(
        project_root / "artifacts/reproducibility/reproducibility_manifest.json",
        {
            "run_id": "repro_c001_lbw63",
            "run_type": "evaluation",
            "created_at_utc": "2026-04-23T12:00:00Z",
            "project_root": ".",
            "seed_policy": {
                "global_seed": 20260421,
                "sampled_schedule_hash": "sha256:abc",
            },
            "selected_lbw": 63,
            "selected_candidate_id": "C-001",
            "artifact_locations": {
                "dataset_registry": "artifacts/manifests/dataset_registry.json",
                "checkpoint": "artifacts/training/candidates/candidate_000/best_model.keras",
                "latest_positions": "artifacts/inference/latest_positions.csv",
                "latest_sequence_manifest": "artifacts/inference/latest_sequence_manifest.csv",
                "raw_validation_returns": "artifacts/evaluation/raw_validation_returns.csv",
                "raw_validation_metrics": "artifacts/evaluation/raw_validation_metrics.json",
                "rescaled_validation_returns": (
                    "artifacts/evaluation/rescaled_validation_returns.csv"
                ),
                "rescaled_validation_metrics": (
                    "artifacts/evaluation/rescaled_validation_metrics.json"
                ),
                "report_paths": ["artifacts/evaluation/evaluation_report.md"],
            },
            "source_artifacts": {
                "ftmo_asset_universe_manifest": "artifacts/manifests/ftmo_asset_universe.json",
                "canonical_daily_close_manifest": (
                    "artifacts/manifests/canonical_daily_close_manifest.json"
                ),
                "dataset_registry": "artifacts/manifests/dataset_registry.json",
                "search_schedule": "artifacts/training/search_schedule.json",
                "best_candidate": "artifacts/training/best_candidate.json",
                "best_config": "artifacts/training/best_config.json",
            },
            "status": "completed",
            "hashes": {
                "sampled_schedule_hash": "sha256:abc",
                "ftmo_asset_universe_hash": "sha256:def",
                "canonical_daily_close_manifest_hash": "sha256:ghi",
                "dataset_registry_hash": "sha256:jkl",
            },
            "entrypoints": {
                "inference": "src/lstm_cpd/inference/online_inference.py",
                "evaluation": "src/lstm_cpd/evaluation/validation_evaluation.py",
                "reproducibility": "src/lstm_cpd/reproducibility/manifest.py",
            },
        },
    )


def _write_g09_outputs(project_root: Path) -> None:
    build_replication_notebook(
        project_root=project_root,
        output_path=project_root / "notebooks/lstm_cpd_replication.ipynb",
    )
    notebook_path = project_root / "notebooks/lstm_cpd_replication.ipynb"
    executed_notebook_path = project_root / "notebooks/lstm_cpd_replication.executed.ipynb"
    executed_notebook_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(notebook_path, executed_notebook_path)
    section_catalog = build_replication_section_catalog(project_root=project_root)
    rows: list[list[str]] = []
    for cell_index, section in enumerate(section_catalog, start=1):
        for artifact_ref in section.artifact_refs:
            rows.append(
                [
                    section.section_id,
                    section.title,
                    str(cell_index),
                    artifact_ref,
                    ";".join(section.module_refs),
                ]
            )
    write_csv(
        project_root / "artifacts/notebook/notebook_artifact_map.csv",
        NOTEBOOK_ARTIFACT_MAP_HEADER,
        rows,
    )
    (
        project_root / "artifacts/notebook/notebook_execution_report.md"
    ).parent.mkdir(parents=True, exist_ok=True)
    (
        project_root / "artifacts/notebook/notebook_execution_report.md"
    ).write_text(
        "# Notebook Execution Report\n\n- Status: success\n- Manual intervention required: no\n",
        encoding="utf-8",
    )


def build_complete_project_root(project_root: Path) -> None:
    _materialize_section_fixture_refs(project_root)
    _write_canonical_layer(project_root)
    _write_g04_outputs(project_root)
    _write_dataset_outputs(project_root)
    _write_smoke_outputs(project_root)
    _write_search_outputs(project_root)
    _write_g08_outputs(project_root)
    _write_g09_outputs(project_root)


class OfficialClosureAuditTests(unittest.TestCase):
    def test_audit_default_report_paths_follow_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            with patch("lstm_cpd.closure.audit._write_json"), patch(
                "lstm_cpd.closure.audit._write_text"
            ):
                audit = audit_official_closure(project_root=tmp_path)

            self.assertEqual(
                audit.json_report_path,
                tmp_path / "artifacts/reports/official_closure_audit.json",
            )
            self.assertEqual(
                audit.markdown_report_path,
                tmp_path / "artifacts/reports/official_closure_audit.md",
            )

    def test_audit_blocks_partial_state_at_g04(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            write_json(
                tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json",
                [
                    {
                        "asset_id": "TEST",
                        "symbol": "TEST",
                        "category": "Equities",
                        "path_pattern": "category_symbol_d",
                        "source_d_file_path": "data/source.csv",
                        "canonical_csv_path": "artifacts/canonical_daily_close/TEST.csv",
                        "row_count": 3,
                        "first_timestamp": make_timestamp(0),
                        "last_timestamp": make_timestamp(2),
                        "file_hash": "sha256:test",
                    }
                ],
            )
            write_csv(
                tmp_path / "artifacts/reports/cpd_precompute_progress.csv",
                (
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
                ),
                [
                    ["10", "TEST", "completed", "3", make_timestamp(2), "0", "0", "", "", "artifacts/features/cpd/lbw_10/TEST_cpd.csv", ""],
                    ["21", "TEST", "running", "0", "", "0", "0", "", "", "artifacts/features/cpd/lbw_21/TEST_cpd.csv", ""],
                ],
            )
            write_json(
                tmp_path / "artifacts/manifests/dataset_registry.json",
                {"path": str(tmp_path / "artifacts/manifests/dataset_registry.json")},
            )
            (tmp_path / "artifacts/features/cpd/lbw_10/TEST_cpd.partial.csv").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (
                tmp_path / "artifacts/features/cpd/lbw_10/TEST_cpd.partial.csv"
            ).write_text("partial\n", encoding="utf-8")

            audit = audit_official_closure(project_root=tmp_path)

            self.assertFalse(audit.ready_for_project_closure)
            self.assertEqual(audit.first_blocking_stage, STAGE_G04)
            finding_codes = {finding.code for finding in audit.findings}
            self.assertIn("G04_T14_INCOMPLETE", finding_codes)
            self.assertIn("G04_PARTIAL_OUTPUTS", finding_codes)
            self.assertTrue(audit.json_report_path.exists())
            self.assertTrue(audit.markdown_report_path.exists())

    def test_audit_blocks_g05_when_official_registry_still_points_to_interim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            build_complete_project_root(tmp_path)
            registry_path = tmp_path / "artifacts/manifests/dataset_registry.json"
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            payload[0]["artifacts"]["train_inputs_path"] = (
                "artifacts/interim/datasets/lbw_10/train_inputs.npy"
            )
            write_json(registry_path, payload)

            audit = audit_official_closure(project_root=tmp_path)

            self.assertFalse(audit.ready_for_project_closure)
            self.assertEqual(audit.first_blocking_stage, STAGE_G05)
            finding_codes = {finding.code for finding in audit.findings}
            self.assertIn("G05_DATASET_REGISTRY_INTERIM", finding_codes)

    def test_audit_marks_complete_official_fixture_as_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            build_complete_project_root(tmp_path)

            audit = audit_official_closure(project_root=tmp_path)

            self.assertTrue(audit.ready_for_project_closure)
            self.assertIsNone(audit.first_blocking_stage)
            self.assertEqual(
                tuple(stage.stage_id for stage in audit.stages if stage.status == "ready"),
                (STAGE_G04, STAGE_G05, "G-06", "G-07", "G-08", STAGE_G09),
            )
            report_text = audit.markdown_report_path.read_text(encoding="utf-8")
            self.assertIn("Official artifact chain is complete", report_text)
            notebook_rows = list(
                csv.DictReader(
                    (tmp_path / "artifacts/notebook/notebook_artifact_map.csv").open(
                        "r",
                        encoding="utf-8",
                        newline="",
                    )
                )
            )
            self.assertEqual(
                tuple(dict.fromkeys(row["section_id"] for row in notebook_rows)),
                notebook_section_id_order(),
            )


if __name__ == "__main__":
    unittest.main()
