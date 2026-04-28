from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from lstm_cpd.canonical_daily_close_store import sha256_prefixed
from lstm_cpd.daily_close_contract import (
    default_asset_manifest_path,
    default_project_root,
    default_repo_root,
)
from lstm_cpd.datasets.join_and_split import project_relative_path
from lstm_cpd.evaluation.validation_evaluation import (
    default_evaluation_report_output,
    default_raw_validation_metrics_output,
    default_raw_validation_returns_output,
    default_rescaled_validation_metrics_output,
    default_rescaled_validation_returns_output,
)
from lstm_cpd.features.volatility import default_canonical_manifest_input
from lstm_cpd.inference.online_inference import (
    default_latest_positions_output,
    default_latest_sequence_manifest_output,
)
from lstm_cpd.model_source import (
    ResolvedModelSource,
    default_best_candidate_path,
    default_best_config_path,
    resolve_selected_model_source,
)
from lstm_cpd.training.search_schedule import default_search_schedule_json_path


DEFAULT_REPRODUCIBILITY_MANIFEST_OUTPUT = (
    "artifacts/reproducibility/reproducibility_manifest.json"
)
_VALID_RUN_TYPES = {"candidate", "smoke", "search_summary", "inference", "evaluation"}
_VALID_STATUSES = {"planned", "running", "completed", "failed", "aborted"}


@dataclass(frozen=True)
class ReproducibilityManifestArtifacts:
    manifest_path: Path
    model_source: ResolvedModelSource
    sampled_schedule_hash: str


def default_reproducibility_manifest_output() -> Path:
    return default_project_root() / DEFAULT_REPRODUCIBILITY_MANIFEST_OUTPUT


def _resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _write_json(path: Path | str, payload: object) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return json_path


def _project_root_text(project_root: Path) -> str:
    repo_root = default_repo_root().resolve()
    try:
        return project_root.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(project_root)


def _require_existing_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_reproducibility_manifest(
    *,
    best_candidate_path: Path | str = default_best_candidate_path(),
    best_config_path: Path | str = default_best_config_path(),
    model_path: Path | str | None = None,
    candidate_config_path: Path | str | None = None,
    dataset_registry_path: Path | str | None = None,
    search_schedule_json_path: Path | str = default_search_schedule_json_path(),
    ftmo_asset_universe_manifest_path: Path | str = default_asset_manifest_path(),
    canonical_daily_close_manifest_path: Path | str = default_canonical_manifest_input(),
    latest_positions_path: Path | str = default_latest_positions_output(),
    latest_sequence_manifest_path: Path | str = default_latest_sequence_manifest_output(),
    raw_validation_returns_path: Path | str = default_raw_validation_returns_output(),
    raw_validation_metrics_path: Path | str = default_raw_validation_metrics_output(),
    rescaled_validation_returns_path: Path | str = default_rescaled_validation_returns_output(),
    rescaled_validation_metrics_path: Path | str = default_rescaled_validation_metrics_output(),
    evaluation_report_path: Path | str = default_evaluation_report_output(),
    output_path: Path | str = default_reproducibility_manifest_output(),
    project_root: Path | str | None = None,
    run_id: str | None = None,
    created_at_utc: str | None = None,
    run_type: str = "evaluation",
    status: str = "completed",
) -> ReproducibilityManifestArtifacts:
    if run_type not in _VALID_RUN_TYPES:
        raise ValueError(f"Unsupported run_type: {run_type}")
    if status not in _VALID_STATUSES:
        raise ValueError(f"Unsupported status: {status}")

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
    resolved_search_schedule_path = _require_existing_file(
        _resolve_project_path(project_root_path, search_schedule_json_path),
        label="search_schedule.json",
    )
    resolved_ftmo_manifest_path = _require_existing_file(
        _resolve_project_path(project_root_path, ftmo_asset_universe_manifest_path),
        label="ftmo_asset_universe.json",
    )
    resolved_canonical_manifest_path = _require_existing_file(
        _resolve_project_path(project_root_path, canonical_daily_close_manifest_path),
        label="canonical_daily_close_manifest.json",
    )
    resolved_dataset_registry_path = _require_existing_file(
        model_source.dataset_registry_path,
        label="dataset_registry.json",
    )
    resolved_latest_positions_path = _require_existing_file(
        _resolve_project_path(project_root_path, latest_positions_path),
        label="latest_positions.csv",
    )
    resolved_latest_sequence_manifest_path = _require_existing_file(
        _resolve_project_path(project_root_path, latest_sequence_manifest_path),
        label="latest_sequence_manifest.csv",
    )
    resolved_raw_validation_returns_path = _require_existing_file(
        _resolve_project_path(project_root_path, raw_validation_returns_path),
        label="raw_validation_returns.csv",
    )
    resolved_raw_validation_metrics_path = _require_existing_file(
        _resolve_project_path(project_root_path, raw_validation_metrics_path),
        label="raw_validation_metrics.json",
    )
    resolved_rescaled_validation_returns_path = _require_existing_file(
        _resolve_project_path(project_root_path, rescaled_validation_returns_path),
        label="rescaled_validation_returns.csv",
    )
    resolved_rescaled_validation_metrics_path = _require_existing_file(
        _resolve_project_path(project_root_path, rescaled_validation_metrics_path),
        label="rescaled_validation_metrics.json",
    )
    resolved_evaluation_report_path = _require_existing_file(
        _resolve_project_path(project_root_path, evaluation_report_path),
        label="evaluation_report.md",
    )
    resolved_output_path = _resolve_project_path(project_root_path, output_path)

    sampled_schedule_hash = sha256_prefixed(resolved_search_schedule_path.read_bytes())
    payload = {
        "run_id": (
            run_id
            if run_id is not None
            else f"repro_{model_source.candidate_id.lower()}_lbw{model_source.lbw}"
        ),
        "run_type": run_type,
        "created_at_utc": created_at_utc if created_at_utc is not None else _now_utc_iso(),
        "project_root": _project_root_text(project_root_path),
        "authority_documents": {
            "spec": "spec_lstm_cpd_model_revised_sole_authority.md",
            "implementation_plan": "lstm_cpd_implementation_plan.md",
            "project_overview": "project_overview_slow_momentum_with_fast_reversion.md",
            "invariant_ledger": "docs/contracts/invariant_ledger.md",
            "exclusions_ledger": "docs/contracts/exclusions_ledger.md",
            "execution_policy_rules": "docs/contracts/execution_policy_rules.md",
        },
        "seed_policy": {
            "global_seed": 20260421,
            "epoch_seed_formula": "20260421 + epoch_index",
            "candidate_seed_formula": "20260421 + candidate_index",
            "candidate_index_basis": "zero_based_immutable_sample_schedule",
            "sampled_schedule_seed": 20260421,
            "sampled_schedule_hash": sampled_schedule_hash,
        },
        "selected_lbw": model_source.lbw,
        "candidate_ids": [model_source.candidate_id],
        "selected_candidate_id": model_source.candidate_id,
        "artifact_locations": {
            "dataset_registry": project_relative_path(
                resolved_dataset_registry_path,
                project_root_path,
            ),
            "checkpoint": project_relative_path(model_source.model_path, project_root_path),
            "latest_positions": project_relative_path(
                resolved_latest_positions_path,
                project_root_path,
            ),
            "latest_sequence_manifest": project_relative_path(
                resolved_latest_sequence_manifest_path,
                project_root_path,
            ),
            "raw_validation_returns": project_relative_path(
                resolved_raw_validation_returns_path,
                project_root_path,
            ),
            "raw_validation_metrics": project_relative_path(
                resolved_raw_validation_metrics_path,
                project_root_path,
            ),
            "rescaled_validation_returns": project_relative_path(
                resolved_rescaled_validation_returns_path,
                project_root_path,
            ),
            "rescaled_validation_metrics": project_relative_path(
                resolved_rescaled_validation_metrics_path,
                project_root_path,
            ),
            "report_paths": [
                project_relative_path(resolved_evaluation_report_path, project_root_path)
            ],
        },
        "source_artifacts": {
            "ftmo_asset_universe_manifest": project_relative_path(
                resolved_ftmo_manifest_path,
                project_root_path,
            ),
            "canonical_daily_close_manifest": project_relative_path(
                resolved_canonical_manifest_path,
                project_root_path,
            ),
            "dataset_registry": project_relative_path(
                resolved_dataset_registry_path,
                project_root_path,
            ),
            "search_schedule": project_relative_path(
                resolved_search_schedule_path,
                project_root_path,
            ),
            "best_candidate": (
                None
                if model_source.best_candidate_path is None
                else project_relative_path(model_source.best_candidate_path, project_root_path)
            ),
            "best_config": (
                None
                if model_source.candidate_config_path is None
                else project_relative_path(model_source.candidate_config_path, project_root_path)
            ),
        },
        "status": status,
        "hashes": {
            "sampled_schedule_hash": sampled_schedule_hash,
            "ftmo_asset_universe_hash": sha256_prefixed(
                resolved_ftmo_manifest_path.read_bytes()
            ),
            "canonical_daily_close_manifest_hash": sha256_prefixed(
                resolved_canonical_manifest_path.read_bytes()
            ),
            "dataset_registry_hash": sha256_prefixed(
                resolved_dataset_registry_path.read_bytes()
            ),
        },
        "entrypoints": {
            "inference": "src/lstm_cpd/inference/online_inference.py",
            "evaluation": "src/lstm_cpd/evaluation/validation_evaluation.py",
            "reproducibility": "src/lstm_cpd/reproducibility/manifest.py",
        },
    }
    manifest_path = _write_json(resolved_output_path, payload)
    return ReproducibilityManifestArtifacts(
        manifest_path=manifest_path,
        model_source=model_source,
        sampled_schedule_hash=sampled_schedule_hash,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final reproducibility manifest for inference/evaluation artifacts."
    )
    parser.add_argument("--best-candidate", type=Path, default=default_best_candidate_path())
    parser.add_argument("--best-config", type=Path, default=default_best_config_path())
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--candidate-config-path", type=Path)
    parser.add_argument("--dataset-registry-path", type=Path)
    parser.add_argument(
        "--search-schedule-json-path",
        type=Path,
        default=default_search_schedule_json_path(),
    )
    parser.add_argument(
        "--ftmo-asset-universe-manifest-path",
        type=Path,
        default=default_asset_manifest_path(),
    )
    parser.add_argument(
        "--canonical-daily-close-manifest-path",
        type=Path,
        default=default_canonical_manifest_input(),
    )
    parser.add_argument(
        "--latest-positions-path",
        type=Path,
        default=default_latest_positions_output(),
    )
    parser.add_argument(
        "--latest-sequence-manifest-path",
        type=Path,
        default=default_latest_sequence_manifest_output(),
    )
    parser.add_argument(
        "--raw-validation-returns-path",
        type=Path,
        default=default_raw_validation_returns_output(),
    )
    parser.add_argument(
        "--raw-validation-metrics-path",
        type=Path,
        default=default_raw_validation_metrics_output(),
    )
    parser.add_argument(
        "--rescaled-validation-returns-path",
        type=Path,
        default=default_rescaled_validation_returns_output(),
    )
    parser.add_argument(
        "--rescaled-validation-metrics-path",
        type=Path,
        default=default_rescaled_validation_metrics_output(),
    )
    parser.add_argument(
        "--evaluation-report-path",
        type=Path,
        default=default_evaluation_report_output(),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_reproducibility_manifest_output(),
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--created-at-utc", type=str)
    parser.add_argument("--run-type", type=str, default="evaluation")
    parser.add_argument("--status", type=str, default="completed")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = build_reproducibility_manifest(
        best_candidate_path=args.best_candidate,
        best_config_path=args.best_config,
        model_path=args.model_path,
        candidate_config_path=args.candidate_config_path,
        dataset_registry_path=args.dataset_registry_path,
        search_schedule_json_path=args.search_schedule_json_path,
        ftmo_asset_universe_manifest_path=args.ftmo_asset_universe_manifest_path,
        canonical_daily_close_manifest_path=args.canonical_daily_close_manifest_path,
        latest_positions_path=args.latest_positions_path,
        latest_sequence_manifest_path=args.latest_sequence_manifest_path,
        raw_validation_returns_path=args.raw_validation_returns_path,
        raw_validation_metrics_path=args.raw_validation_metrics_path,
        rescaled_validation_returns_path=args.rescaled_validation_returns_path,
        rescaled_validation_metrics_path=args.rescaled_validation_metrics_path,
        evaluation_report_path=args.evaluation_report_path,
        output_path=args.output_path,
        project_root=args.project_root,
        run_id=args.run_id,
        created_at_utc=args.created_at_utc,
        run_type=args.run_type,
        status=args.status,
    )
    print(f"Wrote reproducibility manifest to {artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
