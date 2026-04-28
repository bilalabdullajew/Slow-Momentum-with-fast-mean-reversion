from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.datasets.join_and_split import project_relative_path
from lstm_cpd.evaluation.validation_evaluation import (
    default_evaluation_report_output,
    default_raw_validation_metrics_output,
    default_raw_validation_returns_output,
    default_rescaled_validation_metrics_output,
    default_rescaled_validation_returns_output,
)
from lstm_cpd.inference.online_inference import (
    default_latest_positions_output,
    default_latest_sequence_manifest_output,
)
from lstm_cpd.model_source import (
    DEFAULT_BEST_CANDIDATE_PATH,
    DEFAULT_BEST_CONFIG_PATH,
    DEFAULT_DATASET_REGISTRY_PATH,
)
from lstm_cpd.reproducibility.manifest import (
    DEFAULT_REPRODUCIBILITY_MANIFEST_OUTPUT,
)
from lstm_cpd.training.selection import DEFAULT_SEARCH_SUMMARY_REPORT_PATH
from lstm_cpd.training.search_schedule import (
    DEFAULT_SEARCH_SCHEDULE_CSV_PATH,
    DEFAULT_SEARCH_SCHEDULE_JSON_PATH,
)


DEFAULT_INFERENCE_DIR = "artifacts/inference"
DEFAULT_EVALUATION_DIR = "artifacts/evaluation"


@dataclass(frozen=True)
class NotebookSectionSpec:
    section_id: str
    title: str
    narrative: str
    artifact_refs: tuple[str, ...]
    module_refs: tuple[str, ...]

    def metadata_payload(self) -> dict[str, object]:
        return {
            "section_id": self.section_id,
            "section_title": self.title,
            "artifact_refs": list(self.artifact_refs),
            "module_refs": list(self.module_refs),
        }


def _resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _artifact_ref_text(project_root: Path, path: Path | str) -> str:
    return project_relative_path(_resolve_project_path(project_root, path), project_root)


def build_replication_section_catalog(
    *,
    best_candidate_path: Path | str = DEFAULT_BEST_CANDIDATE_PATH,
    best_config_path: Path | str = DEFAULT_BEST_CONFIG_PATH,
    dataset_registry_path: Path | str = DEFAULT_DATASET_REGISTRY_PATH,
    search_summary_report_path: Path | str = DEFAULT_SEARCH_SUMMARY_REPORT_PATH,
    reproducibility_manifest_path: Path | str = DEFAULT_REPRODUCIBILITY_MANIFEST_OUTPUT,
    inference_dir: Path | str = DEFAULT_INFERENCE_DIR,
    evaluation_dir: Path | str = DEFAULT_EVALUATION_DIR,
    project_root: Path | str | None = None,
) -> tuple[NotebookSectionSpec, ...]:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    inference_dir_ref = _artifact_ref_text(project_root_path, inference_dir)
    evaluation_dir_ref = _artifact_ref_text(project_root_path, evaluation_dir)
    sections = (
        NotebookSectionSpec(
            section_id="implementation_contract",
            title="Implementation Contract",
            narrative=(
                "Freeze the authority documents that define the allowed methodology, "
                "execution-policy rules, and exclusions."
            ),
            artifact_refs=(
                "spec_lstm_cpd_model_revised_sole_authority.md",
                "lstm_cpd_implementation_plan.md",
                "docs/contracts/invariant_ledger.md",
                "docs/contracts/exclusions_ledger.md",
                "docs/contracts/execution_policy_rules.md",
            ),
            module_refs=(),
        ),
        NotebookSectionSpec(
            section_id="ftmo_data_contract",
            title="FTMO Data Contract",
            narrative=(
                "Show the admissible FTMO universe, D-timeframe path resolution, and the "
                "schema and screening reports that bound the raw input layer."
            ),
            artifact_refs=(
                "artifacts/manifests/ftmo_asset_universe.json",
                "artifacts/manifests/d_timeframe_path_manifest.json",
                "docs/contracts/daily_close_schema_contract.md",
                "artifacts/reports/schema_inspection_report.csv",
                "artifacts/reports/asset_eligibility_report.csv",
                "artifacts/reports/asset_exclusion_report.csv",
                "artifacts/reports/minimum_history_screening_report.csv",
            ),
            module_refs=(
                "lstm_cpd.ftmo_asset_universe",
                "lstm_cpd.daily_close_contract",
                "lstm_cpd.raw_history_screening",
            ),
        ),
        NotebookSectionSpec(
            section_id="canonical_daily_close_layer",
            title="Canonical Daily-Close Layer",
            narrative=(
                "Reference the frozen canonical close manifest that anchors all later "
                "feature and sequence work."
            ),
            artifact_refs=("artifacts/manifests/canonical_daily_close_manifest.json",),
            module_refs=("lstm_cpd.canonical_daily_close_store",),
        ),
        NotebookSectionSpec(
            section_id="base_features",
            title="Base Features",
            narrative=(
                "Summarize the deterministic return, volatility, normalized-return, MACD, "
                "and winsorization stack that produces the eight non-CPD features."
            ),
            artifact_refs=("artifacts/reports/feature_provenance_report.md",),
            module_refs=(
                "lstm_cpd.features.returns",
                "lstm_cpd.features.volatility",
                "lstm_cpd.features.normalized_returns",
                "lstm_cpd.features.macd",
                "lstm_cpd.features.winsorize",
            ),
        ),
        NotebookSectionSpec(
            section_id="cpd_outputs",
            title="CPD Outputs",
            narrative=(
                "Tie the GP-based CPD engine to its progress, telemetry, failure, fallback, "
                "and manifest artifacts."
            ),
            artifact_refs=(
                "docs/contracts/cpd_engine_contract.md",
                "artifacts/reports/cpd_precompute_progress.csv",
                "artifacts/reports/cpd_fit_telemetry.csv",
                "artifacts/reports/cpd_failure_ledger.csv",
                "artifacts/reports/cpd_fallback_ledger.csv",
                "artifacts/manifests/cpd_feature_store_manifest.json",
            ),
            module_refs=(
                "lstm_cpd.cpd.gp_kernels",
                "lstm_cpd.cpd.fit_window",
                "lstm_cpd.cpd.precompute",
                "lstm_cpd.cpd.telemetry",
            ),
        ),
        NotebookSectionSpec(
            section_id="dataset_assembly",
            title="Dataset Assembly",
            narrative=(
                "Reference the frozen dataset registry and the source modules that create "
                "joins, chronological splits, sequences, and array-ready registries."
            ),
            artifact_refs=(
                _artifact_ref_text(project_root_path, dataset_registry_path),
            ),
            module_refs=(
                "lstm_cpd.datasets.join_and_split",
                "lstm_cpd.datasets.sequences",
                "lstm_cpd.datasets.registry",
            ),
        ),
        NotebookSectionSpec(
            section_id="model_training_setup",
            title="Model and Training Setup",
            narrative=(
                "Expose the shared LSTM runtime, Sharpe-loss contract, single-candidate "
                "runner, and the smoke-train fidelity artifacts."
            ),
            artifact_refs=(
                "docs/contracts/model_runtime_contract.md",
                "artifacts/training/training_runner_contract.md",
                "artifacts/training/smoke_run/smoke_config.json",
                "artifacts/training/smoke_run/smoke_best_model.keras",
                "artifacts/training/smoke_run/smoke_epoch_log.csv",
                "artifacts/training/smoke_run/smoke_validation_history.csv",
                "artifacts/reports/model_fidelity_report.md",
            ),
            module_refs=(
                "lstm_cpd.model.network",
                "lstm_cpd.training.losses",
                "lstm_cpd.training.train_candidate",
            ),
        ),
        NotebookSectionSpec(
            section_id="search_results",
            title="Search Results",
            narrative=(
                "Display the immutable search schedule and the search summary report that "
                "collapses all candidate outcomes into a reviewable table."
            ),
            artifact_refs=(
                _artifact_ref_text(project_root_path, DEFAULT_SEARCH_SCHEDULE_JSON_PATH),
                _artifact_ref_text(project_root_path, DEFAULT_SEARCH_SCHEDULE_CSV_PATH),
                _artifact_ref_text(project_root_path, search_summary_report_path),
            ),
            module_refs=(
                "lstm_cpd.training.search_schedule",
                "lstm_cpd.training.search_runner",
            ),
        ),
        NotebookSectionSpec(
            section_id="selected_model",
            title="Selected Model",
            narrative=(
                "Show the winning-candidate metadata and the source helpers that resolve the "
                "selected checkpoint and candidate configuration."
            ),
            artifact_refs=(
                _artifact_ref_text(project_root_path, best_candidate_path),
                _artifact_ref_text(project_root_path, best_config_path),
            ),
            module_refs=(
                "lstm_cpd.model_source",
                "lstm_cpd.training.selection",
            ),
        ),
        NotebookSectionSpec(
            section_id="causal_inference",
            title="Causal Inference",
            narrative=(
                "Reference the latest causal sequence manifest and the next-day position "
                "outputs from the online inference path."
            ),
            artifact_refs=(
                _artifact_ref_text(project_root_path, Path(inference_dir_ref) / Path(default_latest_positions_output()).name),
                _artifact_ref_text(project_root_path, Path(inference_dir_ref) / Path(default_latest_sequence_manifest_output()).name),
            ),
            module_refs=("lstm_cpd.inference.online_inference",),
        ),
        NotebookSectionSpec(
            section_id="validation_evaluation",
            title="Validation Evaluation",
            narrative=(
                "Show the raw and rescaled validation outputs and the report that explains "
                "the model-faithful FTMO-vs.-paper evaluation boundary."
            ),
            artifact_refs=(
                _artifact_ref_text(project_root_path, Path(evaluation_dir_ref) / Path(default_raw_validation_returns_output()).name),
                _artifact_ref_text(project_root_path, Path(evaluation_dir_ref) / Path(default_raw_validation_metrics_output()).name),
                _artifact_ref_text(project_root_path, Path(evaluation_dir_ref) / Path(default_rescaled_validation_returns_output()).name),
                _artifact_ref_text(project_root_path, Path(evaluation_dir_ref) / Path(default_rescaled_validation_metrics_output()).name),
                _artifact_ref_text(project_root_path, Path(evaluation_dir_ref) / Path(default_evaluation_report_output()).name),
            ),
            module_refs=("lstm_cpd.evaluation.validation_evaluation",),
        ),
        NotebookSectionSpec(
            section_id="reproducibility_manifest",
            title="Reproducibility Manifest",
            narrative=(
                "Close the notebook with the manifest that hashes the selected model inputs "
                "and binds the frozen evaluation run back to the repo."
            ),
            artifact_refs=(
                _artifact_ref_text(project_root_path, reproducibility_manifest_path),
            ),
            module_refs=("lstm_cpd.reproducibility.manifest",),
        ),
    )
    validate_replication_section_catalog(sections)
    return sections


def validate_replication_section_catalog(
    sections: Sequence[NotebookSectionSpec],
) -> None:
    seen_ids: set[str] = set()
    for section in sections:
        if section.section_id in seen_ids:
            raise ValueError(f"Duplicate notebook section_id: {section.section_id}")
        seen_ids.add(section.section_id)
        if not section.title:
            raise ValueError(f"Notebook section has empty title: {section.section_id}")
        if not section.artifact_refs:
            raise ValueError(
                f"Notebook section must reference at least one artifact: {section.section_id}"
            )


def notebook_section_id_order(
    sections: Sequence[NotebookSectionSpec] | None = None,
) -> tuple[str, ...]:
    section_list = (
        tuple(sections)
        if sections is not None
        else build_replication_section_catalog()
    )
    return tuple(section.section_id for section in section_list)


def iter_artifact_refs(
    sections: Sequence[NotebookSectionSpec],
) -> Iterable[str]:
    for section in sections:
        yield from section.artifact_refs


def iter_module_refs(
    sections: Sequence[NotebookSectionSpec],
) -> Iterable[str]:
    for section in sections:
        yield from section.module_refs
