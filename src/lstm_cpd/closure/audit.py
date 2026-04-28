from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import nbformat
import numpy as np

from lstm_cpd.cpd.precompute_contract import ALLOWED_CPD_LBWS
from lstm_cpd.cpd.telemetry import (
    FALLBACK_LEDGER_HEADER,
    FAILURE_LEDGER_HEADER,
    TELEMETRY_HEADER,
)
from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.datasets.join_and_split import (
    MODEL_INPUT_COLUMNS,
    T16_SPLIT_MANIFEST_HEADER,
    load_cpd_feature_store_manifest,
    load_split_manifest_csv,
    project_relative_path,
)
from lstm_cpd.datasets.registry import T18_SEQUENCE_INDEX_HEADER
from lstm_cpd.datasets.sequences import (
    SEQUENCE_LENGTH,
    T17_GAP_EXCLUSION_HEADER,
    T17_SEQUENCE_MANIFEST_HEADER,
    T17_TARGET_ALIGNMENT_HEADER,
    load_sequence_manifest_csv,
    load_target_alignment_registry_csv,
)
from lstm_cpd.features.volatility import load_canonical_daily_close_manifest
from lstm_cpd.notebook.catalog import notebook_section_id_order
from lstm_cpd.notebook.execute import NOTEBOOK_ARTIFACT_MAP_HEADER


STAGE_G04 = "G-04"
STAGE_G05 = "G-05"
STAGE_G06 = "G-06"
STAGE_G07 = "G-07"
STAGE_G08 = "G-08"
STAGE_G09 = "G-09"

_STAGE_NAMES = {
    STAGE_G04: "CPD Fidelity",
    STAGE_G05: "Split, Sequence, and Dataset Assembly",
    STAGE_G06: "Model and Training Setup",
    STAGE_G07: "Search and Selection",
    STAGE_G08: "Inference, Evaluation, and Reproducibility",
    STAGE_G09: "Notebook",
}
_STAGE_ORDER = (
    STAGE_G04,
    STAGE_G05,
    STAGE_G06,
    STAGE_G07,
    STAGE_G08,
    STAGE_G09,
)

SEVERITY_BLOCKER = "blocker"
SEVERITY_WARNING = "warning"

STATUS_READY = "ready"
STATUS_BLOCKED = "blocked"

INTERIM_PATH_TOKEN = "artifacts/interim/"
DEFAULT_JSON_REPORT_PATH = "artifacts/reports/official_closure_audit.json"
DEFAULT_MARKDOWN_REPORT_PATH = "artifacts/reports/official_closure_audit.md"

DEFAULT_PROGRESS_PATH = "artifacts/reports/cpd_precompute_progress.csv"
DEFAULT_T15_TELEMETRY_PATH = "artifacts/reports/cpd_fit_telemetry.csv"
DEFAULT_T15_FAILURE_LEDGER_PATH = "artifacts/reports/cpd_failure_ledger.csv"
DEFAULT_T15_FALLBACK_LEDGER_PATH = "artifacts/reports/cpd_fallback_ledger.csv"
DEFAULT_T15_MANIFEST_PATH = "artifacts/manifests/cpd_feature_store_manifest.json"
DEFAULT_CANONICAL_MANIFEST_PATH = "artifacts/manifests/canonical_daily_close_manifest.json"
DEFAULT_DATASET_REGISTRY_PATH = "artifacts/manifests/dataset_registry.json"

DEFAULT_SMOKE_DIR = "artifacts/training/smoke_run"
DEFAULT_SMOKE_CONFIG_PATH = f"{DEFAULT_SMOKE_DIR}/smoke_config.json"
DEFAULT_SMOKE_MODEL_PATH = f"{DEFAULT_SMOKE_DIR}/smoke_best_model.keras"
DEFAULT_SMOKE_EPOCH_LOG_PATH = f"{DEFAULT_SMOKE_DIR}/smoke_epoch_log.csv"
DEFAULT_SMOKE_VALIDATION_HISTORY_PATH = (
    f"{DEFAULT_SMOKE_DIR}/smoke_validation_history.csv"
)
DEFAULT_MODEL_FIDELITY_REPORT_PATH = "artifacts/reports/model_fidelity_report.md"

DEFAULT_SEARCH_SCHEDULE_JSON_PATH = "artifacts/training/search_schedule.json"
DEFAULT_SEARCH_SCHEDULE_CSV_PATH = "artifacts/training/search_schedule.csv"
DEFAULT_SEARCH_COMPLETION_LOG_PATH = "artifacts/training/search_completion_log.csv"
DEFAULT_BEST_CANDIDATE_PATH = "artifacts/training/best_candidate.json"
DEFAULT_BEST_CONFIG_PATH = "artifacts/training/best_config.json"
DEFAULT_SEARCH_SUMMARY_REPORT_PATH = "artifacts/reports/search_summary_report.csv"

DEFAULT_LATEST_POSITIONS_PATH = "artifacts/inference/latest_positions.csv"
DEFAULT_LATEST_SEQUENCE_MANIFEST_PATH = (
    "artifacts/inference/latest_sequence_manifest.csv"
)
DEFAULT_RAW_VALIDATION_RETURNS_PATH = "artifacts/evaluation/raw_validation_returns.csv"
DEFAULT_RAW_VALIDATION_METRICS_PATH = "artifacts/evaluation/raw_validation_metrics.json"
DEFAULT_RESCALED_VALIDATION_RETURNS_PATH = (
    "artifacts/evaluation/rescaled_validation_returns.csv"
)
DEFAULT_RESCALED_VALIDATION_METRICS_PATH = (
    "artifacts/evaluation/rescaled_validation_metrics.json"
)
DEFAULT_EVALUATION_REPORT_PATH = "artifacts/evaluation/evaluation_report.md"
DEFAULT_REPRODUCIBILITY_MANIFEST_PATH = (
    "artifacts/reproducibility/reproducibility_manifest.json"
)

DEFAULT_NOTEBOOK_PATH = "notebooks/lstm_cpd_replication.ipynb"
DEFAULT_EXECUTED_NOTEBOOK_PATH = "notebooks/lstm_cpd_replication.executed.ipynb"
DEFAULT_NOTEBOOK_REPORT_PATH = "artifacts/notebook/notebook_execution_report.md"
DEFAULT_NOTEBOOK_ARTIFACT_MAP_PATH = "artifacts/notebook/notebook_artifact_map.csv"

SEARCH_SCHEDULE_FIELDS = (
    "candidate_index",
    "candidate_id",
    "dropout",
    "hidden_size",
    "minibatch_size",
    "learning_rate",
    "max_grad_norm",
    "lbw",
)
SEARCH_COMPLETION_LOG_HEADER = (
    "candidate_index",
    "candidate_id",
    "status",
    "failure_reason",
    "dropout",
    "hidden_size",
    "minibatch_size",
    "learning_rate",
    "max_grad_norm",
    "lbw",
    "best_validation_loss",
    "best_epoch_index",
    "epochs_completed",
    "dataset_registry_path",
    "candidate_dir",
    "config_path",
    "best_model_path",
    "epoch_log_path",
    "validation_history_path",
    "final_metadata_path",
)
SEARCH_SUMMARY_REPORT_HEADER = (
    "candidate_index",
    "candidate_id",
    "status",
    "selected",
    "best_validation_loss",
    "failure_reason",
    "dropout",
    "hidden_size",
    "minibatch_size",
    "learning_rate",
    "max_grad_norm",
    "lbw",
    "best_model_path",
    "final_metadata_path",
)
SMOKE_CONFIG_REQUIRED_FIELDS = (
    "candidate_id",
    "candidate_index",
    "dropout",
    "hidden_size",
    "minibatch_size",
    "learning_rate",
    "max_grad_norm",
    "lbw",
    "dataset_registry_path",
)
SMOKE_EPOCH_LOG_HEADER = (
    "epoch_index",
    "train_loss",
    "val_loss",
    "best_val_loss",
    "mean_gradient_norm",
    "improved",
)
SMOKE_VALIDATION_HISTORY_HEADER = (
    "epoch_index",
    "val_loss",
    "best_so_far",
    "improved_vs_previous",
    "improved_vs_best",
)
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
VALIDATION_RETURNS_HEADER = ("return_timestamp", "asset_count", "portfolio_return")
VALIDATION_METRIC_KEYS = (
    "annualized_return",
    "annualized_volatility",
    "annualized_downside_deviation",
    "sharpe_ratio",
    "sortino_ratio",
    "maximum_drawdown",
    "calmar_ratio",
    "percentage_positive_daily_returns",
)
REPRODUCIBILITY_REQUIRED_KEYS = (
    "run_id",
    "run_type",
    "created_at_utc",
    "project_root",
    "seed_policy",
    "selected_lbw",
    "selected_candidate_id",
    "artifact_locations",
    "source_artifacts",
    "status",
    "hashes",
    "entrypoints",
)


@dataclass(frozen=True)
class ClosureAuditFinding:
    stage_id: str
    code: str
    severity: str
    message: str
    path: str | None = None
    details: dict[str, object] | None = None


@dataclass(frozen=True)
class ClosureStageStatus:
    stage_id: str
    stage_name: str
    status: str
    blocker_count: int
    warning_count: int
    summary: str


@dataclass(frozen=True)
class OfficialClosureAudit:
    json_report_path: Path
    markdown_report_path: Path
    ready_for_project_closure: bool
    first_blocking_stage: str | None
    stages: tuple[ClosureStageStatus, ...]
    findings: tuple[ClosureAuditFinding, ...]


def default_json_report_path() -> Path:
    return default_project_root() / DEFAULT_JSON_REPORT_PATH


def default_markdown_report_path() -> Path:
    return default_project_root() / DEFAULT_MARKDOWN_REPORT_PATH


def _resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _write_json(path: Path | str, payload: object) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return json_path


def _write_text(path: Path | str, text: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _relative_text(path: Path, project_root: Path) -> str:
    return project_relative_path(path, project_root)


def _iter_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for nested_value in value.values():
            yield from _iter_strings(nested_value)
        return
    if isinstance(value, (list, tuple)):
        for nested_value in value:
            yield from _iter_strings(nested_value)


def _contains_interim_reference(value: object) -> bool:
    return any(INTERIM_PATH_TOKEN in item for item in _iter_strings(value))


def _is_placeholder_json_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return set(payload.keys()) == {"path"} and isinstance(payload["path"], str)


def _read_json(path: Path | str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv_header(path: Path | str) -> tuple[str, ...]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {csv_path}") from exc
    return tuple(header)


def _read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_placeholder_csv(path: Path | str) -> bool:
    csv_path = Path(path)
    text = csv_path.read_text(encoding="utf-8")
    return text == "column_a,column_b\nvalue_a,value_b\n"


def _path_from_field(project_root: Path, raw_path: str) -> Path:
    return _resolve_project_path(project_root, raw_path)


def _append_finding(
    findings: list[ClosureAuditFinding],
    *,
    stage_id: str,
    code: str,
    message: str,
    severity: str = SEVERITY_BLOCKER,
    path: Path | str | None = None,
    project_root: Path | None = None,
    details: dict[str, object] | None = None,
) -> None:
    path_text: str | None = None
    if path is not None:
        path_obj = Path(path)
        if project_root is not None:
            path_text = _relative_text(path_obj, project_root)
        else:
            path_text = str(path_obj)
    findings.append(
        ClosureAuditFinding(
            stage_id=stage_id,
            code=code,
            severity=severity,
            message=message,
            path=path_text,
            details=details,
        )
    )


def _require_file(
    findings: list[ClosureAuditFinding],
    *,
    stage_id: str,
    project_root: Path,
    path: Path,
    code: str,
    label: str,
) -> bool:
    if not path.exists():
        _append_finding(
            findings,
            stage_id=stage_id,
            code=code,
            message=f"{label} is missing.",
            path=path,
            project_root=project_root,
        )
        return False
    if not path.is_file():
        _append_finding(
            findings,
            stage_id=stage_id,
            code=code,
            message=f"{label} is not a file.",
            path=path,
            project_root=project_root,
        )
        return False
    return True


def _validate_csv_header(
    findings: list[ClosureAuditFinding],
    *,
    stage_id: str,
    project_root: Path,
    path: Path,
    expected_header: Sequence[str],
    code: str,
    label: str,
) -> bool:
    if not _require_file(
        findings,
        stage_id=stage_id,
        project_root=project_root,
        path=path,
        code=code,
        label=label,
    ):
        return False
    if _is_placeholder_csv(path):
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_PLACEHOLDER",
            message=f"{label} is still a placeholder fixture CSV.",
            path=path,
            project_root=project_root,
        )
        return False
    try:
        header = _read_csv_header(path)
    except Exception as exc:
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_UNREADABLE",
            message=f"{label} could not be read as CSV: {exc}",
            path=path,
            project_root=project_root,
        )
        return False
    if tuple(expected_header) != tuple(header):
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_HEADER",
            message=f"{label} has an unexpected CSV header.",
            path=path,
            project_root=project_root,
            details={"expected_header": list(expected_header), "actual_header": list(header)},
        )
        return False
    return True


def _validate_json_object(
    findings: list[ClosureAuditFinding],
    *,
    stage_id: str,
    project_root: Path,
    path: Path,
    code: str,
    label: str,
    required_keys: Sequence[str] = (),
) -> dict[str, object] | None:
    if not _require_file(
        findings,
        stage_id=stage_id,
        project_root=project_root,
        path=path,
        code=code,
        label=label,
    ):
        return None
    try:
        payload = _read_json(path)
    except Exception as exc:
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_UNREADABLE",
            message=f"{label} could not be parsed as JSON: {exc}",
            path=path,
            project_root=project_root,
        )
        return None
    if _is_placeholder_json_payload(payload):
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_PLACEHOLDER",
            message=f"{label} is still a placeholder fixture JSON object.",
            path=path,
            project_root=project_root,
        )
        return None
    if not isinstance(payload, dict):
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_TYPE",
            message=f"{label} must contain a JSON object.",
            path=path,
            project_root=project_root,
        )
        return None
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_FIELDS",
            message=f"{label} is missing required keys.",
            path=path,
            project_root=project_root,
            details={"missing_keys": missing_keys},
        )
        return None
    return payload


def _validate_json_list(
    findings: list[ClosureAuditFinding],
    *,
    stage_id: str,
    project_root: Path,
    path: Path,
    code: str,
    label: str,
) -> list[object] | None:
    if not _require_file(
        findings,
        stage_id=stage_id,
        project_root=project_root,
        path=path,
        code=code,
        label=label,
    ):
        return None
    try:
        payload = _read_json(path)
    except Exception as exc:
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_UNREADABLE",
            message=f"{label} could not be parsed as JSON: {exc}",
            path=path,
            project_root=project_root,
        )
        return None
    if _is_placeholder_json_payload(payload):
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_PLACEHOLDER",
            message=f"{label} is still a placeholder fixture JSON object.",
            path=path,
            project_root=project_root,
        )
        return None
    if not isinstance(payload, list):
        _append_finding(
            findings,
            stage_id=stage_id,
            code=f"{code}_TYPE",
            message=f"{label} must contain a JSON array.",
            path=path,
            project_root=project_root,
        )
        return None
    return payload


def _validate_no_interim_reference(
    findings: list[ClosureAuditFinding],
    *,
    stage_id: str,
    project_root: Path,
    path: Path,
    payload: object,
    code: str,
    label: str,
) -> bool:
    if _contains_interim_reference(payload):
        _append_finding(
            findings,
            stage_id=stage_id,
            code=code,
            message=f"{label} still references interim artifacts.",
            path=path,
            project_root=project_root,
        )
        return False
    return True


def _validate_metrics_json(
    findings: list[ClosureAuditFinding],
    *,
    stage_id: str,
    project_root: Path,
    path: Path,
    code: str,
    label: str,
) -> bool:
    payload = _validate_json_object(
        findings,
        stage_id=stage_id,
        project_root=project_root,
        path=path,
        code=code,
        label=label,
        required_keys=VALIDATION_METRIC_KEYS,
    )
    if payload is None:
        return False
    if not _validate_no_interim_reference(
        findings,
        stage_id=stage_id,
        project_root=project_root,
        path=path,
        payload=payload,
        code=f"{code}_INTERIM",
        label=label,
    ):
        return False
    for key in VALIDATION_METRIC_KEYS:
        value = payload[key]
        if value is None and key == "calmar_ratio":
            continue
        if not isinstance(value, (int, float)):
            _append_finding(
                findings,
                stage_id=stage_id,
                code=f"{code}_VALUE",
                message=f"{label} has a non-numeric metric value for {key}.",
                path=path,
                project_root=project_root,
            )
            return False
    return True


def _check_g04(project_root: Path) -> list[ClosureAuditFinding]:
    findings: list[ClosureAuditFinding] = []
    progress_path = _resolve_project_path(project_root, DEFAULT_PROGRESS_PATH)
    canonical_manifest_path = _resolve_project_path(project_root, DEFAULT_CANONICAL_MANIFEST_PATH)
    telemetry_path = _resolve_project_path(project_root, DEFAULT_T15_TELEMETRY_PATH)
    failure_ledger_path = _resolve_project_path(project_root, DEFAULT_T15_FAILURE_LEDGER_PATH)
    fallback_ledger_path = _resolve_project_path(project_root, DEFAULT_T15_FALLBACK_LEDGER_PATH)
    manifest_path = _resolve_project_path(project_root, DEFAULT_T15_MANIFEST_PATH)

    expected_pairs: set[tuple[str, int]] = set()
    canonical_records = None
    if _require_file(
        findings,
        stage_id=STAGE_G04,
        project_root=project_root,
        path=canonical_manifest_path,
        code="G04_CANONICAL_MANIFEST_MISSING",
        label="canonical_daily_close_manifest.json",
    ):
        try:
            canonical_records = load_canonical_daily_close_manifest(canonical_manifest_path)
        except Exception as exc:
            _append_finding(
                findings,
                stage_id=STAGE_G04,
                code="G04_CANONICAL_MANIFEST_INVALID",
                message=f"canonical_daily_close_manifest.json is invalid: {exc}",
                path=canonical_manifest_path,
                project_root=project_root,
            )
        else:
            expected_pairs = {
                (record.asset_id, lbw)
                for record in canonical_records
                for lbw in ALLOWED_CPD_LBWS
            }

    if _validate_csv_header(
        findings,
        stage_id=STAGE_G04,
        project_root=project_root,
        path=progress_path,
        expected_header=(
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
        code="G04_PROGRESS",
        label="cpd_precompute_progress.csv",
    ):
        rows = _read_csv_rows(progress_path)
        counts = {"completed": 0, "running": 0, "pending": 0}
        seen_pairs: set[tuple[str, int]] = set()
        for row in rows:
            state = row["state"]
            if state in counts:
                counts[state] += 1
            pair = (row["asset_id"], int(row["lbw"]))
            seen_pairs.add(pair)
        if counts["running"] != 0 or counts["pending"] != 0:
            _append_finding(
                findings,
                stage_id=STAGE_G04,
                code="G04_T14_INCOMPLETE",
                message=(
                    "T-14 is still materializing; cpd_precompute_progress.csv contains "
                    f"{counts['completed']} completed, {counts['running']} running, "
                    f"and {counts['pending']} pending rows."
                ),
                path=progress_path,
                project_root=project_root,
                details=counts,
            )
        if expected_pairs and seen_pairs != expected_pairs:
            missing_pairs = sorted(expected_pairs - seen_pairs)
            extra_pairs = sorted(seen_pairs - expected_pairs)
            _append_finding(
                findings,
                stage_id=STAGE_G04,
                code="G04_PROGRESS_COVERAGE",
                message="cpd_precompute_progress.csv does not match canonical asset/LBW coverage.",
                path=progress_path,
                project_root=project_root,
                details={
                    "missing_pairs": [list(item) for item in missing_pairs[:10]],
                    "extra_pairs": [list(item) for item in extra_pairs[:10]],
                    "expected_pair_count": len(expected_pairs),
                    "actual_pair_count": len(seen_pairs),
                },
            )

    partial_paths = sorted(
        (project_root / "artifacts/features/cpd").rglob("*.partial.csv")
    )
    if partial_paths:
        _append_finding(
            findings,
            stage_id=STAGE_G04,
            code="G04_PARTIAL_OUTPUTS",
            message="Final CPD feature-store still contains .partial.csv files.",
            path=partial_paths[0],
            project_root=project_root,
            details={"partial_file_count": len(partial_paths)},
        )

    _validate_csv_header(
        findings,
        stage_id=STAGE_G04,
        project_root=project_root,
        path=telemetry_path,
        expected_header=TELEMETRY_HEADER,
        code="G04_T15_TELEMETRY",
        label="cpd_fit_telemetry.csv",
    )
    _validate_csv_header(
        findings,
        stage_id=STAGE_G04,
        project_root=project_root,
        path=failure_ledger_path,
        expected_header=FAILURE_LEDGER_HEADER,
        code="G04_T15_FAILURE_LEDGER",
        label="cpd_failure_ledger.csv",
    )
    _validate_csv_header(
        findings,
        stage_id=STAGE_G04,
        project_root=project_root,
        path=fallback_ledger_path,
        expected_header=FALLBACK_LEDGER_HEADER,
        code="G04_T15_FALLBACK_LEDGER",
        label="cpd_fallback_ledger.csv",
    )

    manifest_rows = _validate_json_list(
        findings,
        stage_id=STAGE_G04,
        project_root=project_root,
        path=manifest_path,
        code="G04_T15_MANIFEST",
        label="cpd_feature_store_manifest.json",
    )
    if manifest_rows is None:
        return findings
    try:
        manifest_records = load_cpd_feature_store_manifest(manifest_path)
    except Exception as exc:
        _append_finding(
            findings,
            stage_id=STAGE_G04,
            code="G04_T15_MANIFEST_INVALID",
            message=f"cpd_feature_store_manifest.json is invalid: {exc}",
            path=manifest_path,
            project_root=project_root,
        )
        return findings

    if not _validate_no_interim_reference(
        findings,
        stage_id=STAGE_G04,
        project_root=project_root,
        path=manifest_path,
        payload=manifest_rows,
        code="G04_T15_MANIFEST_INTERIM",
        label="cpd_feature_store_manifest.json",
    ):
        return findings

    manifest_pairs = {(record.asset_id, record.lbw) for record in manifest_records}
    if expected_pairs and manifest_pairs != expected_pairs:
        _append_finding(
            findings,
            stage_id=STAGE_G04,
            code="G04_T15_MANIFEST_COVERAGE",
            message="cpd_feature_store_manifest.json does not cover every canonical asset/LBW pair.",
            path=manifest_path,
            project_root=project_root,
            details={
                "expected_pair_count": len(expected_pairs),
                "actual_pair_count": len(manifest_pairs),
            },
        )
    for record in manifest_records:
        if (
            record.state != "present"
            or record.missing_reason is not None
            or record.cpd_csv_path == ""
            or not record.matches_canonical_timeline
            or record.file_hash is None
            or record.row_count != record.canonical_row_count
        ):
            _append_finding(
                findings,
                stage_id=STAGE_G04,
                code="G04_T15_MANIFEST_INCOMPLETE",
                message=(
                    "cpd_feature_store_manifest.json still contains incomplete CPD entries."
                ),
                path=manifest_path,
                project_root=project_root,
                details={
                    "asset_id": record.asset_id,
                    "lbw": record.lbw,
                    "state": record.state,
                    "missing_reason": record.missing_reason,
                },
            )
            break
        cpd_csv_path = _path_from_field(project_root, record.cpd_csv_path)
        if not cpd_csv_path.exists():
            _append_finding(
                findings,
                stage_id=STAGE_G04,
                code="G04_T15_FINAL_OUTPUT_MISSING",
                message="A final CPD CSV referenced by cpd_feature_store_manifest.json is missing.",
                path=cpd_csv_path,
                project_root=project_root,
            )
            break
    return findings


def _validate_dataset_registry_entry(
    findings: list[ClosureAuditFinding],
    *,
    project_root: Path,
    registry_path: Path,
    row: dict[str, object],
    lbw: int,
) -> None:
    if not isinstance(row, dict):
        _append_finding(
            findings,
            stage_id=STAGE_G05,
            code="G05_DATASET_REGISTRY_ROW_TYPE",
            message="dataset_registry.json contains a non-object row.",
            path=registry_path,
            project_root=project_root,
        )
        return
    required_keys = {
        "lbw",
        "feature_columns",
        "sequence_length",
        "train_sequence_count",
        "val_sequence_count",
        "train_input_shape",
        "train_target_shape",
        "val_input_shape",
        "val_target_shape",
        "artifacts",
        "source_artifacts",
    }
    missing_keys = sorted(required_keys - set(row))
    if missing_keys:
        _append_finding(
            findings,
            stage_id=STAGE_G05,
            code="G05_DATASET_REGISTRY_ROW_FIELDS",
            message="dataset_registry.json contains a row with missing fields.",
            path=registry_path,
            project_root=project_root,
            details={"lbw": lbw, "missing_keys": missing_keys},
        )
        return
    if int(row["lbw"]) != lbw:
        _append_finding(
            findings,
            stage_id=STAGE_G05,
            code="G05_DATASET_REGISTRY_LBW",
            message="dataset_registry.json contains an LBW mismatch.",
            path=registry_path,
            project_root=project_root,
            details={"expected_lbw": lbw, "actual_lbw": row["lbw"]},
        )
        return
    feature_columns = tuple(str(item) for item in row["feature_columns"])
    if feature_columns != MODEL_INPUT_COLUMNS:
        _append_finding(
            findings,
            stage_id=STAGE_G05,
            code="G05_DATASET_REGISTRY_FEATURES",
            message="dataset_registry.json has an unexpected feature column order.",
            path=registry_path,
            project_root=project_root,
            details={"lbw": lbw},
        )
        return
    if int(row["sequence_length"]) != SEQUENCE_LENGTH:
        _append_finding(
            findings,
            stage_id=STAGE_G05,
            code="G05_DATASET_REGISTRY_SEQUENCE_LENGTH",
            message="dataset_registry.json has an unexpected sequence_length.",
            path=registry_path,
            project_root=project_root,
            details={"lbw": lbw, "sequence_length": row["sequence_length"]},
        )
        return
    if _contains_interim_reference(row):
        _append_finding(
            findings,
            stage_id=STAGE_G05,
            code="G05_DATASET_REGISTRY_INTERIM",
            message="dataset_registry.json still references interim artifacts.",
            path=registry_path,
            project_root=project_root,
            details={"lbw": lbw},
        )
        return
    artifacts = row["artifacts"]
    source_artifacts = row["source_artifacts"]
    if not isinstance(artifacts, dict) or not isinstance(source_artifacts, dict):
        _append_finding(
            findings,
            stage_id=STAGE_G05,
            code="G05_DATASET_REGISTRY_GROUPS",
            message="dataset_registry.json artifacts/source_artifacts groups are malformed.",
            path=registry_path,
            project_root=project_root,
            details={"lbw": lbw},
        )
        return

    array_specs = (
        ("train_inputs_path", tuple(int(item) for item in row["train_input_shape"])),
        ("train_target_scale_path", tuple(int(item) for item in row["train_target_shape"])),
        ("val_inputs_path", tuple(int(item) for item in row["val_input_shape"])),
        ("val_target_scale_path", tuple(int(item) for item in row["val_target_shape"])),
    )
    for artifact_key, expected_shape in array_specs:
        raw_path = artifacts.get(artifact_key)
        if not isinstance(raw_path, str):
            _append_finding(
                findings,
                stage_id=STAGE_G05,
                code="G05_DATASET_REGISTRY_ARRAY_PATH",
                message=f"dataset_registry.json is missing {artifact_key}.",
                path=registry_path,
                project_root=project_root,
                details={"lbw": lbw},
            )
            return
        artifact_path = _path_from_field(project_root, raw_path)
        if not artifact_path.exists():
            _append_finding(
                findings,
                stage_id=STAGE_G05,
                code="G05_DATASET_ARRAY_MISSING",
                message=f"Dataset array {artifact_key} is missing.",
                path=artifact_path,
                project_root=project_root,
                details={"lbw": lbw},
            )
            return
        try:
            actual_array = np.load(artifact_path)
        except Exception as exc:
            _append_finding(
                findings,
                stage_id=STAGE_G05,
                code="G05_DATASET_ARRAY_INVALID",
                message=f"Dataset array {artifact_key} could not be loaded: {exc}",
                path=artifact_path,
                project_root=project_root,
                details={"lbw": lbw},
            )
            return
        if tuple(actual_array.shape) != expected_shape:
            _append_finding(
                findings,
                stage_id=STAGE_G05,
                code="G05_DATASET_ARRAY_SHAPE",
                message=f"Dataset array {artifact_key} has an unexpected shape.",
                path=artifact_path,
                project_root=project_root,
                details={
                    "lbw": lbw,
                    "expected_shape": list(expected_shape),
                    "actual_shape": list(actual_array.shape),
                },
            )
            return

    split_manifest_path = _path_from_field(
        project_root, str(source_artifacts.get("split_manifest_path", ""))
    )
    if _validate_csv_header(
        findings,
        stage_id=STAGE_G05,
        project_root=project_root,
        path=split_manifest_path,
        expected_header=T16_SPLIT_MANIFEST_HEADER,
        code="G05_SPLIT_MANIFEST",
        label=f"split_manifest.csv for lbw={lbw}",
    ):
        try:
            load_split_manifest_csv(split_manifest_path, expected_lbw=lbw)
        except Exception as exc:
            _append_finding(
                findings,
                stage_id=STAGE_G05,
                code="G05_SPLIT_MANIFEST_INVALID",
                message=f"split_manifest.csv is invalid for lbw={lbw}: {exc}",
                path=split_manifest_path,
                project_root=project_root,
            )

    sequence_manifest_path = _path_from_field(
        project_root, str(source_artifacts.get("sequence_manifest_path", ""))
    )
    if _validate_csv_header(
        findings,
        stage_id=STAGE_G05,
        project_root=project_root,
        path=sequence_manifest_path,
        expected_header=T17_SEQUENCE_MANIFEST_HEADER,
        code="G05_SEQUENCE_MANIFEST",
        label=f"sequence_manifest.csv for lbw={lbw}",
    ):
        try:
            load_sequence_manifest_csv(sequence_manifest_path, expected_lbw=lbw)
        except Exception as exc:
            _append_finding(
                findings,
                stage_id=STAGE_G05,
                code="G05_SEQUENCE_MANIFEST_INVALID",
                message=f"sequence_manifest.csv is invalid for lbw={lbw}: {exc}",
                path=sequence_manifest_path,
                project_root=project_root,
            )

    target_alignment_path = _path_from_field(
        project_root, str(source_artifacts.get("target_alignment_registry_path", ""))
    )
    if _validate_csv_header(
        findings,
        stage_id=STAGE_G05,
        project_root=project_root,
        path=target_alignment_path,
        expected_header=T17_TARGET_ALIGNMENT_HEADER,
        code="G05_TARGET_ALIGNMENT",
        label=f"target_alignment_registry.csv for lbw={lbw}",
    ):
        try:
            load_target_alignment_registry_csv(target_alignment_path, expected_lbw=lbw)
        except Exception as exc:
            _append_finding(
                findings,
                stage_id=STAGE_G05,
                code="G05_TARGET_ALIGNMENT_INVALID",
                message=f"target_alignment_registry.csv is invalid for lbw={lbw}: {exc}",
                path=target_alignment_path,
                project_root=project_root,
            )


def _check_g05(project_root: Path) -> list[ClosureAuditFinding]:
    findings: list[ClosureAuditFinding] = []
    registry_path = _resolve_project_path(project_root, DEFAULT_DATASET_REGISTRY_PATH)
    registry_rows = _validate_json_list(
        findings,
        stage_id=STAGE_G05,
        project_root=project_root,
        path=registry_path,
        code="G05_DATASET_REGISTRY",
        label="dataset_registry.json",
    )
    if registry_rows is None:
        return findings
    if _contains_interim_reference(registry_rows):
        _append_finding(
            findings,
            stage_id=STAGE_G05,
            code="G05_DATASET_REGISTRY_INTERIM",
            message="dataset_registry.json still references interim artifacts.",
            path=registry_path,
            project_root=project_root,
        )
        return findings
    rows_by_lbw: dict[int, list[object]] = {}
    for row in registry_rows:
        if isinstance(row, dict) and "lbw" in row:
            rows_by_lbw.setdefault(int(row["lbw"]), []).append(row)
    for lbw in ALLOWED_CPD_LBWS:
        matching_rows = rows_by_lbw.get(lbw, [])
        if len(matching_rows) != 1:
            _append_finding(
                findings,
                stage_id=STAGE_G05,
                code="G05_DATASET_REGISTRY_COVERAGE",
                message="dataset_registry.json does not contain exactly one row for every official LBW.",
                path=registry_path,
                project_root=project_root,
                details={"lbw": lbw, "row_count": len(matching_rows)},
            )
            continue
        _validate_dataset_registry_entry(
            findings,
            project_root=project_root,
            registry_path=registry_path,
            row=matching_rows[0],
            lbw=lbw,
        )
    return findings


def _check_g06(project_root: Path) -> list[ClosureAuditFinding]:
    findings: list[ClosureAuditFinding] = []
    smoke_config_path = _resolve_project_path(project_root, DEFAULT_SMOKE_CONFIG_PATH)
    smoke_model_path = _resolve_project_path(project_root, DEFAULT_SMOKE_MODEL_PATH)
    smoke_epoch_log_path = _resolve_project_path(project_root, DEFAULT_SMOKE_EPOCH_LOG_PATH)
    smoke_validation_history_path = _resolve_project_path(
        project_root,
        DEFAULT_SMOKE_VALIDATION_HISTORY_PATH,
    )
    model_fidelity_report_path = _resolve_project_path(
        project_root,
        DEFAULT_MODEL_FIDELITY_REPORT_PATH,
    )

    smoke_config = _validate_json_object(
        findings,
        stage_id=STAGE_G06,
        project_root=project_root,
        path=smoke_config_path,
        code="G06_SMOKE_CONFIG",
        label="smoke_config.json",
        required_keys=SMOKE_CONFIG_REQUIRED_FIELDS,
    )
    if smoke_config is not None:
        _validate_no_interim_reference(
            findings,
            stage_id=STAGE_G06,
            project_root=project_root,
            path=smoke_config_path,
            payload=smoke_config,
            code="G06_SMOKE_CONFIG_INTERIM",
            label="smoke_config.json",
        )
    _require_file(
        findings,
        stage_id=STAGE_G06,
        project_root=project_root,
        path=smoke_model_path,
        code="G06_SMOKE_MODEL_MISSING",
        label="smoke_best_model.keras",
    )
    _validate_csv_header(
        findings,
        stage_id=STAGE_G06,
        project_root=project_root,
        path=smoke_epoch_log_path,
        expected_header=SMOKE_EPOCH_LOG_HEADER,
        code="G06_SMOKE_EPOCH_LOG",
        label="smoke_epoch_log.csv",
    )
    _validate_csv_header(
        findings,
        stage_id=STAGE_G06,
        project_root=project_root,
        path=smoke_validation_history_path,
        expected_header=SMOKE_VALIDATION_HISTORY_HEADER,
        code="G06_SMOKE_VALIDATION_HISTORY",
        label="smoke_validation_history.csv",
    )
    if _require_file(
        findings,
        stage_id=STAGE_G06,
        project_root=project_root,
        path=model_fidelity_report_path,
        code="G06_MODEL_FIDELITY_REPORT",
        label="model_fidelity_report.md",
    ):
        report_text = model_fidelity_report_path.read_text(encoding="utf-8")
        if "# Model Fidelity Report" not in report_text:
            _append_finding(
                findings,
                stage_id=STAGE_G06,
                code="G06_MODEL_FIDELITY_REPORT_CONTENT",
                message="model_fidelity_report.md does not look like an official smoke fidelity report.",
                path=model_fidelity_report_path,
                project_root=project_root,
            )
        if INTERIM_PATH_TOKEN in report_text:
            _append_finding(
                findings,
                stage_id=STAGE_G06,
                code="G06_MODEL_FIDELITY_REPORT_INTERIM",
                message="model_fidelity_report.md still references interim artifacts.",
                path=model_fidelity_report_path,
                project_root=project_root,
            )
    return findings


def _validate_search_schedule_rows(
    findings: list[ClosureAuditFinding],
    *,
    project_root: Path,
    schedule_path: Path,
    rows: list[object],
) -> list[dict[str, object]]:
    parsed_rows: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    if len(rows) != 50:
        _append_finding(
            findings,
            stage_id=STAGE_G07,
            code="G07_SEARCH_SCHEDULE_SIZE",
            message="search_schedule.json does not contain exactly 50 candidate rows.",
            path=schedule_path,
            project_root=project_root,
            details={"row_count": len(rows)},
        )
    for row_position, row in enumerate(rows):
        if not isinstance(row, dict):
            _append_finding(
                findings,
                stage_id=STAGE_G07,
                code="G07_SEARCH_SCHEDULE_ROW_TYPE",
                message="search_schedule.json contains a non-object row.",
                path=schedule_path,
                project_root=project_root,
            )
            continue
        missing_keys = [key for key in SEARCH_SCHEDULE_FIELDS if key not in row]
        if missing_keys:
            _append_finding(
                findings,
                stage_id=STAGE_G07,
                code="G07_SEARCH_SCHEDULE_FIELDS",
                message="search_schedule.json contains a row with missing candidate fields.",
                path=schedule_path,
                project_root=project_root,
                details={"missing_keys": missing_keys, "row_position": row_position},
            )
            continue
        candidate_index = int(row["candidate_index"])
        candidate_id = str(row["candidate_id"])
        if candidate_index != row_position:
            _append_finding(
                findings,
                stage_id=STAGE_G07,
                code="G07_SEARCH_SCHEDULE_INDEX_ORDER",
                message="search_schedule.json candidate_index values are not zero-based row positions.",
                path=schedule_path,
                project_root=project_root,
                details={"row_position": row_position, "candidate_index": candidate_index},
            )
        if candidate_index in seen_indices or candidate_id in seen_ids:
            _append_finding(
                findings,
                stage_id=STAGE_G07,
                code="G07_SEARCH_SCHEDULE_DUPLICATE",
                message="search_schedule.json contains duplicate candidate identifiers.",
                path=schedule_path,
                project_root=project_root,
                details={"candidate_index": candidate_index, "candidate_id": candidate_id},
            )
            continue
        seen_indices.add(candidate_index)
        seen_ids.add(candidate_id)
        parsed_rows.append(row)
    return parsed_rows


def _check_g07(project_root: Path) -> list[ClosureAuditFinding]:
    findings: list[ClosureAuditFinding] = []
    schedule_json_path = _resolve_project_path(project_root, DEFAULT_SEARCH_SCHEDULE_JSON_PATH)
    schedule_csv_path = _resolve_project_path(project_root, DEFAULT_SEARCH_SCHEDULE_CSV_PATH)
    completion_log_path = _resolve_project_path(project_root, DEFAULT_SEARCH_COMPLETION_LOG_PATH)
    best_candidate_path = _resolve_project_path(project_root, DEFAULT_BEST_CANDIDATE_PATH)
    best_config_path = _resolve_project_path(project_root, DEFAULT_BEST_CONFIG_PATH)
    search_summary_report_path = _resolve_project_path(
        project_root,
        DEFAULT_SEARCH_SUMMARY_REPORT_PATH,
    )

    schedule_rows = _validate_json_list(
        findings,
        stage_id=STAGE_G07,
        project_root=project_root,
        path=schedule_json_path,
        code="G07_SEARCH_SCHEDULE_JSON",
        label="search_schedule.json",
    )
    parsed_schedule_rows: list[dict[str, object]] = []
    if schedule_rows is not None:
        parsed_schedule_rows = _validate_search_schedule_rows(
            findings,
            project_root=project_root,
            schedule_path=schedule_json_path,
            rows=schedule_rows,
        )
        _validate_no_interim_reference(
            findings,
            stage_id=STAGE_G07,
            project_root=project_root,
            path=schedule_json_path,
            payload=schedule_rows,
            code="G07_SEARCH_SCHEDULE_INTERIM",
            label="search_schedule.json",
        )
    _validate_csv_header(
        findings,
        stage_id=STAGE_G07,
        project_root=project_root,
        path=schedule_csv_path,
        expected_header=SEARCH_SCHEDULE_FIELDS,
        code="G07_SEARCH_SCHEDULE_CSV",
        label="search_schedule.csv",
    )
    if _validate_csv_header(
        findings,
        stage_id=STAGE_G07,
        project_root=project_root,
        path=completion_log_path,
        expected_header=SEARCH_COMPLETION_LOG_HEADER,
        code="G07_SEARCH_COMPLETION_LOG",
        label="search_completion_log.csv",
    ):
        completion_rows = _read_csv_rows(completion_log_path)
        if parsed_schedule_rows and len(completion_rows) != len(parsed_schedule_rows):
            _append_finding(
                findings,
                stage_id=STAGE_G07,
                code="G07_SEARCH_COMPLETION_LOG_COVERAGE",
                message="search_completion_log.csv does not contain one row per scheduled candidate.",
                path=completion_log_path,
                project_root=project_root,
                details={
                    "schedule_rows": len(parsed_schedule_rows),
                    "completion_rows": len(completion_rows),
                },
            )
        completed_count = 0
        seen_indices: set[int] = set()
        schedule_by_index = {
            int(row["candidate_index"]): row for row in parsed_schedule_rows
        }
        for row in completion_rows:
            candidate_index = int(row["candidate_index"])
            candidate_id = str(row["candidate_id"])
            if candidate_index in seen_indices:
                _append_finding(
                    findings,
                    stage_id=STAGE_G07,
                    code="G07_SEARCH_COMPLETION_LOG_DUPLICATE",
                    message="search_completion_log.csv contains duplicate candidate_index rows.",
                    path=completion_log_path,
                    project_root=project_root,
                    details={"candidate_index": candidate_index},
                )
                continue
            seen_indices.add(candidate_index)
            schedule_row = schedule_by_index.get(candidate_index)
            if schedule_row is None or str(schedule_row["candidate_id"]) != candidate_id:
                _append_finding(
                    findings,
                    stage_id=STAGE_G07,
                    code="G07_SEARCH_COMPLETION_LOG_SCHEDULE_MISMATCH",
                    message="search_completion_log.csv does not align with search_schedule.json.",
                    path=completion_log_path,
                    project_root=project_root,
                    details={"candidate_index": candidate_index, "candidate_id": candidate_id},
                )
                continue
            if row["status"] not in {"completed", "failed"}:
                _append_finding(
                    findings,
                    stage_id=STAGE_G07,
                    code="G07_SEARCH_COMPLETION_LOG_STATUS",
                    message="search_completion_log.csv contains an unsupported status.",
                    path=completion_log_path,
                    project_root=project_root,
                    details={"candidate_index": candidate_index, "status": row["status"]},
                )
                continue
            if INTERIM_PATH_TOKEN in json.dumps(row):
                _append_finding(
                    findings,
                    stage_id=STAGE_G07,
                    code="G07_SEARCH_COMPLETION_LOG_INTERIM",
                    message="search_completion_log.csv still references interim artifacts.",
                    path=completion_log_path,
                    project_root=project_root,
                    details={"candidate_index": candidate_index},
                )
                continue
            if row["status"] == "completed":
                completed_count += 1
                for field_name in (
                    "config_path",
                    "best_model_path",
                    "epoch_log_path",
                    "validation_history_path",
                    "final_metadata_path",
                ):
                    field_path = _path_from_field(project_root, row[field_name])
                    if not field_path.exists():
                        _append_finding(
                            findings,
                            stage_id=STAGE_G07,
                            code="G07_SEARCH_COMPLETION_ARTIFACT_MISSING",
                            message=f"A completed search candidate is missing {field_name}.",
                            path=field_path,
                            project_root=project_root,
                            details={"candidate_index": candidate_index},
                        )
                        break
        if completion_rows and completed_count == 0:
            _append_finding(
                findings,
                stage_id=STAGE_G07,
                code="G07_SEARCH_COMPLETION_NO_SUCCESS",
                message="search_completion_log.csv has no successful candidates.",
                path=completion_log_path,
                project_root=project_root,
            )

    best_candidate_payload = _validate_json_object(
        findings,
        stage_id=STAGE_G07,
        project_root=project_root,
        path=best_candidate_path,
        code="G07_BEST_CANDIDATE",
        label="best_candidate.json",
        required_keys=(
            "candidate_id",
            "candidate_index",
            "dropout",
            "hidden_size",
            "minibatch_size",
            "learning_rate",
            "max_grad_norm",
            "lbw",
            "dataset_registry_path",
            "best_model_path",
        ),
    )
    best_config_payload = _validate_json_object(
        findings,
        stage_id=STAGE_G07,
        project_root=project_root,
        path=best_config_path,
        code="G07_BEST_CONFIG",
        label="best_config.json",
        required_keys=(
            "candidate_id",
            "candidate_index",
            "dropout",
            "hidden_size",
            "minibatch_size",
            "learning_rate",
            "max_grad_norm",
            "lbw",
        ),
    )
    if best_candidate_payload is not None:
        _validate_no_interim_reference(
            findings,
            stage_id=STAGE_G07,
            project_root=project_root,
            path=best_candidate_path,
            payload=best_candidate_payload,
            code="G07_BEST_CANDIDATE_INTERIM",
            label="best_candidate.json",
        )
        best_model_path = _path_from_field(
            project_root,
            str(best_candidate_payload["best_model_path"]),
        )
        _require_file(
            findings,
            stage_id=STAGE_G07,
            project_root=project_root,
            path=best_model_path,
            code="G07_BEST_MODEL_MISSING",
            label="best candidate checkpoint",
        )
    if best_config_payload is not None:
        _validate_no_interim_reference(
            findings,
            stage_id=STAGE_G07,
            project_root=project_root,
            path=best_config_path,
            payload=best_config_payload,
            code="G07_BEST_CONFIG_INTERIM",
            label="best_config.json",
        )
    if (
        best_candidate_payload is not None
        and best_config_payload is not None
        and any(
            best_candidate_payload[key] != best_config_payload[key]
            for key in (
                "candidate_id",
                "candidate_index",
                "dropout",
                "hidden_size",
                "minibatch_size",
                "learning_rate",
                "max_grad_norm",
                "lbw",
            )
        )
    ):
        _append_finding(
            findings,
            stage_id=STAGE_G07,
            code="G07_BEST_SELECTION_MISMATCH",
            message="best_candidate.json and best_config.json do not describe the same candidate.",
            path=best_candidate_path,
            project_root=project_root,
        )

    _validate_csv_header(
        findings,
        stage_id=STAGE_G07,
        project_root=project_root,
        path=search_summary_report_path,
        expected_header=SEARCH_SUMMARY_REPORT_HEADER,
        code="G07_SEARCH_SUMMARY_REPORT",
        label="search_summary_report.csv",
    )
    return findings


def _check_g08(project_root: Path) -> list[ClosureAuditFinding]:
    findings: list[ClosureAuditFinding] = []
    latest_positions_path = _resolve_project_path(project_root, DEFAULT_LATEST_POSITIONS_PATH)
    latest_sequence_manifest_path = _resolve_project_path(
        project_root,
        DEFAULT_LATEST_SEQUENCE_MANIFEST_PATH,
    )
    raw_returns_path = _resolve_project_path(project_root, DEFAULT_RAW_VALIDATION_RETURNS_PATH)
    raw_metrics_path = _resolve_project_path(project_root, DEFAULT_RAW_VALIDATION_METRICS_PATH)
    rescaled_returns_path = _resolve_project_path(
        project_root,
        DEFAULT_RESCALED_VALIDATION_RETURNS_PATH,
    )
    rescaled_metrics_path = _resolve_project_path(
        project_root,
        DEFAULT_RESCALED_VALIDATION_METRICS_PATH,
    )
    evaluation_report_path = _resolve_project_path(project_root, DEFAULT_EVALUATION_REPORT_PATH)
    reproducibility_manifest_path = _resolve_project_path(
        project_root,
        DEFAULT_REPRODUCIBILITY_MANIFEST_PATH,
    )

    _validate_csv_header(
        findings,
        stage_id=STAGE_G08,
        project_root=project_root,
        path=latest_positions_path,
        expected_header=LATEST_POSITIONS_HEADER,
        code="G08_LATEST_POSITIONS",
        label="latest_positions.csv",
    )
    _validate_csv_header(
        findings,
        stage_id=STAGE_G08,
        project_root=project_root,
        path=latest_sequence_manifest_path,
        expected_header=LATEST_SEQUENCE_MANIFEST_HEADER,
        code="G08_LATEST_SEQUENCE_MANIFEST",
        label="latest_sequence_manifest.csv",
    )
    _validate_csv_header(
        findings,
        stage_id=STAGE_G08,
        project_root=project_root,
        path=raw_returns_path,
        expected_header=VALIDATION_RETURNS_HEADER,
        code="G08_RAW_VALIDATION_RETURNS",
        label="raw_validation_returns.csv",
    )
    _validate_metrics_json(
        findings,
        stage_id=STAGE_G08,
        project_root=project_root,
        path=raw_metrics_path,
        code="G08_RAW_VALIDATION_METRICS",
        label="raw_validation_metrics.json",
    )
    _validate_csv_header(
        findings,
        stage_id=STAGE_G08,
        project_root=project_root,
        path=rescaled_returns_path,
        expected_header=VALIDATION_RETURNS_HEADER,
        code="G08_RESCALED_VALIDATION_RETURNS",
        label="rescaled_validation_returns.csv",
    )
    _validate_metrics_json(
        findings,
        stage_id=STAGE_G08,
        project_root=project_root,
        path=rescaled_metrics_path,
        code="G08_RESCALED_VALIDATION_METRICS",
        label="rescaled_validation_metrics.json",
    )
    if _require_file(
        findings,
        stage_id=STAGE_G08,
        project_root=project_root,
        path=evaluation_report_path,
        code="G08_EVALUATION_REPORT",
        label="evaluation_report.md",
    ):
        report_text = evaluation_report_path.read_text(encoding="utf-8")
        if "FTMO" not in report_text:
            _append_finding(
                findings,
                stage_id=STAGE_G08,
                code="G08_EVALUATION_REPORT_CONTENT",
                message="evaluation_report.md is missing the FTMO-vs.-paper boundary explanation.",
                path=evaluation_report_path,
                project_root=project_root,
            )
        if INTERIM_PATH_TOKEN in report_text:
            _append_finding(
                findings,
                stage_id=STAGE_G08,
                code="G08_EVALUATION_REPORT_INTERIM",
                message="evaluation_report.md still references interim artifacts.",
                path=evaluation_report_path,
                project_root=project_root,
            )
    reproducibility_payload = _validate_json_object(
        findings,
        stage_id=STAGE_G08,
        project_root=project_root,
        path=reproducibility_manifest_path,
        code="G08_REPRODUCIBILITY_MANIFEST",
        label="reproducibility_manifest.json",
        required_keys=REPRODUCIBILITY_REQUIRED_KEYS,
    )
    if reproducibility_payload is not None:
        _validate_no_interim_reference(
            findings,
            stage_id=STAGE_G08,
            project_root=project_root,
            path=reproducibility_manifest_path,
            payload=reproducibility_payload,
            code="G08_REPRODUCIBILITY_MANIFEST_INTERIM",
            label="reproducibility_manifest.json",
        )
    return findings


def _extract_notebook_section_order(path: Path) -> tuple[str, ...]:
    notebook = nbformat.read(path, as_version=4)
    ordered_ids: list[str] = []
    for cell in notebook.cells:
        metadata = cell.get("metadata", {})
        section_metadata = metadata.get("lstm_cpd")
        if not isinstance(section_metadata, dict):
            continue
        section_id = section_metadata.get("section_id")
        if isinstance(section_id, str) and section_id not in ordered_ids:
            ordered_ids.append(section_id)
    return tuple(ordered_ids)


def _check_g09(project_root: Path) -> list[ClosureAuditFinding]:
    findings: list[ClosureAuditFinding] = []
    notebook_path = _resolve_project_path(project_root, DEFAULT_NOTEBOOK_PATH)
    executed_notebook_path = _resolve_project_path(project_root, DEFAULT_EXECUTED_NOTEBOOK_PATH)
    notebook_report_path = _resolve_project_path(project_root, DEFAULT_NOTEBOOK_REPORT_PATH)
    notebook_artifact_map_path = _resolve_project_path(
        project_root,
        DEFAULT_NOTEBOOK_ARTIFACT_MAP_PATH,
    )

    if _require_file(
        findings,
        stage_id=STAGE_G09,
        project_root=project_root,
        path=notebook_path,
        code="G09_NOTEBOOK_MISSING",
        label="lstm_cpd_replication.ipynb",
    ):
        try:
            section_order = _extract_notebook_section_order(notebook_path)
        except Exception as exc:
            _append_finding(
                findings,
                stage_id=STAGE_G09,
                code="G09_NOTEBOOK_INVALID",
                message=f"lstm_cpd_replication.ipynb could not be parsed: {exc}",
                path=notebook_path,
                project_root=project_root,
            )
        else:
            expected_order = notebook_section_id_order()
            if section_order != expected_order:
                _append_finding(
                    findings,
                    stage_id=STAGE_G09,
                    code="G09_NOTEBOOK_SECTION_ORDER",
                    message="lstm_cpd_replication.ipynb does not contain the expected section order.",
                    path=notebook_path,
                    project_root=project_root,
                    details={
                        "expected_order": list(expected_order),
                        "actual_order": list(section_order),
                    },
                )
    _require_file(
        findings,
        stage_id=STAGE_G09,
        project_root=project_root,
        path=executed_notebook_path,
        code="G09_EXECUTED_NOTEBOOK_MISSING",
        label="lstm_cpd_replication.executed.ipynb",
    )
    if _require_file(
        findings,
        stage_id=STAGE_G09,
        project_root=project_root,
        path=notebook_report_path,
        code="G09_NOTEBOOK_REPORT",
        label="notebook_execution_report.md",
    ):
        report_text = notebook_report_path.read_text(encoding="utf-8")
        if "- Status: success" not in report_text:
            _append_finding(
                findings,
                stage_id=STAGE_G09,
                code="G09_NOTEBOOK_REPORT_CONTENT",
                message="notebook_execution_report.md does not confirm a successful execution.",
                path=notebook_report_path,
                project_root=project_root,
            )
        if INTERIM_PATH_TOKEN in report_text:
            _append_finding(
                findings,
                stage_id=STAGE_G09,
                code="G09_NOTEBOOK_REPORT_INTERIM",
                message="notebook_execution_report.md still references interim artifacts.",
                path=notebook_report_path,
                project_root=project_root,
            )
    if _validate_csv_header(
        findings,
        stage_id=STAGE_G09,
        project_root=project_root,
        path=notebook_artifact_map_path,
        expected_header=NOTEBOOK_ARTIFACT_MAP_HEADER,
        code="G09_NOTEBOOK_ARTIFACT_MAP",
        label="notebook_artifact_map.csv",
    ):
        rows = _read_csv_rows(notebook_artifact_map_path)
        actual_order = tuple(dict.fromkeys(row["section_id"] for row in rows))
        expected_order = notebook_section_id_order()
        if actual_order != expected_order:
            _append_finding(
                findings,
                stage_id=STAGE_G09,
                code="G09_NOTEBOOK_ARTIFACT_MAP_ORDER",
                message="notebook_artifact_map.csv does not cover notebook sections in canonical order.",
                path=notebook_artifact_map_path,
                project_root=project_root,
                details={
                    "expected_order": list(expected_order),
                    "actual_order": list(actual_order),
                },
            )
        if any(INTERIM_PATH_TOKEN in row["artifact_ref"] for row in rows):
            _append_finding(
                findings,
                stage_id=STAGE_G09,
                code="G09_NOTEBOOK_ARTIFACT_MAP_INTERIM",
                message="notebook_artifact_map.csv still references interim artifacts.",
                path=notebook_artifact_map_path,
                project_root=project_root,
            )
    return findings


def _stage_statuses(findings: Sequence[ClosureAuditFinding]) -> tuple[ClosureStageStatus, ...]:
    statuses: list[ClosureStageStatus] = []
    for stage_id in _STAGE_ORDER:
        stage_findings = [finding for finding in findings if finding.stage_id == stage_id]
        blocker_count = sum(1 for finding in stage_findings if finding.severity == SEVERITY_BLOCKER)
        warning_count = sum(1 for finding in stage_findings if finding.severity == SEVERITY_WARNING)
        if blocker_count == 0:
            summary = "All official artifacts for this stage are present and validated."
            status = STATUS_READY
        else:
            summary = stage_findings[0].message
            status = STATUS_BLOCKED
        statuses.append(
            ClosureStageStatus(
                stage_id=stage_id,
                stage_name=_STAGE_NAMES[stage_id],
                status=status,
                blocker_count=blocker_count,
                warning_count=warning_count,
                summary=summary,
            )
        )
    return tuple(statuses)


def _first_blocking_stage(stage_statuses: Sequence[ClosureStageStatus]) -> str | None:
    for stage_status in stage_statuses:
        if stage_status.status == STATUS_BLOCKED:
            return stage_status.stage_id
    return None


def _recommended_replay_chain(first_blocking_stage: str | None) -> list[str]:
    if first_blocking_stage is None:
        return [
            "Official artifact chain is complete; artifacts/interim remains archival only.",
            "Close any remaining open Asana tasks and gates against the canonical official paths.",
        ]
    if first_blocking_stage == STAGE_G04:
        return [
            "Wait for T-14 to reach 630/630 completed with 0 running and 0 pending rows in artifacts/reports/cpd_precompute_progress.csv.",
            "Materialize official T-15 outputs on canonical paths: cpd_fit_telemetry.csv, cpd_failure_ledger.csv, cpd_fallback_ledger.csv, and cpd_feature_store_manifest.json.",
            "Replay T-16, T-17, and T-18 on artifacts/datasets/... and rewrite artifacts/manifests/dataset_registry.json.",
            "Replay official smoke, search, selection, inference, evaluation, reproducibility, and notebook outputs on canonical paths.",
        ]
    if first_blocking_stage == STAGE_G05:
        return [
            "Replay T-16, T-17, and T-18 on official paths and rewrite artifacts/manifests/dataset_registry.json without interim references.",
            "Replay the official smoke run under artifacts/training/smoke_run and refresh artifacts/reports/model_fidelity_report.md.",
            "Replay official search, selection, inference, evaluation, reproducibility, and notebook outputs.",
        ]
    if first_blocking_stage == STAGE_G06:
        return [
            "Replay the official smoke run under artifacts/training/smoke_run against the official dataset registry.",
            "Then run the official search, selection, inference, evaluation, reproducibility, and notebook chain.",
        ]
    if first_blocking_stage == STAGE_G07:
        return [
            "Run the full official 50-candidate search against artifacts/manifests/dataset_registry.json.",
            "Select the winning official candidate and rewrite best_candidate.json, best_config.json, and search_summary_report.csv.",
            "Then replay inference, evaluation, reproducibility, and notebook outputs.",
        ]
    if first_blocking_stage == STAGE_G08:
        return [
            "Replay official inference, evaluation, and reproducibility outputs from the official best candidate and official dataset registry.",
            "Then assemble and execute the final official notebook outputs.",
        ]
    return [
        "Assemble notebooks/lstm_cpd_replication.ipynb and execute it top-to-bottom on the official artifact chain.",
        "Write artifacts/notebook/notebook_execution_report.md and notebook_artifact_map.csv, then close G-09.",
    ]


def _render_markdown_report(
    *,
    project_root: Path,
    stage_statuses: Sequence[ClosureStageStatus],
    findings: Sequence[ClosureAuditFinding],
    first_blocking_stage: str | None,
    ready_for_project_closure: bool,
) -> str:
    lines = [
        "# Official Closure Audit",
        "",
        f"- Project root: `{project_root}`",
        f"- Ready for project closure: {'yes' if ready_for_project_closure else 'no'}",
        (
            "- First blocking stage: none"
            if first_blocking_stage is None
            else f"- First blocking stage: `{first_blocking_stage}`"
        ),
        "",
        "## Stage Status",
        "",
    ]
    for stage_status in stage_statuses:
        lines.append(
            "- "
            f"`{stage_status.stage_id}` {stage_status.stage_name}: "
            f"{stage_status.status} "
            f"(blockers={stage_status.blocker_count}, warnings={stage_status.warning_count})"
        )
        lines.append(f"  {stage_status.summary}")
    lines.extend(["", "## Findings", ""])
    if not findings:
        lines.append("- No findings.")
    else:
        for finding in findings:
            path_suffix = "" if finding.path is None else f" [{finding.path}]"
            lines.append(
                "- "
                f"`{finding.stage_id}` `{finding.code}` `{finding.severity}`: "
                f"{finding.message}{path_suffix}"
            )
    lines.extend(["", "## Recommended Replay Chain", ""])
    for step in _recommended_replay_chain(first_blocking_stage):
        lines.append(f"- {step}")
    return "\n".join(lines) + "\n"


def audit_official_closure(
    *,
    project_root: Path | str | None = None,
    json_report_path: Path | str | None = None,
    markdown_report_path: Path | str | None = None,
) -> OfficialClosureAudit:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    findings = tuple(
        _check_g04(project_root_path)
        + _check_g05(project_root_path)
        + _check_g06(project_root_path)
        + _check_g07(project_root_path)
        + _check_g08(project_root_path)
        + _check_g09(project_root_path)
    )
    stage_statuses = _stage_statuses(findings)
    first_blocking_stage = _first_blocking_stage(stage_statuses)
    ready_for_project_closure = all(
        stage_status.status == STATUS_READY for stage_status in stage_statuses
    )
    resolved_json_report_path = _resolve_project_path(
        project_root_path,
        DEFAULT_JSON_REPORT_PATH if json_report_path is None else json_report_path,
    )
    resolved_markdown_report_path = _resolve_project_path(
        project_root_path,
        DEFAULT_MARKDOWN_REPORT_PATH
        if markdown_report_path is None
        else markdown_report_path,
    )
    payload = {
        "generated_at_utc": _now_utc_iso(),
        "project_root": str(project_root_path),
        "ready_for_project_closure": ready_for_project_closure,
        "first_blocking_stage": first_blocking_stage,
        "stages": [asdict(stage_status) for stage_status in stage_statuses],
        "findings": [asdict(finding) for finding in findings],
        "recommended_replay_chain": _recommended_replay_chain(first_blocking_stage),
    }
    _write_json(resolved_json_report_path, payload)
    _write_text(
        resolved_markdown_report_path,
        _render_markdown_report(
            project_root=project_root_path,
            stage_statuses=stage_statuses,
            findings=findings,
            first_blocking_stage=first_blocking_stage,
            ready_for_project_closure=ready_for_project_closure,
        ),
    )
    return OfficialClosureAudit(
        json_report_path=resolved_json_report_path,
        markdown_report_path=resolved_markdown_report_path,
        ready_for_project_closure=ready_for_project_closure,
        first_blocking_stage=first_blocking_stage,
        stages=stage_statuses,
        findings=findings,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the official closure readiness of the LSTM-CPD project."
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--json-report", type=Path, default=None)
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = audit_official_closure(
        project_root=args.project_root,
        json_report_path=args.json_report,
        markdown_report_path=args.markdown_report,
    )
    print(
        "Wrote official closure audit to "
        f"{audit.json_report_path} and {audit.markdown_report_path}"
    )
    if audit.first_blocking_stage is None:
        print("Project is ready for formal closure.")
    else:
        print(f"First blocking stage: {audit.first_blocking_stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
