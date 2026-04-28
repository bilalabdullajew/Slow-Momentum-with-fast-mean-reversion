from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.training.train_candidate import CandidateConfig, load_candidate_config


DEFAULT_BEST_CANDIDATE_PATH = "artifacts/training/best_candidate.json"
DEFAULT_BEST_CONFIG_PATH = "artifacts/training/best_config.json"
DEFAULT_DATASET_REGISTRY_PATH = "artifacts/manifests/dataset_registry.json"

_CANDIDATE_CONFIG_FIELDS = (
    "candidate_id",
    "candidate_index",
    "dropout",
    "hidden_size",
    "minibatch_size",
    "learning_rate",
    "max_grad_norm",
    "lbw",
)


@dataclass(frozen=True)
class ResolvedModelSource:
    candidate_config: CandidateConfig
    model_path: Path
    dataset_registry_path: Path
    best_candidate_path: Path | None
    candidate_config_path: Path | None

    @property
    def candidate_id(self) -> str:
        return self.candidate_config.candidate_id

    @property
    def lbw(self) -> int:
        return self.candidate_config.lbw


def default_best_candidate_path() -> Path:
    return default_project_root() / DEFAULT_BEST_CANDIDATE_PATH


def default_best_config_path() -> Path:
    return default_project_root() / DEFAULT_BEST_CONFIG_PATH


def default_dataset_registry_path() -> Path:
    return default_project_root() / DEFAULT_DATASET_REGISTRY_PATH


def _resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _candidate_config_from_payload(
    payload: dict[str, object],
    *,
    source_label: str,
) -> CandidateConfig:
    missing_fields = [field_name for field_name in _CANDIDATE_CONFIG_FIELDS if field_name not in payload]
    if missing_fields:
        raise ValueError(
            f"{source_label} is missing candidate-config fields: {missing_fields}"
        )
    return CandidateConfig(
        candidate_id=str(payload["candidate_id"]),
        candidate_index=int(payload["candidate_index"]),
        dropout=float(payload["dropout"]),
        hidden_size=int(payload["hidden_size"]),
        minibatch_size=int(payload["minibatch_size"]),
        learning_rate=float(payload["learning_rate"]),
        max_grad_norm=float(payload["max_grad_norm"]),
        lbw=int(payload["lbw"]),
    )


def _candidate_configs_match(
    left: CandidateConfig,
    right: CandidateConfig,
) -> bool:
    return left == right


def _path_from_payload(
    payload: dict[str, object],
    *,
    top_level_key: str,
    nested_group: str | None = None,
    nested_key: str | None = None,
) -> str | None:
    value = payload.get(top_level_key)
    if isinstance(value, str) and value != "":
        return value
    if nested_group is None or nested_key is None:
        return None
    nested_value = payload.get(nested_group)
    if not isinstance(nested_value, dict):
        return None
    candidate_value = nested_value.get(nested_key)
    if isinstance(candidate_value, str) and candidate_value != "":
        return candidate_value
    return None


def resolve_selected_model_source(
    *,
    best_candidate_path: Path | str = default_best_candidate_path(),
    best_config_path: Path | str = default_best_config_path(),
    model_path: Path | str | None = None,
    candidate_config_path: Path | str | None = None,
    dataset_registry_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> ResolvedModelSource:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    resolved_best_candidate_path = _resolve_project_path(project_root_path, best_candidate_path)
    resolved_best_config_path = _resolve_project_path(project_root_path, best_config_path)

    best_candidate_payload: dict[str, object] | None = None
    if resolved_best_candidate_path.exists():
        best_candidate_payload = _load_json_object(
            resolved_best_candidate_path,
            label="best_candidate.json",
        )

    candidate_payload: dict[str, object] | None = None
    resolved_candidate_config_path: Path | None = None
    if candidate_config_path is not None:
        resolved_candidate_config_path = _resolve_project_path(
            project_root_path,
            candidate_config_path,
        )
        candidate_payload = _load_json_object(
            resolved_candidate_config_path,
            label="candidate config",
        )
        candidate_config = load_candidate_config(resolved_candidate_config_path)
    elif resolved_best_config_path.exists():
        resolved_candidate_config_path = resolved_best_config_path
        candidate_payload = _load_json_object(
            resolved_candidate_config_path,
            label="best_config.json",
        )
        candidate_config = load_candidate_config(resolved_candidate_config_path)
    elif best_candidate_payload is not None:
        candidate_config = _candidate_config_from_payload(
            best_candidate_payload,
            source_label="best_candidate.json",
        )
    else:
        raise FileNotFoundError(
            "Unable to resolve candidate config. Provide candidate_config_path or "
            "materialize artifacts/training/best_config.json."
        )

    if best_candidate_payload is not None and candidate_config_path is None:
        best_candidate_config = _candidate_config_from_payload(
            best_candidate_payload,
            source_label="best_candidate.json",
        )
        if not _candidate_configs_match(best_candidate_config, candidate_config):
            raise ValueError(
                "best_candidate.json candidate fields do not match the resolved candidate "
                "config source"
            )

    if model_path is not None:
        resolved_model_path = _resolve_project_path(project_root_path, model_path)
    else:
        if best_candidate_payload is None:
            raise FileNotFoundError(
                "Unable to resolve model_path. Provide model_path explicitly or "
                "materialize artifacts/training/best_candidate.json."
            )
        raw_model_path = _path_from_payload(
            best_candidate_payload,
            top_level_key="best_model_path",
            nested_group="artifacts",
            nested_key="best_model_path",
        )
        if raw_model_path is None:
            raise ValueError("best_candidate.json is missing best_model_path")
        resolved_model_path = _resolve_project_path(project_root_path, raw_model_path)

    if dataset_registry_path is not None:
        resolved_dataset_registry_path = _resolve_project_path(
            project_root_path,
            dataset_registry_path,
        )
    else:
        raw_dataset_registry_path: str | None = None
        if best_candidate_payload is not None:
            raw_dataset_registry_path = _path_from_payload(
                best_candidate_payload,
                top_level_key="dataset_registry_path",
            )
        if raw_dataset_registry_path is None and candidate_payload is not None:
            raw_dataset_registry_path = _path_from_payload(
                candidate_payload,
                top_level_key="dataset_registry_path",
            )
        if raw_dataset_registry_path is None:
            resolved_dataset_registry_path = default_dataset_registry_path()
        else:
            resolved_dataset_registry_path = _resolve_project_path(
                project_root_path,
                raw_dataset_registry_path,
            )

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Resolved model checkpoint does not exist: {resolved_model_path}")
    if not resolved_dataset_registry_path.exists():
        raise FileNotFoundError(
            "Resolved dataset registry does not exist: "
            f"{resolved_dataset_registry_path}"
        )

    return ResolvedModelSource(
        candidate_config=candidate_config,
        model_path=resolved_model_path,
        dataset_registry_path=resolved_dataset_registry_path,
        best_candidate_path=(
            resolved_best_candidate_path if best_candidate_payload is not None else None
        ),
        candidate_config_path=resolved_candidate_config_path,
    )
