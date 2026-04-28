from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Sequence

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.model_source import (
    DEFAULT_BEST_CANDIDATE_PATH,
    DEFAULT_BEST_CONFIG_PATH,
    DEFAULT_DATASET_REGISTRY_PATH,
    default_best_candidate_path,
    default_best_config_path,
    default_dataset_registry_path,
)
from lstm_cpd.notebook.catalog import (
    DEFAULT_EVALUATION_DIR,
    DEFAULT_INFERENCE_DIR,
    NotebookSectionSpec,
    build_replication_section_catalog,
)
from lstm_cpd.reproducibility.manifest import (
    DEFAULT_REPRODUCIBILITY_MANIFEST_OUTPUT,
    default_reproducibility_manifest_output,
)
from lstm_cpd.training.selection import (
    DEFAULT_SEARCH_SUMMARY_REPORT_PATH,
    default_search_summary_report_path,
)


DEFAULT_NOTEBOOK_OUTPUT = "notebooks/lstm_cpd_replication.ipynb"
NOTEBOOK_METADATA_NAMESPACE = "lstm_cpd"


@dataclass(frozen=True)
class NotebookAssemblyArtifacts:
    notebook_path: Path
    project_root: Path
    section_catalog: tuple[NotebookSectionSpec, ...]


def default_notebook_output() -> Path:
    return default_project_root() / DEFAULT_NOTEBOOK_OUTPUT


def _resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _require_existing_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _cell_metadata(section: NotebookSectionSpec) -> dict[str, object]:
    return {NOTEBOOK_METADATA_NAMESPACE: section.metadata_payload()}


def _title_cell() -> nbformat.NotebookNode:
    return new_markdown_cell(
        dedent(
            """
            # LSTM CPD Replication

            This notebook is the final presentation and reproducibility wrapper around
            frozen implementation artifacts. It loads persisted outputs, imports existing
            source modules, and intentionally keeps all core methodology outside notebook cells.
            """
        ).strip()
    )


def _context_cell(section_count: int) -> nbformat.NotebookNode:
    return new_markdown_cell(
        dedent(
            f"""
            ## Audience, Prerequisites, and Goal

            This notebook is for readers who need a top-to-bottom view of the LSTM-CPD
            replication pipeline after implementation artifacts have been frozen.

            Prerequisites:
            - the project root contains the frozen artifacts referenced below
            - the `src/lstm_cpd` package is available from the project root
            - this notebook is a wrapper only; no notebook-only core logic is allowed

            Goal:
            - surface the full pipeline in {section_count} ordered sections
            - keep every result traceable to an artifact path or source module
            """
        ).strip()
    )


def _outline_cell(sections: Sequence[NotebookSectionSpec]) -> nbformat.NotebookNode:
    outline_lines = ["## Outline", ""]
    for index, section in enumerate(sections, start=1):
        outline_lines.append(f"{index}. {section.title}")
    return new_markdown_cell("\n".join(outline_lines))


def _helper_cell() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            from __future__ import annotations

            import csv
            import importlib
            import json
            import sys
            from itertools import islice
            from pathlib import Path


            def _find_project_root(start: Path | None = None) -> Path:
                candidate = (start or Path.cwd()).resolve()
                for root in (candidate, *candidate.parents):
                    if (root / "src" / "lstm_cpd").exists() and (
                        root / "spec_lstm_cpd_model_revised_sole_authority.md"
                    ).exists():
                        return root
                raise FileNotFoundError(
                    "Unable to locate the project root from the notebook execution context."
                )


            PROJECT_ROOT = _find_project_root()
            SRC_ROOT = PROJECT_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))


            def _resolve_project_path(reference: str) -> Path:
                candidate = Path(reference)
                if candidate.is_absolute():
                    return candidate
                return PROJECT_ROOT / candidate


            def _preview_text(path: Path, *, max_lines: int = 8) -> str:
                with path.open("r", encoding="utf-8") as handle:
                    lines = [line.rstrip("\\n") for line in islice(handle, max_lines)]
                return "\\n".join(lines)


            def _preview_csv(path: Path, *, max_lines: int = 6) -> str:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    return "".join(list(islice(handle, max_lines))).rstrip()


            def _preview_json(path: Path) -> str:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    preview = {key: payload[key] for key in list(payload)[:5]}
                elif isinstance(payload, list):
                    preview = payload[:2]
                else:
                    preview = payload
                return json.dumps(preview, indent=2)


            def preview_artifact(reference: str) -> str:
                path = _resolve_project_path(reference)
                suffix = path.suffix.lower()
                if suffix == ".json":
                    return _preview_json(path)
                if suffix == ".csv":
                    return _preview_csv(path)
                if suffix in {".md", ".txt", ".py"}:
                    return _preview_text(path)
                size_bytes = path.stat().st_size
                return f"{path.name} ({size_bytes} bytes)"


            def import_module_refs(module_refs: list[str]) -> list[tuple[str, str]]:
                resolved = []
                for module_name in module_refs:
                    module = importlib.import_module(module_name)
                    module_file = getattr(module, "__file__", None)
                    module_path = "<built-in>" if module_file is None else str(Path(module_file).resolve())
                    resolved.append((module_name, module_path))
                return resolved


            def render_section(
                section_id: str,
                section_title: str,
                artifact_refs: list[str],
                module_refs: list[str],
            ) -> dict[str, object]:
                print(f"SECTION {section_id}: {section_title}")
                print(f"Project root: {PROJECT_ROOT}")
                print("")
                print("Artifacts")
                for artifact_ref in artifact_refs:
                    print(f"- {artifact_ref}")
                    print(preview_artifact(artifact_ref))
                    print("")
                print("Modules")
                for module_name, module_path in import_module_refs(module_refs):
                    print(f"- {module_name}: {module_path}")
                return {
                    "section_id": section_id,
                    "section_title": section_title,
                    "artifact_refs": artifact_refs,
                    "module_refs": module_refs,
                }
            """
        ).strip(),
        metadata={"tags": ["setup"]},
    )


def _section_markdown_cell(
    section: NotebookSectionSpec,
    *,
    section_index: int,
) -> nbformat.NotebookNode:
    return new_markdown_cell(
        dedent(
            f"""
            ## {section_index:02d}. {section.title}

            {section.narrative}
            """
        ).strip(),
        metadata=_cell_metadata(section),
    )


def _section_code_cell(section: NotebookSectionSpec) -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            f"""
            SECTION_ID = {section.section_id!r}
            SECTION_TITLE = {section.title!r}
            ARTIFACT_REFS = {list(section.artifact_refs)!r}
            MODULE_REFS = {list(section.module_refs)!r}

            render_section(SECTION_ID, SECTION_TITLE, ARTIFACT_REFS, MODULE_REFS)
            """
        ).strip(),
        metadata=_cell_metadata(section),
    )


def _notebook_metadata(section_catalog: Sequence[NotebookSectionSpec]) -> dict[str, object]:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": platform.python_version(),
        },
        NOTEBOOK_METADATA_NAMESPACE: {
            "generated_by": "lstm_cpd.notebook.assemble",
            "section_ids": [section.section_id for section in section_catalog],
        },
    }


def build_replication_notebook(
    *,
    best_candidate_path: Path | str = DEFAULT_BEST_CANDIDATE_PATH,
    best_config_path: Path | str = DEFAULT_BEST_CONFIG_PATH,
    dataset_registry_path: Path | str = DEFAULT_DATASET_REGISTRY_PATH,
    search_summary_report_path: Path | str = DEFAULT_SEARCH_SUMMARY_REPORT_PATH,
    reproducibility_manifest_path: Path | str = DEFAULT_REPRODUCIBILITY_MANIFEST_OUTPUT,
    inference_dir: Path | str = DEFAULT_INFERENCE_DIR,
    evaluation_dir: Path | str = DEFAULT_EVALUATION_DIR,
    output_path: Path | str = default_notebook_output(),
    project_root: Path | str | None = None,
) -> NotebookAssemblyArtifacts:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    section_catalog = build_replication_section_catalog(
        best_candidate_path=best_candidate_path,
        best_config_path=best_config_path,
        dataset_registry_path=dataset_registry_path,
        search_summary_report_path=search_summary_report_path,
        reproducibility_manifest_path=reproducibility_manifest_path,
        inference_dir=inference_dir,
        evaluation_dir=evaluation_dir,
        project_root=project_root_path,
    )
    for section in section_catalog:
        for artifact_ref in section.artifact_refs:
            _require_existing_file(
                _resolve_project_path(project_root_path, artifact_ref),
                label=f"{section.section_id} artifact",
            )

    cells: list[nbformat.NotebookNode] = [
        _title_cell(),
        _context_cell(len(section_catalog)),
        _outline_cell(section_catalog),
        _helper_cell(),
    ]
    for index, section in enumerate(section_catalog, start=1):
        cells.append(_section_markdown_cell(section, section_index=index))
        cells.append(_section_code_cell(section))

    notebook = new_notebook(
        cells=cells,
        metadata=_notebook_metadata(section_catalog),
    )
    nbformat.validate(notebook)

    resolved_output_path = _resolve_project_path(project_root_path, output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    return NotebookAssemblyArtifacts(
        notebook_path=resolved_output_path,
        project_root=project_root_path,
        section_catalog=tuple(section_catalog),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the LSTM-CPD replication notebook from frozen artifacts."
    )
    parser.add_argument("--best-candidate", type=Path, default=default_best_candidate_path())
    parser.add_argument("--best-config", type=Path, default=default_best_config_path())
    parser.add_argument(
        "--dataset-registry-path",
        type=Path,
        default=default_dataset_registry_path(),
    )
    parser.add_argument(
        "--search-summary-report",
        type=Path,
        default=default_search_summary_report_path(),
    )
    parser.add_argument(
        "--reproducibility-manifest",
        type=Path,
        default=default_reproducibility_manifest_output(),
    )
    parser.add_argument("--inference-dir", type=Path, default=Path(DEFAULT_INFERENCE_DIR))
    parser.add_argument("--evaluation-dir", type=Path, default=Path(DEFAULT_EVALUATION_DIR))
    parser.add_argument("--output", type=Path, default=default_notebook_output())
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = build_replication_notebook(
        best_candidate_path=args.best_candidate,
        best_config_path=args.best_config,
        dataset_registry_path=args.dataset_registry_path,
        search_summary_report_path=args.search_summary_report,
        reproducibility_manifest_path=args.reproducibility_manifest,
        inference_dir=args.inference_dir,
        evaluation_dir=args.evaluation_dir,
        output_path=args.output,
        project_root=args.project_root,
    )
    print(f"Wrote replication notebook to {artifacts.notebook_path}")


if __name__ == "__main__":
    main()
