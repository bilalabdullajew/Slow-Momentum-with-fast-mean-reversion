from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nbformat
from nbclient import NotebookClient

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.datasets.join_and_split import project_relative_path
from lstm_cpd.notebook.assemble import default_notebook_output
from lstm_cpd.notebook.catalog import notebook_section_id_order


DEFAULT_EXECUTED_NOTEBOOK_OUTPUT = "notebooks/lstm_cpd_replication.executed.ipynb"
DEFAULT_NOTEBOOK_EXECUTION_REPORT_OUTPUT = (
    "artifacts/notebook/notebook_execution_report.md"
)
DEFAULT_NOTEBOOK_ARTIFACT_MAP_OUTPUT = "artifacts/notebook/notebook_artifact_map.csv"
NOTEBOOK_ARTIFACT_MAP_HEADER = (
    "section_id",
    "section_title",
    "cell_indices",
    "artifact_ref",
    "module_refs",
)


@dataclass(frozen=True)
class NotebookSectionRecord:
    section_id: str
    section_title: str
    cell_indices: tuple[int, ...]
    artifact_refs: tuple[str, ...]
    module_refs: tuple[str, ...]


@dataclass(frozen=True)
class NotebookExecutionArtifacts:
    input_notebook_path: Path
    executed_notebook_path: Path
    execution_report_path: Path
    notebook_artifact_map_path: Path
    section_records: tuple[NotebookSectionRecord, ...]


def default_executed_notebook_output() -> Path:
    return default_project_root() / DEFAULT_EXECUTED_NOTEBOOK_OUTPUT


def default_notebook_execution_report_output() -> Path:
    return default_project_root() / DEFAULT_NOTEBOOK_EXECUTION_REPORT_OUTPUT


def default_notebook_artifact_map_output() -> Path:
    return default_project_root() / DEFAULT_NOTEBOOK_ARTIFACT_MAP_OUTPUT


def _resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path
    return project_root / candidate_path


def _load_notebook(path: Path | str) -> nbformat.NotebookNode:
    notebook_path = Path(path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook does not exist: {notebook_path}")
    with notebook_path.open("r", encoding="utf-8") as handle:
        return nbformat.read(handle, as_version=4)


def _extract_section_records(notebook: nbformat.NotebookNode) -> tuple[NotebookSectionRecord, ...]:
    section_rows: dict[str, dict[str, object]] = {}
    for cell_index, cell in enumerate(notebook.cells):
        metadata = cell.get("metadata", {})
        section_metadata = metadata.get("lstm_cpd")
        if not isinstance(section_metadata, dict):
            continue
        section_id = section_metadata.get("section_id")
        section_title = section_metadata.get("section_title")
        artifact_refs = tuple(str(item) for item in section_metadata.get("artifact_refs", ()))
        module_refs = tuple(str(item) for item in section_metadata.get("module_refs", ()))
        if not isinstance(section_id, str) or not isinstance(section_title, str):
            raise ValueError(f"Notebook cell {cell_index} has malformed lstm_cpd metadata")
        if section_id not in section_rows:
            section_rows[section_id] = {
                "section_title": section_title,
                "artifact_refs": artifact_refs,
                "module_refs": module_refs,
                "cell_indices": [cell_index],
            }
            continue
        existing = section_rows[section_id]
        if existing["section_title"] != section_title:
            raise ValueError(f"Section title mismatch for {section_id}")
        if existing["artifact_refs"] != artifact_refs:
            raise ValueError(f"Artifact refs mismatch for {section_id}")
        if existing["module_refs"] != module_refs:
            raise ValueError(f"Module refs mismatch for {section_id}")
        existing["cell_indices"].append(cell_index)

    ordered_section_ids = tuple(section_rows.keys())
    expected_ids = notebook_section_id_order()
    if ordered_section_ids != expected_ids:
        raise ValueError(
            "Notebook section order mismatch: "
            f"expected {expected_ids}, got {ordered_section_ids}"
        )
    records: list[NotebookSectionRecord] = []
    for section_id in expected_ids:
        row = section_rows[section_id]
        records.append(
            NotebookSectionRecord(
                section_id=section_id,
                section_title=str(row["section_title"]),
                cell_indices=tuple(int(index) for index in row["cell_indices"]),
                artifact_refs=tuple(str(item) for item in row["artifact_refs"]),
                module_refs=tuple(str(item) for item in row["module_refs"]),
            )
        )
    return tuple(records)


def _write_notebook(path: Path | str, notebook: nbformat.NotebookNode) -> Path:
    notebook_path = Path(path)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return notebook_path


def _write_text(path: Path | str, text: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _write_artifact_map_csv(
    path: Path | str,
    *,
    section_records: Sequence[NotebookSectionRecord],
) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(NOTEBOOK_ARTIFACT_MAP_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in section_records:
            joined_cells = ";".join(str(index) for index in record.cell_indices)
            joined_modules = ";".join(record.module_refs)
            for artifact_ref in record.artifact_refs:
                writer.writerow(
                    {
                        "section_id": record.section_id,
                        "section_title": record.section_title,
                        "cell_indices": joined_cells,
                        "artifact_ref": artifact_ref,
                        "module_refs": joined_modules,
                    }
                )
    return csv_path


def _render_execution_report(
    *,
    input_notebook_path: Path,
    executed_notebook_path: Path,
    notebook_artifact_map_path: Path,
    project_root: Path,
    kernel_name: str,
    section_records: Sequence[NotebookSectionRecord],
) -> str:
    lines = [
        "# Notebook Execution Report",
        "",
        "- Status: success",
        "- Manual intervention required: no",
        f"- Input notebook: {project_relative_path(input_notebook_path, project_root)}",
        f"- Executed notebook: {project_relative_path(executed_notebook_path, project_root)}",
        f"- Artifact map: {project_relative_path(notebook_artifact_map_path, project_root)}",
        f"- Kernel name: {kernel_name}",
        f"- Section count: {len(section_records)}",
        "",
        "## Section Order",
        "",
    ]
    for index, record in enumerate(section_records, start=1):
        lines.append(f"{index}. {record.section_id} — {record.section_title}")
    return "\n".join(lines) + "\n"


def execute_replication_notebook(
    *,
    input_path: Path | str = default_notebook_output(),
    executed_output_path: Path | str = default_executed_notebook_output(),
    execution_report_output: Path | str = default_notebook_execution_report_output(),
    artifact_map_output: Path | str = default_notebook_artifact_map_output(),
    project_root: Path | str | None = None,
    timeout_seconds: int = 600,
    kernel_name: str = "python3",
) -> NotebookExecutionArtifacts:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    resolved_input_path = _resolve_project_path(project_root_path, input_path)
    resolved_executed_output_path = _resolve_project_path(project_root_path, executed_output_path)
    resolved_report_output_path = _resolve_project_path(project_root_path, execution_report_output)
    resolved_artifact_map_output_path = _resolve_project_path(
        project_root_path,
        artifact_map_output,
    )

    notebook = _load_notebook(resolved_input_path)
    section_records = _extract_section_records(notebook)
    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(project_root_path)}},
    )
    executed_notebook = client.execute()

    _write_notebook(resolved_executed_output_path, executed_notebook)
    _write_artifact_map_csv(
        resolved_artifact_map_output_path,
        section_records=section_records,
    )
    _write_text(
        resolved_report_output_path,
        _render_execution_report(
            input_notebook_path=resolved_input_path,
            executed_notebook_path=resolved_executed_output_path,
            notebook_artifact_map_path=resolved_artifact_map_output_path,
            project_root=project_root_path,
            kernel_name=kernel_name,
            section_records=section_records,
        ),
    )
    return NotebookExecutionArtifacts(
        input_notebook_path=resolved_input_path,
        executed_notebook_path=resolved_executed_output_path,
        execution_report_path=resolved_report_output_path,
        notebook_artifact_map_path=resolved_artifact_map_output_path,
        section_records=section_records,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the assembled LSTM-CPD replication notebook."
    )
    parser.add_argument("--input", type=Path, default=default_notebook_output())
    parser.add_argument(
        "--executed-output",
        type=Path,
        default=default_executed_notebook_output(),
    )
    parser.add_argument(
        "--execution-report-output",
        type=Path,
        default=default_notebook_execution_report_output(),
    )
    parser.add_argument(
        "--artifact-map-output",
        type=Path,
        default=default_notebook_artifact_map_output(),
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--kernel-name", type=str, default="python3")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = execute_replication_notebook(
        input_path=args.input,
        executed_output_path=args.executed_output,
        execution_report_output=args.execution_report_output,
        artifact_map_output=args.artifact_map_output,
        project_root=args.project_root,
        timeout_seconds=args.timeout_seconds,
        kernel_name=args.kernel_name,
    )
    print(
        "Executed notebook and wrote artifacts to "
        f"{artifacts.executed_notebook_path}, "
        f"{artifacts.execution_report_path}, and "
        f"{artifacts.notebook_artifact_map_path}"
    )


if __name__ == "__main__":
    main()
