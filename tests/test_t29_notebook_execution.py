from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.notebook.assemble import (  # noqa: E402
    DEFAULT_NOTEBOOK_OUTPUT,
    build_replication_notebook,
)
from lstm_cpd.notebook.catalog import (  # noqa: E402
    build_replication_section_catalog,
    iter_artifact_refs,
    iter_module_refs,
    notebook_section_id_order,
)
from lstm_cpd.notebook.execute import (  # noqa: E402
    execute_replication_notebook,
)


def _write_fixture_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps({"path": path.as_posix()}, indent=2) + "\n", encoding="utf-8")
        return
    if suffix == ".csv":
        path.write_text("column_a,column_b\nvalue_a,value_b\n", encoding="utf-8")
        return
    if suffix == ".md":
        path.write_text(f"# {path.name}\nfixture\n", encoding="utf-8")
        return
    path.write_text("fixture\n", encoding="utf-8")


def _materialize_stub_module(project_root: Path, module_name: str) -> None:
    parts = module_name.split(".")
    module_path = project_root / "src"
    for package_name in parts[:-1]:
        module_path = module_path / package_name
        module_path.mkdir(parents=True, exist_ok=True)
        init_path = module_path / "__init__.py"
        if not init_path.exists():
            init_path.write_text('"""stub package."""\n', encoding="utf-8")
    module_file = module_path / f"{parts[-1]}.py"
    module_file.write_text(
        f'MODULE_NAME = "{module_name}"\n',
        encoding="utf-8",
    )


def _build_fixture_project_root(project_root: Path) -> None:
    sections = build_replication_section_catalog(project_root=project_root)
    for artifact_ref in iter_artifact_refs(sections):
        _write_fixture_file(project_root / artifact_ref)
    for module_ref in iter_module_refs(sections):
        _materialize_stub_module(project_root, module_ref)
    build_replication_notebook(
        output_path=project_root / DEFAULT_NOTEBOOK_OUTPUT,
        project_root=project_root,
    )


class T29NotebookExecutionTests(unittest.TestCase):
    def test_execute_replication_notebook_writes_executed_copy_report_and_artifact_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _build_fixture_project_root(tmp_path)

            artifacts = execute_replication_notebook(
                input_path=tmp_path / "notebooks/lstm_cpd_replication.ipynb",
                executed_output_path=tmp_path / "notebooks/lstm_cpd_replication.executed.ipynb",
                execution_report_output=(
                    tmp_path / "artifacts/notebook/notebook_execution_report.md"
                ),
                artifact_map_output=(
                    tmp_path / "artifacts/notebook/notebook_artifact_map.csv"
                ),
                project_root=tmp_path,
                timeout_seconds=120,
            )

            self.assertTrue(artifacts.executed_notebook_path.exists())
            self.assertTrue(artifacts.execution_report_path.exists())
            self.assertTrue(artifacts.notebook_artifact_map_path.exists())

            executed_notebook = nbformat.read(artifacts.executed_notebook_path, as_version=4)
            executed_code_cells = [
                cell
                for cell in executed_notebook.cells
                if cell.cell_type == "code" and cell.get("outputs")
            ]
            self.assertGreater(len(executed_code_cells), 0)

            report_text = artifacts.execution_report_path.read_text(encoding="utf-8")
            self.assertIn("- Status: success", report_text)
            self.assertIn("- Manual intervention required: no", report_text)
            self.assertIn("1. implementation_contract", report_text)
            self.assertIn("12. reproducibility_manifest", report_text)

            with artifacts.notebook_artifact_map_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0)
            self.assertEqual(
                tuple(dict.fromkeys(row["section_id"] for row in rows)),
                notebook_section_id_order(),
            )

    def test_execute_replication_notebook_fails_cleanly_on_cell_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _build_fixture_project_root(tmp_path)
            notebook_path = tmp_path / "notebooks/lstm_cpd_replication.ipynb"
            notebook = nbformat.read(notebook_path, as_version=4)
            notebook.cells.append(nbformat.v4.new_code_cell("raise RuntimeError('boom')"))
            with notebook_path.open("w", encoding="utf-8") as handle:
                nbformat.write(notebook, handle)

            report_path = tmp_path / "artifacts/notebook/notebook_execution_report.md"
            artifact_map_path = tmp_path / "artifacts/notebook/notebook_artifact_map.csv"
            executed_output_path = tmp_path / "notebooks/lstm_cpd_replication.executed.ipynb"

            with self.assertRaises(Exception):
                execute_replication_notebook(
                    input_path=notebook_path,
                    executed_output_path=executed_output_path,
                    execution_report_output=report_path,
                    artifact_map_output=artifact_map_path,
                    project_root=tmp_path,
                    timeout_seconds=120,
                )

            self.assertFalse(report_path.exists())
            self.assertFalse(artifact_map_path.exists())
            self.assertFalse(executed_output_path.exists())


if __name__ == "__main__":
    unittest.main()
