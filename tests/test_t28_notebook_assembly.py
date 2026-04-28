from __future__ import annotations

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


def _build_fixture_project_root(project_root: Path) -> tuple:
    sections = build_replication_section_catalog(project_root=project_root)
    for artifact_ref in iter_artifact_refs(sections):
        _write_fixture_file(project_root / artifact_ref)
    for module_ref in iter_module_refs(sections):
        _materialize_stub_module(project_root, module_ref)
    (project_root / "LSTM_CPD.ipynb").write_text("", encoding="utf-8")
    return sections


class T28NotebookAssemblyTests(unittest.TestCase):
    def test_section_catalog_matches_asana_order(self) -> None:
        expected_order = (
            "implementation_contract",
            "ftmo_data_contract",
            "canonical_daily_close_layer",
            "base_features",
            "cpd_outputs",
            "dataset_assembly",
            "model_training_setup",
            "search_results",
            "selected_model",
            "causal_inference",
            "validation_evaluation",
            "reproducibility_manifest",
        )
        self.assertEqual(notebook_section_id_order(), expected_order)

    def test_section_catalog_defaults_stay_inside_supplied_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sections = build_replication_section_catalog(project_root=Path(tmpdir))

        for artifact_ref in iter_artifact_refs(sections):
            self.assertFalse(Path(artifact_ref).is_absolute(), artifact_ref)

    def test_build_replication_notebook_writes_valid_notebook_with_section_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sections = _build_fixture_project_root(tmp_path)
            artifacts = build_replication_notebook(
                output_path=tmp_path / DEFAULT_NOTEBOOK_OUTPUT,
                project_root=tmp_path,
            )

            self.assertEqual(
                artifacts.notebook_path,
                tmp_path / "notebooks/lstm_cpd_replication.ipynb",
            )
            notebook = nbformat.read(artifacts.notebook_path, as_version=4)
            nbformat.validate(notebook)
            metadata_ids = notebook.metadata["lstm_cpd"]["section_ids"]
            self.assertEqual(metadata_ids, list(notebook_section_id_order()))

            section_cells = [
                cell
                for cell in notebook.cells
                if "lstm_cpd" in cell.get("metadata", {})
            ]
            self.assertEqual(len(section_cells), len(sections) * 2)
            first_section_metadata = section_cells[0].metadata["lstm_cpd"]
            self.assertEqual(first_section_metadata["section_id"], "implementation_contract")
            self.assertIn(
                "docs/contracts/invariant_ledger.md",
                first_section_metadata["artifact_refs"],
            )

    def test_builder_fails_when_required_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sections = _build_fixture_project_root(tmp_path)
            missing_artifact = tmp_path / sections[0].artifact_refs[0]
            missing_artifact.unlink()

            with self.assertRaises(FileNotFoundError):
                build_replication_notebook(
                    output_path=tmp_path / DEFAULT_NOTEBOOK_OUTPUT,
                    project_root=tmp_path,
                )

    def test_builder_ignores_root_placeholder_notebook_and_uses_canonical_notebooks_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _build_fixture_project_root(tmp_path)

            artifacts = build_replication_notebook(
                output_path=tmp_path / DEFAULT_NOTEBOOK_OUTPUT,
                project_root=tmp_path,
            )

            self.assertEqual(artifacts.notebook_path.name, "lstm_cpd_replication.ipynb")
            self.assertEqual(artifacts.notebook_path.parent.name, "notebooks")
            self.assertTrue((tmp_path / "LSTM_CPD.ipynb").exists())


if __name__ == "__main__":
    unittest.main()
