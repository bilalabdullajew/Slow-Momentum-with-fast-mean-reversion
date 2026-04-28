from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.canonical_daily_close_store import sha256_prefixed  # noqa: E402
from lstm_cpd.reproducibility.manifest import (  # noqa: E402
    build_reproducibility_manifest,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class T27ReproducibilityManifestTests(unittest.TestCase):
    def test_manifest_contains_selected_candidate_hashes_and_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            write_json(
                tmp_path / "artifacts/training/search_schedule.json",
                [
                    {
                        "candidate_id": "C-001",
                        "candidate_index": 0,
                        "lbw": 63,
                    }
                ],
            )
            write_json(
                tmp_path / "artifacts/manifests/ftmo_asset_universe.json",
                [{"asset_id": "TEST", "symbol": "TEST", "category": "equity"}],
            )
            write_json(
                tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json",
                [
                    {
                        "asset_id": "TEST",
                        "canonical_csv_path": "artifacts/canonical_daily_close/TEST.csv",
                    }
                ],
            )
            write_json(
                tmp_path / "artifacts/interim/manifests/dataset_registry.json",
                [{"lbw": 63}],
            )
            write_json(
                tmp_path / "artifacts/training/best_candidate.json",
                {
                    "candidate_id": "C-001",
                    "candidate_index": 0,
                    "dropout": 0.1,
                    "hidden_size": 20,
                    "minibatch_size": 64,
                    "learning_rate": 0.001,
                    "max_grad_norm": 1.0,
                    "lbw": 63,
                    "best_model_path": "artifacts/interim/training/smoke_run/smoke_best_model.keras",
                    "dataset_registry_path": "artifacts/interim/manifests/dataset_registry.json",
                },
            )
            write_json(
                tmp_path / "artifacts/training/best_config.json",
                {
                    "candidate_id": "C-001",
                    "candidate_index": 0,
                    "dropout": 0.1,
                    "hidden_size": 20,
                    "minibatch_size": 64,
                    "learning_rate": 0.001,
                    "max_grad_norm": 1.0,
                    "lbw": 63,
                },
            )
            for rel_path in (
                "artifacts/interim/training/smoke_run/smoke_best_model.keras",
                "artifacts/inference/latest_positions.csv",
                "artifacts/inference/latest_sequence_manifest.csv",
                "artifacts/evaluation/raw_validation_returns.csv",
                "artifacts/evaluation/raw_validation_metrics.json",
                "artifacts/evaluation/rescaled_validation_returns.csv",
                "artifacts/evaluation/rescaled_validation_metrics.json",
                "artifacts/evaluation/evaluation_report.md",
            ):
                path = tmp_path / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact\n", encoding="utf-8")

            artifacts = build_reproducibility_manifest(
                best_candidate_path=tmp_path / "artifacts/training/best_candidate.json",
                best_config_path=tmp_path / "artifacts/training/best_config.json",
                search_schedule_json_path=tmp_path / "artifacts/training/search_schedule.json",
                ftmo_asset_universe_manifest_path=(
                    tmp_path / "artifacts/manifests/ftmo_asset_universe.json"
                ),
                canonical_daily_close_manifest_path=(
                    tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"
                ),
                latest_positions_path=tmp_path / "artifacts/inference/latest_positions.csv",
                latest_sequence_manifest_path=(
                    tmp_path / "artifacts/inference/latest_sequence_manifest.csv"
                ),
                raw_validation_returns_path=(
                    tmp_path / "artifacts/evaluation/raw_validation_returns.csv"
                ),
                raw_validation_metrics_path=(
                    tmp_path / "artifacts/evaluation/raw_validation_metrics.json"
                ),
                rescaled_validation_returns_path=(
                    tmp_path / "artifacts/evaluation/rescaled_validation_returns.csv"
                ),
                rescaled_validation_metrics_path=(
                    tmp_path / "artifacts/evaluation/rescaled_validation_metrics.json"
                ),
                evaluation_report_path=tmp_path / "artifacts/evaluation/evaluation_report.md",
                output_path=(
                    tmp_path / "artifacts/reproducibility/reproducibility_manifest.json"
                ),
                project_root=tmp_path,
                run_id="eval_candidate_c001",
                created_at_utc="2026-04-23T12:00:00Z",
            )

            payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "eval_candidate_c001")
            self.assertEqual(payload["run_type"], "evaluation")
            self.assertEqual(payload["selected_candidate_id"], "C-001")
            self.assertEqual(payload["selected_lbw"], 63)
            self.assertEqual(
                payload["artifact_locations"]["checkpoint"],
                "artifacts/interim/training/smoke_run/smoke_best_model.keras",
            )
            self.assertEqual(
                payload["artifact_locations"]["raw_validation_metrics"],
                "artifacts/evaluation/raw_validation_metrics.json",
            )
            expected_schedule_hash = sha256_prefixed(
                (tmp_path / "artifacts/training/search_schedule.json").read_bytes()
            )
            self.assertEqual(payload["seed_policy"]["sampled_schedule_hash"], expected_schedule_hash)
            self.assertEqual(payload["hashes"]["sampled_schedule_hash"], expected_schedule_hash)
            self.assertTrue(payload["hashes"]["ftmo_asset_universe_hash"].startswith("sha256:"))
            self.assertTrue(
                payload["hashes"]["canonical_daily_close_manifest_hash"].startswith("sha256:")
            )
            self.assertTrue(payload["hashes"]["dataset_registry_hash"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
