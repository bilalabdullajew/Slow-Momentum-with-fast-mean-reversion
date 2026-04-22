from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.cpd.precompute import (  # noqa: E402
    build_t14_outputs,
    build_t14_outputs_parallel,
    build_t14_task_specs,
    run_t14_chain_task,
    T14ChainStopRequested,
)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture_project(
    root: Path,
    *,
    assets: dict[str, list[float]],
) -> tuple[Path, Path]:
    manifest_path = root / "artifacts/manifests/canonical_daily_close_manifest.json"
    returns_dir = root / "artifacts/features/base"
    manifest_rows: list[dict[str, object]] = []

    for asset_id, closes in assets.items():
        timestamps = [f"2024-01-{day:02d}T00:00:00" for day in range(1, len(closes) + 1)]
        canonical_path = root / "artifacts/canonical_daily_close" / f"{asset_id}.csv"
        write_csv(
            canonical_path,
            ["timestamp", "asset_id", "close"],
            [
                [timestamp, asset_id, format(close, ".17g")]
                for timestamp, close in zip(timestamps, closes)
            ],
        )

        returns_path = returns_dir / f"{asset_id}_returns_volatility.csv"
        returns_rows: list[list[str]] = []
        for index, (timestamp, close) in enumerate(zip(timestamps, closes)):
            arithmetic_return = ""
            if index > 0:
                arithmetic_return = format(
                    (close - closes[index - 1]) / closes[index - 1],
                    ".17g",
                )
            returns_rows.append(
                [
                    timestamp,
                    asset_id,
                    format(close, ".17g"),
                    arithmetic_return,
                    "",
                ]
            )
        write_csv(
            returns_path,
            ["timestamp", "asset_id", "close", "arithmetic_return", "sigma_t"],
            returns_rows,
        )

        manifest_rows.append(
            {
                "asset_id": asset_id,
                "symbol": asset_id,
                "category": "Equity",
                "path_pattern": "category_symbol_d",
                "source_d_file_path": f"data/FTMO Data/Equity/{asset_id}/D/{asset_id}_data.csv",
                "canonical_csv_path": f"artifacts/canonical_daily_close/{asset_id}.csv",
                "row_count": len(closes),
                "first_timestamp": timestamps[0],
                "last_timestamp": timestamps[-1],
                "file_hash": f"sha256:{asset_id.lower()}",
            }
        )

    write_json(manifest_path, manifest_rows)
    return manifest_path, returns_dir


def collect_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hash_file(path)
        for path in sorted(root.rglob("*_cpd.csv"))
    }


class T14ParallelCheckpointTests(unittest.TestCase):
    def make_assets(self) -> dict[str, list[float]]:
        return {
            "ALPHA": [
                100.0,
                101.2,
                100.7,
                102.1,
                101.5,
                103.0,
                102.4,
                104.2,
                103.7,
                105.1,
                104.6,
                106.0,
            ],
            "BETA": [
                50.0,
                49.8,
                50.4,
                50.9,
                50.6,
                51.3,
                51.0,
                51.8,
                52.2,
                52.0,
                52.9,
                53.4,
            ],
        }

    def test_parallel_outputs_match_serial_hashes_on_small_real_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path, returns_dir = build_fixture_project(
                tmp_path,
                assets=self.make_assets(),
            )
            serial_dir = tmp_path / "serial/features/cpd"
            parallel_dir = tmp_path / "parallel/features/cpd"

            build_t14_outputs(
                canonical_manifest_input=manifest_path,
                returns_input_dir=returns_dir,
                output_dir=serial_dir,
                project_root=tmp_path,
                lbws=(10,),
            )
            build_t14_outputs_parallel(
                canonical_manifest_input=manifest_path,
                returns_input_dir=returns_dir,
                output_dir=parallel_dir,
                project_root=tmp_path,
                lbws=(10,),
                max_workers=2,
                resume=True,
            )

            self.assertEqual(collect_hashes(serial_dir), collect_hashes(parallel_dir))

    def test_resume_checkpoint_matches_fresh_reference_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path, returns_dir = build_fixture_project(
                tmp_path,
                assets={"ALPHA": self.make_assets()["ALPHA"]},
            )
            resume_dir = tmp_path / "resume/features/cpd"
            reference_dir = tmp_path / "reference/features/cpd"

            task = build_t14_task_specs(
                canonical_manifest_input=manifest_path,
                returns_input_dir=returns_dir,
                output_dir=resume_dir,
                project_root=tmp_path,
                lbws=(10,),
            )[0]

            with self.assertRaisesRegex(T14ChainStopRequested, "Stopped after 11 rows"):
                run_t14_chain_task(
                    task,
                    resume=False,
                    stop_after_rows=11,
                )

            self.assertTrue(task.partial_output_path.exists())
            self.assertTrue(task.checkpoint_path.exists())
            self.assertFalse(task.output_path.exists())

            resumed_output = run_t14_chain_task(
                task,
                resume=True,
                skip_if_complete=True,
            )
            reference_output = build_t14_outputs(
                canonical_manifest_input=manifest_path,
                returns_input_dir=returns_dir,
                output_dir=reference_dir,
                project_root=tmp_path,
                lbws=(10,),
            )[0]

            self.assertFalse(task.partial_output_path.exists())
            self.assertFalse(task.checkpoint_path.exists())
            self.assertEqual(hash_file(resumed_output), hash_file(reference_output))

    def test_parallel_worker_count_invariance_on_small_real_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path, returns_dir = build_fixture_project(
                tmp_path,
                assets=self.make_assets(),
            )
            one_worker_dir = tmp_path / "one_worker/features/cpd"
            two_worker_dir = tmp_path / "two_worker/features/cpd"

            build_t14_outputs_parallel(
                canonical_manifest_input=manifest_path,
                returns_input_dir=returns_dir,
                output_dir=one_worker_dir,
                project_root=tmp_path,
                lbws=(10,),
                max_workers=1,
                resume=True,
            )
            build_t14_outputs_parallel(
                canonical_manifest_input=manifest_path,
                returns_input_dir=returns_dir,
                output_dir=two_worker_dir,
                project_root=tmp_path,
                lbws=(10,),
                max_workers=2,
                resume=True,
            )

            self.assertEqual(collect_hashes(one_worker_dir), collect_hashes(two_worker_dir))


if __name__ == "__main__":
    unittest.main()
