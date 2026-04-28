from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lstm_cpd.canonical_daily_close_store import sha256_prefixed
from lstm_cpd.cpd.precompute_contract import ALLOWED_CPD_LBWS, CPD_RESULT_STATUSES, is_allowed_lbw
from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.datasets.join_and_split import (
    CPD_OUTPUT_HEADER,
    CPDFeatureStoreManifestRecord,
    build_t16_outputs,
    load_cpd_feature_csv,
    project_relative_path,
)
from lstm_cpd.datasets.registry import build_t18_outputs
from lstm_cpd.datasets.sequences import build_t17_outputs
from lstm_cpd.features.returns import load_canonical_daily_close_csv
from lstm_cpd.features.volatility import (
    CanonicalDailyCloseManifestRecord,
    load_canonical_daily_close_manifest,
)


PROGRESS_HEADER = (
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
)
PROGRESS_STATE_COMPLETED = "completed"


@dataclass(frozen=True)
class CPDPrecomputeProgressRow:
    lbw: int
    asset_id: str
    state: str
    rows_written: int
    last_timestamp: str | None
    retry_count: int
    fallback_count: int
    started_at: str | None
    finished_at: str | None
    output_path: str
    error_message: str | None


@dataclass(frozen=True)
class InterimMaterializationArtifacts:
    coverage_summary_path: Path
    dataset_registry_path: Path
    cpd_manifest_paths: list[Path]
    materialized_lbws: tuple[int, ...]


def default_progress_input() -> Path:
    return default_project_root() / "artifacts/reports/cpd_precompute_progress.csv"


def default_canonical_manifest_input() -> Path:
    return default_project_root() / "artifacts/manifests/canonical_daily_close_manifest.json"


def default_base_input_dir() -> Path:
    return default_project_root() / "artifacts/features/base"


def default_returns_input_dir() -> Path:
    return default_project_root() / "artifacts/features/base"


def default_cpd_input_dir() -> Path:
    return default_project_root() / "artifacts/features/cpd"


def default_output_root() -> Path:
    return default_project_root() / "artifacts/interim"


def default_summary_output() -> Path:
    return default_output_root() / "manifests/interim_materialization_summary.json"


def default_dataset_registry_output() -> Path:
    return default_output_root() / "manifests/dataset_registry.json"


def default_dataset_output_dir() -> Path:
    return default_output_root() / "datasets"


def _validate_requested_lbws(lbws: Sequence[int]) -> tuple[int, ...]:
    unique_lbws = tuple(dict.fromkeys(int(lbw) for lbw in lbws))
    if not unique_lbws:
        raise ValueError("At least one lbw is required")
    invalid_lbws = [lbw for lbw in unique_lbws if not is_allowed_lbw(lbw)]
    if invalid_lbws:
        raise ValueError(f"Unsupported lbws requested: {invalid_lbws}")
    return unique_lbws


def load_cpd_precompute_progress(
    path: Path | str,
) -> list[CPDPrecomputeProgressRow]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PROGRESS_HEADER:
            raise ValueError(f"CPD progress header mismatch: {csv_path}")

        rows: list[CPDPrecomputeProgressRow] = []
        seen_pairs: set[tuple[int, str]] = set()
        for row_index, row in enumerate(reader):
            lbw = int(row["lbw"])
            asset_id = row["asset_id"]
            pair = (lbw, asset_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate asset/lbw pair in CPD progress at row {row_index}: {pair}"
                )
            seen_pairs.add(pair)
            rows.append(
                CPDPrecomputeProgressRow(
                    lbw=lbw,
                    asset_id=asset_id,
                    state=row["state"],
                    rows_written=int(row["rows_written"]),
                    last_timestamp=row["last_timestamp"] or None,
                    retry_count=int(row["retry_count"]),
                    fallback_count=int(row["fallback_count"]),
                    started_at=row["started_at"] or None,
                    finished_at=row["finished_at"] or None,
                    output_path=row["output_path"],
                    error_message=row["error_message"] or None,
                )
            )
    return rows


def _status_counts_from_cpd_rows(rows) -> dict[str, int]:
    counts = {status: 0 for status in CPD_RESULT_STATUSES}
    for row in rows:
        counts[row.status] += 1
    return counts


def _load_cpd_raw_rows(path: Path | str) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CPD_OUTPUT_HEADER:
            raise ValueError(f"CPD feature header mismatch: {csv_path}")
        return [dict(row) for row in reader]


def _validate_canonical_timeline(
    canonical_record: CanonicalDailyCloseManifestRecord,
    *,
    canonical_rows,
    cpd_rows,
) -> None:
    if len(canonical_rows) != canonical_record.row_count:
        raise ValueError(
            f"Canonical row count mismatch for {canonical_record.asset_id}: "
            f"{len(canonical_rows)} != {canonical_record.row_count}"
        )
    if len(cpd_rows) != canonical_record.row_count:
        raise ValueError(
            f"CPD row count mismatch for {canonical_record.asset_id}: "
            f"{len(cpd_rows)} != {canonical_record.row_count}"
        )
    if not cpd_rows:
        raise ValueError(f"Completed CPD output is empty for {canonical_record.asset_id}")
    if cpd_rows[0].timestamp != canonical_record.first_timestamp:
        raise ValueError(
            f"CPD first timestamp mismatch for {canonical_record.asset_id}: "
            f"{cpd_rows[0].timestamp} != {canonical_record.first_timestamp}"
        )
    if cpd_rows[-1].timestamp != canonical_record.last_timestamp:
        raise ValueError(
            f"CPD last timestamp mismatch for {canonical_record.asset_id}: "
            f"{cpd_rows[-1].timestamp} != {canonical_record.last_timestamp}"
        )
    for row_index, (canonical_row, cpd_row) in enumerate(zip(canonical_rows, cpd_rows)):
        if canonical_row.timestamp != cpd_row.timestamp:
            raise ValueError(
                f"Canonical/CPD timestamp mismatch for {canonical_record.asset_id} "
                f"at row {row_index}: {canonical_row.timestamp} != {cpd_row.timestamp}"
            )


def _validate_completed_output_path(
    progress_row: CPDPrecomputeProgressRow,
) -> str:
    expected = f"artifacts/features/cpd/lbw_{progress_row.lbw}/{progress_row.asset_id}_cpd.csv"
    if progress_row.output_path != expected:
        raise ValueError(
            "Completed CPD progress row has unexpected output_path for "
            f"{(progress_row.lbw, progress_row.asset_id)}: {progress_row.output_path}"
        )
    if progress_row.output_path.endswith(".partial.csv"):
        raise ValueError(
            f"Completed CPD progress row must not point to partial output: {progress_row.output_path}"
        )
    return expected


def _write_json(path: Path | str, payload: object) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _count_unique_assets(path: Path | str) -> int:
    csv_path = Path(path)
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return len({row["asset_id"] for row in reader})


def _load_dataset_registry(path: Path | str) -> list[dict[str, object]]:
    json_path = Path(path)
    return json.loads(json_path.read_text(encoding="utf-8"))


def build_interim_cpd_manifest(
    *,
    canonical_manifest_records: Sequence[CanonicalDailyCloseManifestRecord],
    progress_rows: Sequence[CPDPrecomputeProgressRow],
    project_root: Path | str,
    cpd_input_dir: Path | str,
    manifest_output_path: Path | str,
    lbw: int,
    asset_ids: Sequence[str],
) -> Path:
    project_root_path = Path(project_root)
    cpd_input_dir_path = Path(cpd_input_dir)
    progress_index = {(row.lbw, row.asset_id): row for row in progress_rows}
    canonical_index = {record.asset_id: record for record in canonical_manifest_records}
    canonical_rows_cache = {}

    manifest_entries: list[dict[str, object]] = []
    for asset_id in asset_ids:
        canonical_record = canonical_index.get(asset_id)
        if canonical_record is None:
            raise ValueError(f"Canonical manifest is missing asset_id required for interim run: {asset_id}")
        progress_row = progress_index.get((lbw, asset_id))
        if progress_row is None or progress_row.state != PROGRESS_STATE_COMPLETED:
            raise ValueError(
                f"Interim manifest requested non-completed asset/lbw pair: {(lbw, asset_id)}"
            )

        relative_path = _validate_completed_output_path(progress_row)
        cpd_path = project_root_path / relative_path
        if not cpd_path.exists():
            raise ValueError(f"Completed CPD output does not exist: {cpd_path}")
        if cpd_path.suffix != ".csv" or cpd_path.name.endswith(".partial.csv"):
            raise ValueError(f"Completed CPD output must be a final CSV: {cpd_path}")
        if cpd_path.parent.resolve() != (cpd_input_dir_path / f"lbw_{lbw}").resolve():
            raise ValueError(
                f"Completed CPD output is outside the expected input dir for lbw={lbw}: {cpd_path}"
            )

        if asset_id not in canonical_rows_cache:
            canonical_rows_cache[asset_id] = load_canonical_daily_close_csv(
                project_root_path / canonical_record.canonical_csv_path,
                expected_asset_id=asset_id,
            )
        canonical_rows = canonical_rows_cache[asset_id]
        cpd_rows = load_cpd_feature_csv(
            cpd_path,
            expected_asset_id=asset_id,
            expected_lbw=lbw,
        )
        cpd_raw_rows = _load_cpd_raw_rows(cpd_path)
        if len(cpd_raw_rows) != len(cpd_rows):
            raise ValueError(f"Raw/parsed CPD row-count mismatch: {cpd_path}")
        _validate_canonical_timeline(
            canonical_record,
            canonical_rows=canonical_rows,
            cpd_rows=cpd_rows,
        )

        status_counts = _status_counts_from_cpd_rows(cpd_rows)
        manifest_entries.append(
            {
                "asset_id": asset_id,
                "lbw": lbw,
                "state": "present",
                "missing_reason": None,
                "cpd_csv_path": project_relative_path(cpd_path, project_root_path),
                "row_count": len(cpd_rows),
                "canonical_row_count": canonical_record.row_count,
                "first_timestamp": cpd_rows[0].timestamp,
                "last_timestamp": cpd_rows[-1].timestamp,
                "output_row_count": sum(1 for row in cpd_rows if row.has_outputs),
                "retry_used_count": sum(
                    1 for row in cpd_raw_rows if row["retry_used"] == "true"
                ),
                "fallback_used_count": sum(
                    1 for row in cpd_raw_rows if row["fallback_used"] == "true"
                ),
                "status_counts": status_counts,
                "matches_canonical_timeline": True,
                "file_hash": sha256_prefixed(cpd_path.read_bytes()),
            }
        )

    if not manifest_entries:
        raise ValueError(f"No completed CPD coverage available for interim lbw={lbw}")

    _write_json(manifest_output_path, manifest_entries)
    loaded_records = load_interim_manifest(manifest_output_path)
    for record in loaded_records:
        if record.state != "present" or not record.matches_canonical_timeline:
            raise ValueError(f"Interim CPD manifest contains invalid record: {record}")
    return Path(manifest_output_path)


def load_interim_manifest(path: Path | str) -> list[CPDFeatureStoreManifestRecord]:
    from lstm_cpd.datasets.join_and_split import load_cpd_feature_store_manifest

    return load_cpd_feature_store_manifest(path)


def materialize_interim_datasets(
    *,
    progress_input: Path | str = default_progress_input(),
    canonical_manifest_input: Path | str = default_canonical_manifest_input(),
    base_input_dir: Path | str = default_base_input_dir(),
    returns_input_dir: Path | str = default_returns_input_dir(),
    cpd_input_dir: Path | str = default_cpd_input_dir(),
    output_root: Path | str = default_output_root(),
    project_root: Path | str | None = None,
    coverage_summary_output: Path | str = default_summary_output(),
    dataset_registry_output: Path | str = default_dataset_registry_output(),
    lbws: Sequence[int] = ALLOWED_CPD_LBWS,
) -> InterimMaterializationArtifacts:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    requested_lbws = _validate_requested_lbws(lbws)
    output_root_path = Path(output_root)
    datasets_output_dir = output_root_path / "datasets"
    manifests_output_dir = output_root_path / "manifests"

    progress_rows = load_cpd_precompute_progress(progress_input)
    canonical_manifest_records = load_canonical_daily_close_manifest(
        canonical_manifest_input
    )
    canonical_asset_order = [record.asset_id for record in canonical_manifest_records]

    progress_state_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    completed_pairs: set[tuple[int, str]] = set()
    for row in progress_rows:
        progress_state_counts[row.lbw][row.state] += 1
        if row.state == PROGRESS_STATE_COMPLETED:
            completed_pairs.add((row.lbw, row.asset_id))

    selected_assets_by_lbw: dict[int, tuple[str, ...]] = {}
    for lbw in requested_lbws:
        selected_assets_by_lbw[lbw] = tuple(
            asset_id
            for asset_id in canonical_asset_order
            if (lbw, asset_id) in completed_pairs
        )

    cpd_manifest_paths: list[Path] = []
    materialized_lbws: list[int] = []
    for lbw in requested_lbws:
        asset_ids = selected_assets_by_lbw[lbw]
        if not asset_ids:
            continue
        manifest_path = manifests_output_dir / f"cpd_feature_store_manifest_lbw_{lbw}.json"
        build_interim_cpd_manifest(
            canonical_manifest_records=canonical_manifest_records,
            progress_rows=progress_rows,
            project_root=project_root_path,
            cpd_input_dir=cpd_input_dir,
            manifest_output_path=manifest_path,
            lbw=lbw,
            asset_ids=asset_ids,
        )
        build_t16_outputs(
            base_input_dir=base_input_dir,
            returns_input_dir=returns_input_dir,
            cpd_manifest_input=manifest_path,
            output_dir=datasets_output_dir,
            project_root=project_root_path,
            lbws=(lbw,),
            asset_ids=asset_ids,
        )
        cpd_manifest_paths.append(manifest_path)
        materialized_lbws.append(lbw)

    if not materialized_lbws:
        raise ValueError(
            "No completed CPD coverage available for requested lbws; interim materialization cannot proceed."
        )

    build_t17_outputs(
        input_dir=datasets_output_dir,
        output_dir=datasets_output_dir,
        project_root=project_root_path,
        lbws=tuple(materialized_lbws),
    )
    build_t18_outputs(
        input_dir=datasets_output_dir,
        output_dir=datasets_output_dir,
        dataset_registry_output=dataset_registry_output,
        project_root=project_root_path,
        lbws=tuple(materialized_lbws),
    )

    dataset_registry_entries = _load_dataset_registry(dataset_registry_output)
    dataset_registry_by_lbw = {int(entry["lbw"]): entry for entry in dataset_registry_entries}

    coverage_summary = {
        "mode": "interim_partial_cpd_coverage",
        "notes": [
            "Uses only CPD progress rows with state=completed.",
            "Excludes running, pending, failed, and partial CPD outputs.",
            "Does not close G-04 or G-05 and does not replace official T-15/T-18 artifacts.",
        ],
        "progress_input": project_relative_path(progress_input, project_root_path),
        "dataset_output_dir": project_relative_path(datasets_output_dir, project_root_path),
        "dataset_registry_path": project_relative_path(dataset_registry_output, project_root_path),
        "requested_lbws": list(requested_lbws),
        "materialized_lbws": list(materialized_lbws),
        "coverage_by_lbw": [],
    }
    for lbw in requested_lbws:
        registry_entry = dataset_registry_by_lbw.get(lbw)
        val_index_path = (
            project_root_path / registry_entry["artifacts"]["val_sequence_index_path"]
            if registry_entry is not None
            else None
        )
        coverage_summary["coverage_by_lbw"].append(
            {
                "lbw": lbw,
                "progress_state_counts": dict(progress_state_counts.get(lbw, {})),
                "completed_asset_count": len(selected_assets_by_lbw[lbw]),
                "selected_asset_ids": list(selected_assets_by_lbw[lbw]),
                "interim_cpd_manifest_path": (
                    project_relative_path(
                        manifests_output_dir / f"cpd_feature_store_manifest_lbw_{lbw}.json",
                        project_root_path,
                    )
                    if lbw in materialized_lbws
                    else None
                ),
                "train_sequence_count": (
                    int(registry_entry["train_sequence_count"]) if registry_entry else 0
                ),
                "validation_sequence_count": (
                    int(registry_entry["val_sequence_count"]) if registry_entry else 0
                ),
                "retained_asset_count": (
                    _count_unique_assets(val_index_path) if val_index_path is not None else 0
                ),
            }
        )
    _write_json(coverage_summary_output, coverage_summary)

    return InterimMaterializationArtifacts(
        coverage_summary_path=Path(coverage_summary_output),
        dataset_registry_path=Path(dataset_registry_output),
        cpd_manifest_paths=cpd_manifest_paths,
        materialized_lbws=tuple(materialized_lbws),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize interim T-16 through T-18 dataset artifacts from currently completed "
            "CPD outputs only."
        )
    )
    parser.add_argument("--progress-input", type=Path, default=default_progress_input())
    parser.add_argument(
        "--canonical-manifest-input",
        type=Path,
        default=default_canonical_manifest_input(),
    )
    parser.add_argument("--base-input-dir", type=Path, default=default_base_input_dir())
    parser.add_argument(
        "--returns-input-dir",
        type=Path,
        default=default_returns_input_dir(),
    )
    parser.add_argument("--cpd-input-dir", type=Path, default=default_cpd_input_dir())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument(
        "--coverage-summary-output",
        type=Path,
        default=default_summary_output(),
    )
    parser.add_argument(
        "--dataset-registry-output",
        type=Path,
        default=default_dataset_registry_output(),
    )
    parser.add_argument(
        "--lbw",
        type=int,
        action="append",
        dest="lbws",
        default=None,
        help="Restrict interim materialization to one or more allowed LBWs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = materialize_interim_datasets(
        progress_input=args.progress_input,
        canonical_manifest_input=args.canonical_manifest_input,
        base_input_dir=args.base_input_dir,
        returns_input_dir=args.returns_input_dir,
        cpd_input_dir=args.cpd_input_dir,
        output_root=args.output_root,
        project_root=args.project_root,
        coverage_summary_output=args.coverage_summary_output,
        dataset_registry_output=args.dataset_registry_output,
        lbws=ALLOWED_CPD_LBWS if args.lbws is None else tuple(args.lbws),
    )
    print(
        "Wrote interim dataset artifacts for lbws="
        f"{list(artifacts.materialized_lbws)} with dataset registry at "
        f"{artifacts.dataset_registry_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
