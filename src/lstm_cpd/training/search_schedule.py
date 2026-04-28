from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Sequence

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.training.train_candidate import (
    SEED_BASE,
    CandidateConfig,
    candidate_config_to_payload,
    validate_candidate_config,
)


DEFAULT_SEARCH_SCHEDULE_JSON_PATH = "artifacts/training/search_schedule.json"
DEFAULT_SEARCH_SCHEDULE_CSV_PATH = "artifacts/training/search_schedule.csv"
SEARCH_DROPOUT_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)
SEARCH_HIDDEN_SIZE_VALUES = (5, 10, 20, 40, 80, 160)
SEARCH_MINIBATCH_SIZE_VALUES = (64, 128, 256)
SEARCH_LEARNING_RATE_VALUES = (1e-4, 1e-3, 1e-2, 1e-1)
SEARCH_MAX_GRAD_NORM_VALUES = (1e-2, 1.0, 1e2)
SEARCH_LBW_VALUES = (10, 21, 63, 126, 252)
SCHEDULE_SAMPLE_SIZE = 50
SEARCH_SCHEDULE_HEADER = (
    "candidate_index",
    "candidate_id",
    "dropout",
    "hidden_size",
    "minibatch_size",
    "learning_rate",
    "max_grad_norm",
    "lbw",
)


def default_search_schedule_json_path() -> Path:
    return default_project_root() / DEFAULT_SEARCH_SCHEDULE_JSON_PATH


def default_search_schedule_csv_path() -> Path:
    return default_project_root() / DEFAULT_SEARCH_SCHEDULE_CSV_PATH


def enumerate_full_search_grid() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for dropout in SEARCH_DROPOUT_VALUES:
        for hidden_size in SEARCH_HIDDEN_SIZE_VALUES:
            for minibatch_size in SEARCH_MINIBATCH_SIZE_VALUES:
                for learning_rate in SEARCH_LEARNING_RATE_VALUES:
                    for max_grad_norm in SEARCH_MAX_GRAD_NORM_VALUES:
                        for lbw in SEARCH_LBW_VALUES:
                            rows.append(
                                {
                                    "dropout": dropout,
                                    "hidden_size": hidden_size,
                                    "minibatch_size": minibatch_size,
                                    "learning_rate": learning_rate,
                                    "max_grad_norm": max_grad_norm,
                                    "lbw": lbw,
                                }
                            )
    return tuple(rows)


def build_search_schedule(
    *,
    seed: int = SEED_BASE,
    sample_size: int = SCHEDULE_SAMPLE_SIZE,
) -> tuple[CandidateConfig, ...]:
    full_grid = enumerate_full_search_grid()
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(full_grid):
        raise ValueError(
            f"sample_size={sample_size} exceeds full grid size {len(full_grid)}"
        )
    sampled_rows = random.Random(seed).sample(list(full_grid), sample_size)
    schedule: list[CandidateConfig] = []
    for candidate_index, row in enumerate(sampled_rows):
        candidate_config = CandidateConfig(
            candidate_id=f"C-{candidate_index + 1:03d}",
            candidate_index=candidate_index,
            dropout=float(row["dropout"]),
            hidden_size=int(row["hidden_size"]),
            minibatch_size=int(row["minibatch_size"]),
            learning_rate=float(row["learning_rate"]),
            max_grad_norm=float(row["max_grad_norm"]),
            lbw=int(row["lbw"]),
        )
        validate_candidate_config(candidate_config)
        schedule.append(candidate_config)
    return tuple(schedule)


def write_search_schedule_json(
    path: Path | str,
    schedule: Sequence[CandidateConfig],
) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [candidate_config_to_payload(candidate) for candidate in schedule]
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return json_path


def write_search_schedule_csv(
    path: Path | str,
    schedule: Sequence[CandidateConfig],
) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(SEARCH_SCHEDULE_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for candidate in schedule:
            writer.writerow(candidate_config_to_payload(candidate))
    return csv_path


def load_search_schedule(path: Path | str) -> tuple[CandidateConfig, ...]:
    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Search schedule JSON must be a list of candidate rows")
    schedule: list[CandidateConfig] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for row_position, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError("Search schedule rows must be JSON objects")
        candidate_config = CandidateConfig(
            candidate_id=str(row["candidate_id"]),
            candidate_index=int(row["candidate_index"]),
            dropout=float(row["dropout"]),
            hidden_size=int(row["hidden_size"]),
            minibatch_size=int(row["minibatch_size"]),
            learning_rate=float(row["learning_rate"]),
            max_grad_norm=float(row["max_grad_norm"]),
            lbw=int(row["lbw"]),
        )
        validate_candidate_config(candidate_config)
        if candidate_config.candidate_index != row_position:
            raise ValueError(
                "Search schedule candidate_index must equal the zero-based row position"
            )
        if candidate_config.candidate_index in seen_indices:
            raise ValueError(
                f"Duplicate candidate_index in search schedule: {candidate_config.candidate_index}"
            )
        if candidate_config.candidate_id in seen_ids:
            raise ValueError(
                f"Duplicate candidate_id in search schedule: {candidate_config.candidate_id}"
            )
        seen_indices.add(candidate_config.candidate_index)
        seen_ids.add(candidate_config.candidate_id)
        schedule.append(candidate_config)
    return tuple(schedule)


def materialize_search_schedule(
    *,
    schedule_json_path: Path | str = default_search_schedule_json_path(),
    schedule_csv_path: Path | str = default_search_schedule_csv_path(),
    seed: int = SEED_BASE,
    sample_size: int = SCHEDULE_SAMPLE_SIZE,
) -> tuple[CandidateConfig, ...]:
    schedule = build_search_schedule(seed=seed, sample_size=sample_size)
    write_search_schedule_json(schedule_json_path, schedule)
    write_search_schedule_csv(schedule_csv_path, schedule)
    return schedule


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the immutable 50-candidate search schedule."
    )
    parser.add_argument(
        "--schedule-json",
        type=Path,
        default=default_search_schedule_json_path(),
    )
    parser.add_argument(
        "--schedule-csv",
        type=Path,
        default=default_search_schedule_csv_path(),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    schedule = materialize_search_schedule(
        schedule_json_path=args.schedule_json,
        schedule_csv_path=args.schedule_csv,
    )
    print(
        "Wrote immutable search schedule with "
        f"{len(schedule)} candidates to {args.schedule_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
