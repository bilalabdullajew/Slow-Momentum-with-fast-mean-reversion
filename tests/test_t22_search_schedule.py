from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.training.search_schedule import (  # noqa: E402
    SEARCH_SCHEDULE_HEADER,
    build_search_schedule,
    enumerate_full_search_grid,
    load_search_schedule,
    materialize_search_schedule,
)


class T22SearchScheduleTests(unittest.TestCase):
    def test_schedule_has_fifty_unique_candidates(self) -> None:
        schedule = build_search_schedule()

        self.assertEqual(len(schedule), 50)
        self.assertEqual(
            [candidate.candidate_index for candidate in schedule],
            list(range(50)),
        )
        self.assertEqual(
            [candidate.candidate_id for candidate in schedule],
            [f"C-{index + 1:03d}" for index in range(50)],
        )
        unique_hyperparameter_rows = {
            (
                candidate.dropout,
                candidate.hidden_size,
                candidate.minibatch_size,
                candidate.learning_rate,
                candidate.max_grad_norm,
                candidate.lbw,
            )
            for candidate in schedule
        }
        self.assertEqual(len(unique_hyperparameter_rows), 50)
        self.assertEqual(len(enumerate_full_search_grid()), 5400)

    def test_schedule_materialization_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            first_json = tmp_path / "first/search_schedule.json"
            first_csv = tmp_path / "first/search_schedule.csv"
            second_json = tmp_path / "second/search_schedule.json"
            second_csv = tmp_path / "second/search_schedule.csv"

            materialize_search_schedule(
                schedule_json_path=first_json,
                schedule_csv_path=first_csv,
            )
            materialize_search_schedule(
                schedule_json_path=second_json,
                schedule_csv_path=second_csv,
            )

            self.assertEqual(
                first_json.read_text(encoding="utf-8"),
                second_json.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                first_csv.read_text(encoding="utf-8"),
                second_csv.read_text(encoding="utf-8"),
            )

    def test_schedule_contains_expected_deterministic_rows_and_csv_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            json_path = tmp_path / "search_schedule.json"
            csv_path = tmp_path / "search_schedule.csv"
            schedule = materialize_search_schedule(
                schedule_json_path=json_path,
                schedule_csv_path=csv_path,
            )
            reloaded = load_search_schedule(json_path)

            self.assertEqual(schedule, reloaded)
            self.assertEqual(schedule[0].candidate_id, "C-001")
            self.assertEqual(schedule[0].dropout, 0.3)
            self.assertEqual(schedule[0].hidden_size, 5)
            self.assertEqual(schedule[0].minibatch_size, 256)
            self.assertEqual(schedule[0].learning_rate, 1e-3)
            self.assertEqual(schedule[0].max_grad_norm, 1.0)
            self.assertEqual(schedule[0].lbw, 252)

            self.assertEqual(schedule[1].candidate_id, "C-002")
            self.assertEqual(schedule[1].dropout, 0.5)
            self.assertEqual(schedule[1].hidden_size, 40)
            self.assertEqual(schedule[1].minibatch_size, 64)
            self.assertEqual(schedule[1].learning_rate, 1e-1)
            self.assertEqual(schedule[1].max_grad_norm, 1.0)
            self.assertEqual(schedule[1].lbw, 63)

            self.assertEqual(schedule[49].candidate_id, "C-050")
            self.assertEqual(schedule[49].dropout, 0.3)
            self.assertEqual(schedule[49].hidden_size, 5)
            self.assertEqual(schedule[49].minibatch_size, 128)
            self.assertEqual(schedule[49].learning_rate, 1e-2)
            self.assertEqual(schedule[49].max_grad_norm, 1e2)
            self.assertEqual(schedule[49].lbw, 10)

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            self.assertEqual(tuple(reader.fieldnames or ()), SEARCH_SCHEDULE_HEADER)
            self.assertEqual(len(rows), 50)
            self.assertEqual(rows[0]["candidate_index"], "0")
            self.assertEqual(rows[0]["candidate_id"], "C-001")
            self.assertEqual(rows[0]["lbw"], "252")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[47]["candidate_id"], "C-048")
            self.assertEqual(payload[48]["candidate_id"], "C-049")
            self.assertEqual(payload[49]["candidate_id"], "C-050")


if __name__ == "__main__":
    unittest.main()
