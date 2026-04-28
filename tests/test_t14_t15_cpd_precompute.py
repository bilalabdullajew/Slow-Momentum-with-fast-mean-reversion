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

from lstm_cpd.cpd.precompute import (  # noqa: E402
    CPD_CSV_SUFFIX,
    CPD_OUTPUT_HEADER,
    ReturnsVolatilityRecord,
    build_cpd_feature_rows,
    build_t14_outputs,
    load_cpd_feature_csv,
)
from lstm_cpd.cpd.precompute_contract import (  # noqa: E402
    CPDWindowInput,
    CPDWindowResult,
    STATUS_BASELINE_FAILURE,
    STATUS_CHANGEPOINT_FAILURE,
    STATUS_FALLBACK_PREVIOUS,
    STATUS_INVALID_WINDOW,
    STATUS_RETRY_SUCCESS,
    STATUS_SUCCESS,
)
from lstm_cpd.cpd.telemetry import (  # noqa: E402
    build_t15_outputs,
)


def write_csv(path: Path, header: list[str] | tuple[str, ...], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_manifest_entry(
    *,
    asset_id: str = "TEST",
    canonical_csv_path: str = "artifacts/canonical_daily_close/TEST.csv",
    row_count: int = 4,
    first_timestamp: str = "2024-01-01T00:00:00",
    last_timestamp: str = "2024-01-04T00:00:00",
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "symbol": asset_id,
        "category": "Equity",
        "path_pattern": "category_symbol_d",
        "source_d_file_path": f"data/FTMO Data/Equity/{asset_id}/D/{asset_id}_data.csv",
        "canonical_csv_path": canonical_csv_path,
        "row_count": row_count,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "file_hash": "sha256:test",
    }


def make_cpd_result(
    *,
    status: str,
    window_size: int,
    nu: float | None = None,
    gamma: float | None = None,
    nlml_baseline: float | None = None,
    nlml_changepoint: float | None = None,
    retry_used: bool = False,
    fallback_used: bool = False,
    location_c: float | None = None,
    steepness_s: float | None = None,
    failure_stage: str | None = None,
    failure_message: str | None = None,
    lbw: int = 10,
) -> CPDWindowResult:
    return CPDWindowResult(
        status=status,
        lbw=lbw,
        window_size=window_size,
        nu=nu,
        gamma=gamma,
        nlml_baseline=nlml_baseline,
        nlml_changepoint=nlml_changepoint,
        retry_used=retry_used,
        fallback_used=fallback_used,
        location_c=location_c,
        steepness_s=steepness_s,
        failure_stage=failure_stage,
        failure_message=failure_message,
    )


class T14T15CpdPrecomputeTests(unittest.TestCase):
    def test_build_cpd_feature_rows_uses_only_immediate_previous_outputs(self) -> None:
        returns_rows = [
            ReturnsVolatilityRecord("2024-01-01T00:00:00", "TEST", 0.01),
            ReturnsVolatilityRecord("2024-01-02T00:00:00", "TEST", 0.02),
            ReturnsVolatilityRecord("2024-01-03T00:00:00", "TEST", 0.03),
            ReturnsVolatilityRecord("2024-01-04T00:00:00", "TEST", 0.04),
        ]
        expected_previous = [None, (0.11, 0.21), None, (0.33, 0.44)]
        results = [
            make_cpd_result(
                status=STATUS_SUCCESS,
                window_size=1,
                nu=0.11,
                gamma=0.21,
                nlml_baseline=1.0,
                nlml_changepoint=2.0,
                location_c=5.0,
                steepness_s=1.0,
            ),
            make_cpd_result(
                status=STATUS_INVALID_WINDOW,
                window_size=2,
                failure_stage="window_length",
                failure_message="too short",
            ),
            make_cpd_result(
                status=STATUS_SUCCESS,
                window_size=3,
                nu=0.33,
                gamma=0.44,
                nlml_baseline=3.0,
                nlml_changepoint=4.0,
                location_c=4.0,
                steepness_s=1.5,
            ),
            make_cpd_result(
                status=STATUS_FALLBACK_PREVIOUS,
                window_size=4,
                nu=0.33,
                gamma=0.44,
                nlml_baseline=5.0,
                retry_used=True,
                fallback_used=True,
                failure_stage="changepoint_fit",
                failure_message="fallback",
            ),
        ]

        def fake_fit(window_input: CPDWindowInput) -> CPDWindowResult:
            call_index = len(observed_previous)
            previous_outputs = window_input.previous_outputs
            observed_previous.append(
                None
                if previous_outputs is None
                else (previous_outputs.nu, previous_outputs.gamma)
            )
            return results[call_index]

        observed_previous: list[tuple[float, float] | None] = []
        rows = build_cpd_feature_rows(
            returns_rows,
            lbw=10,
            fit_window_fn=fake_fit,
        )

        self.assertEqual(observed_previous, expected_previous)
        self.assertEqual([row["status"] for row in rows], [
            STATUS_SUCCESS,
            STATUS_INVALID_WINDOW,
            STATUS_SUCCESS,
            STATUS_FALLBACK_PREVIOUS,
        ])
        self.assertEqual(rows[0]["fallback_source_timestamp"], "")
        self.assertEqual(rows[1]["nu"], "")
        self.assertEqual(rows[3]["fallback_source_timestamp"], "2024-01-03T00:00:00")
        self.assertEqual(rows[3]["retry_used"], "true")
        self.assertEqual(rows[3]["fallback_used"], "true")

    def test_build_t14_outputs_rejects_timestamp_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"
            canonical_path = tmp_path / "artifacts/canonical_daily_close/TEST.csv"
            returns_path = tmp_path / "artifacts/features/base/TEST_returns_volatility.csv"

            write_json(
                manifest_path,
                [make_manifest_entry(canonical_csv_path="artifacts/canonical_daily_close/TEST.csv")],
            )
            write_csv(
                canonical_path,
                ["timestamp", "asset_id", "close"],
                [
                    ["2024-01-01T00:00:00", "TEST", "100"],
                    ["2024-01-02T00:00:00", "TEST", "101"],
                    ["2024-01-03T00:00:00", "TEST", "102"],
                    ["2024-01-04T00:00:00", "TEST", "103"],
                ],
            )
            write_csv(
                returns_path,
                ["timestamp", "asset_id", "close", "arithmetic_return", "sigma_t"],
                [
                    ["2024-01-01T00:00:00", "TEST", "100", "", ""],
                    ["2024-01-02T00:00:00", "TEST", "101", "0.01", ""],
                    ["2024-01-03T12:00:00", "TEST", "102", "0.02", ""],
                    ["2024-01-04T00:00:00", "TEST", "103", "0.03", ""],
                ],
            )

            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                build_t14_outputs(
                    canonical_manifest_input=manifest_path,
                    returns_input_dir=tmp_path / "artifacts/features/base",
                    output_dir=tmp_path / "artifacts/features/cpd",
                    project_root=tmp_path,
                    lbws=(10,),
                    fit_window_fn=lambda window_input: make_cpd_result(
                        status=STATUS_SUCCESS,
                        lbw=window_input.lbw,
                        window_size=len(window_input.window_returns),
                        nu=0.1,
                        gamma=0.2,
                        nlml_baseline=1.0,
                        nlml_changepoint=2.0,
                        location_c=5.0,
                        steepness_s=1.0,
                    ),
                )

    def test_build_t14_outputs_writes_full_timeline_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"
            canonical_path = tmp_path / "artifacts/canonical_daily_close/TEST.csv"
            returns_path = tmp_path / "artifacts/features/base/TEST_returns_volatility.csv"
            output_dir = tmp_path / "artifacts/features/cpd"

            write_json(
                manifest_path,
                [make_manifest_entry(canonical_csv_path="artifacts/canonical_daily_close/TEST.csv")],
            )
            write_csv(
                canonical_path,
                ["timestamp", "asset_id", "close"],
                [
                    ["2024-01-01T00:00:00", "TEST", "100"],
                    ["2024-01-02T00:00:00", "TEST", "101"],
                    ["2024-01-03T00:00:00", "TEST", "102"],
                    ["2024-01-04T00:00:00", "TEST", "103"],
                ],
            )
            write_csv(
                returns_path,
                ["timestamp", "asset_id", "close", "arithmetic_return", "sigma_t"],
                [
                    ["2024-01-01T00:00:00", "TEST", "100", "", ""],
                    ["2024-01-02T00:00:00", "TEST", "101", "0.01", ""],
                    ["2024-01-03T00:00:00", "TEST", "102", "0.02", ""],
                    ["2024-01-04T00:00:00", "TEST", "103", "0.03", ""],
                ],
            )

            mapping = {
                (10, "2024-01-01T00:00:00"): make_cpd_result(
                    status=STATUS_INVALID_WINDOW,
                    lbw=10,
                    window_size=1,
                    failure_stage="window_length",
                    failure_message="too short",
                ),
                (10, "2024-01-02T00:00:00"): make_cpd_result(
                    status=STATUS_SUCCESS,
                    lbw=10,
                    window_size=2,
                    nu=0.1,
                    gamma=0.2,
                    nlml_baseline=1.0,
                    nlml_changepoint=2.0,
                    location_c=5.0,
                    steepness_s=1.0,
                ),
                (10, "2024-01-03T00:00:00"): make_cpd_result(
                    status=STATUS_RETRY_SUCCESS,
                    lbw=10,
                    window_size=3,
                    nu=0.3,
                    gamma=0.4,
                    nlml_baseline=3.0,
                    nlml_changepoint=4.0,
                    retry_used=True,
                    location_c=4.0,
                    steepness_s=1.5,
                ),
                (10, "2024-01-04T00:00:00"): make_cpd_result(
                    status=STATUS_FALLBACK_PREVIOUS,
                    lbw=10,
                    window_size=4,
                    nu=0.3,
                    gamma=0.4,
                    nlml_baseline=5.0,
                    retry_used=True,
                    fallback_used=True,
                    failure_stage="changepoint_fit",
                    failure_message="fallback",
                ),
                (21, "2024-01-01T00:00:00"): make_cpd_result(
                    status=STATUS_BASELINE_FAILURE,
                    lbw=21,
                    window_size=1,
                    failure_stage="baseline_fit",
                    failure_message="baseline exploded",
                ),
                (21, "2024-01-02T00:00:00"): make_cpd_result(
                    status=STATUS_CHANGEPOINT_FAILURE,
                    lbw=21,
                    window_size=2,
                    nlml_baseline=6.0,
                    retry_used=True,
                    failure_stage="fallback_unavailable",
                    failure_message="retry exhausted",
                ),
                (21, "2024-01-03T00:00:00"): make_cpd_result(
                    status=STATUS_SUCCESS,
                    lbw=21,
                    window_size=3,
                    nu=0.6,
                    gamma=0.7,
                    nlml_baseline=7.0,
                    nlml_changepoint=8.0,
                    location_c=9.0,
                    steepness_s=2.0,
                ),
                (21, "2024-01-04T00:00:00"): make_cpd_result(
                    status=STATUS_SUCCESS,
                    lbw=21,
                    window_size=4,
                    nu=0.8,
                    gamma=0.9,
                    nlml_baseline=9.0,
                    nlml_changepoint=10.0,
                    location_c=10.0,
                    steepness_s=2.5,
                ),
            }

            def fake_fit(window_input: CPDWindowInput) -> CPDWindowResult:
                return mapping[(window_input.lbw, window_input.window_end_timestamp or "")]

            output_paths = build_t14_outputs(
                canonical_manifest_input=manifest_path,
                returns_input_dir=tmp_path / "artifacts/features/base",
                output_dir=output_dir,
                project_root=tmp_path,
                lbws=(10, 21),
                fit_window_fn=fake_fit,
            )

            self.assertEqual(len(output_paths), 2)
            lbw10_path = output_dir / "lbw_10" / f"TEST{CPD_CSV_SUFFIX}"
            lbw21_path = output_dir / "lbw_21" / f"TEST{CPD_CSV_SUFFIX}"
            self.assertTrue(lbw10_path.exists())
            self.assertTrue(lbw21_path.exists())

            with lbw10_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                raw_rows = list(reader)

            self.assertEqual(tuple(reader.fieldnames or ()), CPD_OUTPUT_HEADER)
            self.assertEqual(len(raw_rows), 4)
            self.assertEqual(raw_rows[0]["timestamp"], "2024-01-01T00:00:00")
            self.assertEqual(raw_rows[0]["status"], STATUS_INVALID_WINDOW)
            self.assertEqual(raw_rows[1]["retry_used"], "false")
            self.assertEqual(raw_rows[2]["retry_used"], "true")
            self.assertEqual(raw_rows[3]["fallback_source_timestamp"], "2024-01-03T00:00:00")

            parsed_rows = load_cpd_feature_csv(
                lbw10_path,
                expected_asset_id="TEST",
                expected_lbw=10,
            )
            self.assertEqual([row.status for row in parsed_rows], [
                STATUS_INVALID_WINDOW,
                STATUS_SUCCESS,
                STATUS_RETRY_SUCCESS,
                STATUS_FALLBACK_PREVIOUS,
            ])
            self.assertEqual(parsed_rows[3].fallback_source_timestamp, "2024-01-03T00:00:00")

    def test_build_t15_outputs_aggregates_reports_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"
            canonical_path = tmp_path / "artifacts/canonical_daily_close/TEST.csv"
            cpd_dir = tmp_path / "artifacts/features/cpd"

            write_json(
                manifest_path,
                [make_manifest_entry(canonical_csv_path="artifacts/canonical_daily_close/TEST.csv")],
            )
            write_csv(
                canonical_path,
                ["timestamp", "asset_id", "close"],
                [
                    ["2024-01-01T00:00:00", "TEST", "100"],
                    ["2024-01-02T00:00:00", "TEST", "101"],
                    ["2024-01-03T00:00:00", "TEST", "102"],
                    ["2024-01-04T00:00:00", "TEST", "103"],
                ],
            )
            write_csv(
                cpd_dir / "lbw_10" / f"TEST{CPD_CSV_SUFFIX}",
                CPD_OUTPUT_HEADER,
                [
                    [
                        "2024-01-01T00:00:00",
                        "TEST",
                        "10",
                        "",
                        "",
                        STATUS_INVALID_WINDOW,
                        "1",
                        "",
                        "",
                        "false",
                        "false",
                        "",
                        "",
                        "",
                        "window_length",
                        "too short",
                    ],
                    [
                        "2024-01-02T00:00:00",
                        "TEST",
                        "10",
                        "0.1",
                        "0.2",
                        STATUS_SUCCESS,
                        "2",
                        "1.0",
                        "2.0",
                        "false",
                        "false",
                        "5.0",
                        "1.0",
                        "",
                        "",
                        "",
                    ],
                    [
                        "2024-01-03T00:00:00",
                        "TEST",
                        "10",
                        "0.3",
                        "0.4",
                        STATUS_RETRY_SUCCESS,
                        "3",
                        "3.0",
                        "4.0",
                        "true",
                        "false",
                        "4.0",
                        "1.5",
                        "",
                        "",
                        "",
                    ],
                    [
                        "2024-01-04T00:00:00",
                        "TEST",
                        "10",
                        "0.3",
                        "0.4",
                        STATUS_FALLBACK_PREVIOUS,
                        "4",
                        "5.0",
                        "",
                        "true",
                        "true",
                        "",
                        "",
                        "2024-01-03T00:00:00",
                        "changepoint_fit",
                        "fallback",
                    ],
                ],
            )
            write_csv(
                cpd_dir / "lbw_21" / f"TEST{CPD_CSV_SUFFIX}",
                CPD_OUTPUT_HEADER,
                [
                    [
                        "2024-01-01T00:00:00",
                        "TEST",
                        "21",
                        "",
                        "",
                        STATUS_BASELINE_FAILURE,
                        "1",
                        "",
                        "",
                        "false",
                        "false",
                        "",
                        "",
                        "",
                        "baseline_fit",
                        "baseline exploded",
                    ],
                    [
                        "2024-01-02T00:00:00",
                        "TEST",
                        "21",
                        "",
                        "",
                        STATUS_CHANGEPOINT_FAILURE,
                        "2",
                        "6.0",
                        "",
                        "true",
                        "false",
                        "",
                        "",
                        "",
                        "fallback_unavailable",
                        "retry exhausted",
                    ],
                    [
                        "2024-01-03T00:00:00",
                        "TEST",
                        "21",
                        "0.6",
                        "0.7",
                        STATUS_SUCCESS,
                        "3",
                        "7.0",
                        "8.0",
                        "false",
                        "false",
                        "9.0",
                        "2.0",
                        "",
                        "",
                        "",
                    ],
                    [
                        "2024-01-04T00:00:00",
                        "TEST",
                        "21",
                        "0.8",
                        "0.9",
                        STATUS_SUCCESS,
                        "4",
                        "9.0",
                        "10.0",
                        "false",
                        "false",
                        "10.0",
                        "2.5",
                        "",
                        "",
                        "",
                    ],
                ],
            )

            outputs = build_t15_outputs(
                input_dir=cpd_dir,
                canonical_manifest_input=manifest_path,
                project_root=tmp_path,
                telemetry_report_path=tmp_path / "artifacts/reports/cpd_fit_telemetry.csv",
                failure_ledger_path=tmp_path / "artifacts/reports/cpd_failure_ledger.csv",
                fallback_ledger_path=tmp_path / "artifacts/reports/cpd_fallback_ledger.csv",
                manifest_output_path=tmp_path / "artifacts/manifests/cpd_feature_store_manifest.json",
                lbws=(10, 21),
            )

            self.assertTrue(outputs.telemetry_report_path.exists())
            self.assertTrue(outputs.failure_ledger_path.exists())
            self.assertTrue(outputs.fallback_ledger_path.exists())
            self.assertTrue(outputs.manifest_output_path.exists())

            with outputs.telemetry_report_path.open("r", encoding="utf-8", newline="") as handle:
                telemetry_rows = list(csv.DictReader(handle))
            self.assertEqual(len(telemetry_rows), 2)
            telemetry_by_lbw = {row["lbw"]: row for row in telemetry_rows}
            self.assertEqual(telemetry_by_lbw["10"]["state"], "present")
            self.assertEqual(telemetry_by_lbw["10"]["missing_reason"], "")
            self.assertEqual(telemetry_by_lbw["10"]["row_count"], "4")
            self.assertEqual(telemetry_by_lbw["10"]["output_row_count"], "3")
            self.assertEqual(telemetry_by_lbw["10"]["fallback_previous_count"], "1")
            self.assertEqual(telemetry_by_lbw["10"]["retry_used_count"], "2")
            self.assertEqual(telemetry_by_lbw["21"]["state"], "present")
            self.assertEqual(telemetry_by_lbw["21"]["baseline_failure_count"], "1")
            self.assertEqual(telemetry_by_lbw["21"]["changepoint_failure_count"], "1")

            with outputs.failure_ledger_path.open("r", encoding="utf-8", newline="") as handle:
                failure_rows = list(csv.DictReader(handle))
            self.assertEqual(len(failure_rows), 3)
            self.assertEqual(
                {(row["lbw"], row["status"]) for row in failure_rows},
                {
                    ("10", STATUS_INVALID_WINDOW),
                    ("21", STATUS_BASELINE_FAILURE),
                    ("21", STATUS_CHANGEPOINT_FAILURE),
                },
            )

            with outputs.fallback_ledger_path.open("r", encoding="utf-8", newline="") as handle:
                fallback_rows = list(csv.DictReader(handle))
            self.assertEqual(len(fallback_rows), 1)
            self.assertEqual(fallback_rows[0]["fallback_source_timestamp"], "2024-01-03T00:00:00")
            self.assertEqual(fallback_rows[0]["lbw"], "10")

            manifest_entries = json.loads(outputs.manifest_output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest_entries), 2)
            self.assertTrue(all(entry["matches_canonical_timeline"] for entry in manifest_entries))
            self.assertTrue(all(entry["state"] == "present" for entry in manifest_entries))
            self.assertTrue(all(str(entry["file_hash"]).startswith("sha256:") for entry in manifest_entries))

    def test_build_t15_outputs_marks_missing_expected_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "artifacts/manifests/canonical_daily_close_manifest.json"
            canonical_path = tmp_path / "artifacts/canonical_daily_close/TEST.csv"
            cpd_dir = tmp_path / "artifacts/features/cpd"

            write_json(
                manifest_path,
                [make_manifest_entry(canonical_csv_path="artifacts/canonical_daily_close/TEST.csv")],
            )
            write_csv(
                canonical_path,
                ["timestamp", "asset_id", "close"],
                [
                    ["2024-01-01T00:00:00", "TEST", "100"],
                    ["2024-01-02T00:00:00", "TEST", "101"],
                    ["2024-01-03T00:00:00", "TEST", "102"],
                    ["2024-01-04T00:00:00", "TEST", "103"],
                ],
            )
            write_csv(
                cpd_dir / "lbw_10" / f"TEST{CPD_CSV_SUFFIX}",
                CPD_OUTPUT_HEADER,
                [
                    [
                        "2024-01-01T00:00:00",
                        "TEST",
                        "10",
                        "",
                        "",
                        STATUS_INVALID_WINDOW,
                        "1",
                        "",
                        "",
                        "false",
                        "false",
                        "",
                        "",
                        "",
                        "window_length",
                        "too short",
                    ],
                    [
                        "2024-01-02T00:00:00",
                        "TEST",
                        "10",
                        "0.1",
                        "0.2",
                        STATUS_SUCCESS,
                        "2",
                        "1.0",
                        "2.0",
                        "false",
                        "false",
                        "5.0",
                        "1.0",
                        "",
                        "",
                        "",
                    ],
                    [
                        "2024-01-03T00:00:00",
                        "TEST",
                        "10",
                        "0.2",
                        "0.3",
                        STATUS_SUCCESS,
                        "3",
                        "1.5",
                        "2.5",
                        "false",
                        "false",
                        "5.5",
                        "1.1",
                        "",
                        "",
                        "",
                    ],
                    [
                        "2024-01-04T00:00:00",
                        "TEST",
                        "10",
                        "0.3",
                        "0.4",
                        STATUS_SUCCESS,
                        "4",
                        "2.0",
                        "3.0",
                        "false",
                        "false",
                        "6.0",
                        "1.2",
                        "",
                        "",
                        "",
                    ],
                ],
            )

            outputs = build_t15_outputs(
                input_dir=cpd_dir,
                canonical_manifest_input=manifest_path,
                project_root=tmp_path,
                telemetry_report_path=tmp_path / "artifacts/reports/cpd_fit_telemetry.csv",
                failure_ledger_path=tmp_path / "artifacts/reports/cpd_failure_ledger.csv",
                fallback_ledger_path=tmp_path / "artifacts/reports/cpd_fallback_ledger.csv",
                manifest_output_path=tmp_path / "artifacts/manifests/cpd_feature_store_manifest.json",
                lbws=(10, 21),
            )

            with outputs.telemetry_report_path.open("r", encoding="utf-8", newline="") as handle:
                telemetry_rows = {row["lbw"]: row for row in csv.DictReader(handle)}
            self.assertEqual(telemetry_rows["10"]["state"], "present")
            self.assertEqual(telemetry_rows["21"]["state"], "missing")
            self.assertEqual(telemetry_rows["21"]["missing_reason"], "missing_final_output")
            self.assertEqual(telemetry_rows["21"]["row_count"], "0")

            manifest_entries = {
                str(entry["lbw"]): entry
                for entry in json.loads(outputs.manifest_output_path.read_text(encoding="utf-8"))
            }
            self.assertEqual(manifest_entries["10"]["state"], "present")
            self.assertEqual(manifest_entries["21"]["state"], "missing")
            self.assertEqual(manifest_entries["21"]["missing_reason"], "missing_final_output")
            self.assertFalse(manifest_entries["21"]["matches_canonical_timeline"])


if __name__ == "__main__":
    unittest.main()
