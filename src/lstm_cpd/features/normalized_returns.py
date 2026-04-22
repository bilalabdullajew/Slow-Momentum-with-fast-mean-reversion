from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.features.returns import serialize_optional_float
from lstm_cpd.features.volatility import default_output_dir


T09_INPUT_HEADER = ("timestamp", "asset_id", "close", "arithmetic_return", "sigma_t")
T10_OUTPUT_HEADER = (
    "timestamp",
    "asset_id",
    "normalized_return_1",
    "normalized_return_21",
    "normalized_return_63",
    "normalized_return_126",
    "normalized_return_256",
)
NORMALIZED_RETURN_HORIZONS = (1, 21, 63, 126, 256)


@dataclass(frozen=True)
class ReturnsVolatilityRecord:
    timestamp: str
    asset_id: str
    close_text: str
    close_value: float
    sigma_t_text: str
    sigma_t_value: float | None


def default_input_dir() -> Path:
    return default_output_dir()


def load_returns_volatility_csv(
    path: Path | str,
    expected_asset_id: str | None = None,
) -> list[ReturnsVolatilityRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != T09_INPUT_HEADER:
            raise ValueError(f"Returns-volatility header mismatch: {csv_path}")

        records: list[ReturnsVolatilityRecord] = []
        for index, row in enumerate(reader):
            timestamp = row["timestamp"]
            asset_id = row["asset_id"]
            close_text = row["close"]
            sigma_t_text = row["sigma_t"]
            if not timestamp:
                raise ValueError(f"T-09 row {index} missing timestamp: {csv_path}")
            if not asset_id:
                raise ValueError(f"T-09 row {index} missing asset_id: {csv_path}")
            if expected_asset_id is not None and asset_id != expected_asset_id:
                raise ValueError(
                    f"T-09 row {index} asset_id mismatch for {csv_path}: {asset_id}"
                )
            if close_text == "":
                raise ValueError(f"T-09 row {index} missing close: {csv_path}")

            close_value = float(close_text)
            sigma_t_value = None if sigma_t_text == "" else float(sigma_t_text)
            if not math.isfinite(close_value):
                raise ValueError(f"T-09 row {index} has non-finite close: {csv_path}")
            if sigma_t_value is not None and not math.isfinite(sigma_t_value):
                raise ValueError(f"T-09 row {index} has non-finite sigma_t: {csv_path}")

            records.append(
                ReturnsVolatilityRecord(
                    timestamp=timestamp,
                    asset_id=asset_id,
                    close_text=close_text,
                    close_value=close_value,
                    sigma_t_text=sigma_t_text,
                    sigma_t_value=sigma_t_value,
                )
            )
    return records


def compute_interval_return(
    records: Sequence[ReturnsVolatilityRecord],
    index: int,
    horizon: int,
) -> float | None:
    if index < horizon:
        return None
    denominator = records[index - horizon].close_value
    if denominator == 0.0:
        return None
    numerator = records[index].close_value
    return numerator / denominator - 1.0


def compute_normalized_return_features(
    records: Sequence[ReturnsVolatilityRecord],
    horizons: Sequence[int] = NORMALIZED_RETURN_HORIZONS,
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for index, record in enumerate(records):
        row = {
            "timestamp": record.timestamp,
            "asset_id": record.asset_id,
        }
        for horizon in horizons:
            feature_name = f"normalized_return_{horizon}"
            sigma_t = record.sigma_t_value
            if sigma_t is None or sigma_t == 0.0:
                row[feature_name] = ""
                continue
            interval_return = compute_interval_return(records, index, horizon)
            if interval_return is None:
                row[feature_name] = ""
                continue
            normalized_return = interval_return / (sigma_t * math.sqrt(horizon))
            row[feature_name] = serialize_optional_float(normalized_return)
        output_rows.append(row)
    return output_rows


def write_normalized_returns_csv(
    rows: Sequence[dict[str, str]],
    output_path: Path | str,
) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(T10_OUTPUT_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iter_returns_volatility_inputs(input_dir: Path | str) -> list[Path]:
    input_dir_path = Path(input_dir)
    return sorted(path for path in input_dir_path.glob("*_returns_volatility.csv") if path.is_file())


def build_t10_outputs(
    input_dir: Path | str = default_input_dir(),
    output_dir: Path | str = default_output_dir(),
) -> list[Path]:
    input_paths = iter_returns_volatility_inputs(input_dir)
    if not input_paths:
        raise ValueError(f"No returns-volatility inputs found under {Path(input_dir)}")

    output_dir_path = Path(output_dir)
    output_paths: list[Path] = []
    seen_asset_ids: set[str] = set()

    for input_path in input_paths:
        asset_id = input_path.name[: -len("_returns_volatility.csv")]
        if asset_id in seen_asset_ids:
            raise ValueError(f"Duplicate T-09 input asset_id: {asset_id}")
        seen_asset_ids.add(asset_id)

        records = load_returns_volatility_csv(input_path, expected_asset_id=asset_id)
        rows = compute_normalized_return_features(records)
        output_path = output_dir_path / f"{asset_id}_normalized_returns.csv"
        write_normalized_returns_csv(rows, output_path)
        output_paths.append(output_path)

    for asset_id in seen_asset_ids:
        expected_output_path = output_dir_path / f"{asset_id}_normalized_returns.csv"
        if not expected_output_path.exists():
            raise ValueError(f"Missing normalized-returns output for {asset_id}")
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized multi-horizon return feature files."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_paths = build_t10_outputs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote {len(output_paths)} normalized-return files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
