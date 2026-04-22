from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import pandas as pd

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.features.returns import (
    CanonicalDailyCloseRecord,
    load_canonical_daily_close_csv,
    serialize_optional_float,
)
from lstm_cpd.features.volatility import (
    default_canonical_manifest_input as default_t09_manifest_input,
    default_output_dir as default_feature_output_dir,
    load_canonical_daily_close_manifest,
    validate_canonical_alignment,
)


MACD_PAIRS = ((8, 24), (16, 28), (32, 96))
EMA_ADJUST = False
PRICE_STD_WINDOW = 63
Q_STD_WINDOW = 252
ROLLING_STD_DDOF = 1
T11_OUTPUT_HEADER = (
    "timestamp",
    "asset_id",
    "macd_8_24",
    "macd_16_28",
    "macd_32_96",
)


def default_canonical_manifest_input() -> Path:
    return default_t09_manifest_input()


def default_output_dir() -> Path:
    return default_feature_output_dir()


def compute_exponentially_weighted_moving_average(
    values: Sequence[float],
    *,
    period: int,
    adjust: bool = EMA_ADJUST,
) -> list[float]:
    if period <= 0:
        raise ValueError(f"EMA period must be positive: {period}")
    if not values:
        return []

    series = pd.Series(list(values), dtype="float64")
    ema_series = series.ewm(alpha=1.0 / period, adjust=adjust).mean()
    return [float(value) for value in ema_series.tolist()]


def compute_trailing_standard_deviation(
    values: Sequence[float | None],
    *,
    window: int,
    min_periods: int,
    ddof: int = ROLLING_STD_DDOF,
) -> list[float | None]:
    if window <= 0:
        raise ValueError(f"Rolling window must be positive: {window}")
    if min_periods <= 0:
        raise ValueError(f"min_periods must be positive: {min_periods}")
    if not values:
        return []

    series = pd.Series(list(values), dtype="float64")
    std_series = series.rolling(window=window, min_periods=min_periods).std(ddof=ddof)
    return [None if pd.isna(value) else float(value) for value in std_series.tolist()]


def compute_normalized_macd_feature(
    canonical_rows: Sequence[CanonicalDailyCloseRecord],
    *,
    short_period: int,
    long_period: int,
    price_std_window: int = PRICE_STD_WINDOW,
    q_std_window: int = Q_STD_WINDOW,
    ema_adjust: bool = EMA_ADJUST,
) -> list[float | None]:
    if not canonical_rows:
        return []

    close_values = [row.close_value for row in canonical_rows]
    ema_short = compute_exponentially_weighted_moving_average(
        close_values,
        period=short_period,
        adjust=ema_adjust,
    )
    ema_long = compute_exponentially_weighted_moving_average(
        close_values,
        period=long_period,
        adjust=ema_adjust,
    )
    macd_values = [short_value - long_value for short_value, long_value in zip(ema_short, ema_long)]
    price_std_values = compute_trailing_standard_deviation(
        close_values,
        window=price_std_window,
        min_periods=price_std_window,
    )

    q_values: list[float | None] = []
    for macd_value, price_std_value in zip(macd_values, price_std_values):
        if price_std_value is None or price_std_value == 0.0:
            q_values.append(None)
            continue
        q_values.append(macd_value / price_std_value)

    q_std_values = compute_trailing_standard_deviation(
        q_values,
        window=q_std_window,
        min_periods=q_std_window,
    )

    y_values: list[float | None] = []
    for q_value, q_std_value in zip(q_values, q_std_values):
        if q_value is None or q_std_value is None or q_std_value == 0.0:
            y_values.append(None)
            continue
        y_values.append(q_value / q_std_value)
    return y_values


def build_macd_feature_rows(
    canonical_rows: Sequence[CanonicalDailyCloseRecord],
    macd_pairs: Sequence[tuple[int, int]] = MACD_PAIRS,
) -> list[dict[str, str]]:
    feature_columns: dict[str, list[str]] = {}
    for short_period, long_period in macd_pairs:
        column_name = f"macd_{short_period}_{long_period}"
        values = compute_normalized_macd_feature(
            canonical_rows,
            short_period=short_period,
            long_period=long_period,
        )
        feature_columns[column_name] = [
            serialize_optional_float(value) for value in values
        ]

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(canonical_rows):
        output_row = {
            "timestamp": row.timestamp,
            "asset_id": row.asset_id,
        }
        for column_name in T11_OUTPUT_HEADER[2:]:
            output_row[column_name] = feature_columns[column_name][index]
        output_rows.append(output_row)
    return output_rows


def write_macd_features_csv(
    rows: Sequence[dict[str, str]],
    output_path: Path | str,
) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(T11_OUTPUT_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_t11_outputs(
    canonical_manifest_input: Path | str = default_canonical_manifest_input(),
    output_dir: Path | str = default_output_dir(),
    project_root: Path | str | None = None,
) -> list[Path]:
    project_root_path = (
        Path(project_root) if project_root is not None else default_project_root()
    )
    output_dir_path = Path(output_dir)
    manifest_records = load_canonical_daily_close_manifest(canonical_manifest_input)

    output_paths: list[Path] = []
    for manifest_record in manifest_records:
        canonical_csv_path = project_root_path / manifest_record.canonical_csv_path
        canonical_rows = load_canonical_daily_close_csv(
            canonical_csv_path,
            expected_asset_id=manifest_record.asset_id,
        )
        validate_canonical_alignment(manifest_record, canonical_rows)
        rows = build_macd_feature_rows(canonical_rows)
        output_path = output_dir_path / f"{manifest_record.asset_id}_macd_features.csv"
        write_macd_features_csv(rows, output_path)
        output_paths.append(output_path)

    for manifest_record in manifest_records:
        expected_output_path = output_dir_path / f"{manifest_record.asset_id}_macd_features.csv"
        if not expected_output_path.exists():
            raise ValueError(f"Missing MACD output for {manifest_record.asset_id}")
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized MACD feature files."
    )
    parser.add_argument(
        "--canonical-manifest-input",
        type=Path,
        default=default_canonical_manifest_input(),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_paths = build_t11_outputs(
        canonical_manifest_input=args.canonical_manifest_input,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(f"Wrote {len(output_paths)} MACD feature files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
