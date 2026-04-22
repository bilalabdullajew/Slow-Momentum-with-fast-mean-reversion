from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from lstm_cpd.daily_close_contract import default_project_root
from lstm_cpd.features.macd import T11_OUTPUT_HEADER
from lstm_cpd.features.normalized_returns import T10_OUTPUT_HEADER
from lstm_cpd.features.returns import serialize_optional_float


NORMALIZED_RETURNS_SUFFIX = "_normalized_returns.csv"
MACD_FEATURES_SUFFIX = "_macd_features.csv"
BASE_FEATURES_SUFFIX = "_base_features.csv"

NORMALIZED_RETURN_COLUMNS = T10_OUTPUT_HEADER[2:]
MACD_FEATURE_COLUMNS = T11_OUTPUT_HEADER[2:]
FEATURE_COLUMNS = NORMALIZED_RETURN_COLUMNS + MACD_FEATURE_COLUMNS
T12_OUTPUT_HEADER = ("timestamp", "asset_id") + FEATURE_COLUMNS

WINSORIZATION_HALFLIFE = 252
WINSORIZATION_SIGMA_MULTIPLE = 5.0
EWM_ADJUST = False
EWM_BIAS = False


def default_input_dir() -> Path:
    return default_project_root() / "artifacts/features/base"


def default_output_dir() -> Path:
    return default_project_root() / "artifacts/features/base"


def default_report_path() -> Path:
    return default_project_root() / "artifacts/reports/feature_provenance_report.md"


def _validate_feature_text(
    *,
    csv_path: Path,
    row_index: int,
    column_name: str,
    text: str,
) -> None:
    if text == "":
        return
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(
            f"Row {row_index} column {column_name} has non-finite value: {csv_path}"
        )


def load_feature_csv(
    path: Path | str,
    *,
    expected_header: Sequence[str],
    expected_asset_id: str | None = None,
) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_header):
            raise ValueError(f"Feature header mismatch: {csv_path}")

        rows: list[dict[str, str]] = []
        previous_timestamp: str | None = None
        for row_index, row in enumerate(reader):
            timestamp = row["timestamp"]
            asset_id = row["asset_id"]
            if not timestamp:
                raise ValueError(f"Feature row {row_index} missing timestamp: {csv_path}")
            if not asset_id:
                raise ValueError(f"Feature row {row_index} missing asset_id: {csv_path}")
            if expected_asset_id is not None and asset_id != expected_asset_id:
                raise ValueError(
                    f"Feature row {row_index} asset_id mismatch for {csv_path}: {asset_id}"
                )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(
                    f"Feature timestamps must be strictly ascending in {csv_path}: "
                    f"{previous_timestamp} then {timestamp}"
                )
            previous_timestamp = timestamp
            for column_name in expected_header[2:]:
                _validate_feature_text(
                    csv_path=csv_path,
                    row_index=row_index,
                    column_name=column_name,
                    text=row[column_name],
                )
            rows.append(dict(row))
    return rows


def load_normalized_returns_csv(
    path: Path | str,
    expected_asset_id: str | None = None,
) -> list[dict[str, str]]:
    return load_feature_csv(
        path,
        expected_header=T10_OUTPUT_HEADER,
        expected_asset_id=expected_asset_id,
    )


def load_macd_features_csv(
    path: Path | str,
    expected_asset_id: str | None = None,
) -> list[dict[str, str]]:
    return load_feature_csv(
        path,
        expected_header=T11_OUTPUT_HEADER,
        expected_asset_id=expected_asset_id,
    )


def join_feature_rows(
    normalized_rows: Sequence[dict[str, str]],
    macd_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    if len(normalized_rows) != len(macd_rows):
        raise ValueError(
            f"Feature row-count mismatch: {len(normalized_rows)} != {len(macd_rows)}"
        )

    joined_rows: list[dict[str, str]] = []
    for row_index, (normalized_row, macd_row) in enumerate(
        zip(normalized_rows, macd_rows)
    ):
        normalized_key = (normalized_row["timestamp"], normalized_row["asset_id"])
        macd_key = (macd_row["timestamp"], macd_row["asset_id"])
        if normalized_key != macd_key:
            raise ValueError(
                f"Feature alignment mismatch at row {row_index}: "
                f"{normalized_key} != {macd_key}"
            )

        joined_row = {
            "timestamp": normalized_row["timestamp"],
            "asset_id": normalized_row["asset_id"],
        }
        for column_name in FEATURE_COLUMNS:
            source_row = normalized_row if column_name in NORMALIZED_RETURN_COLUMNS else macd_row
            joined_row[column_name] = source_row[column_name]
        joined_rows.append(joined_row)
    return joined_rows


def _parse_optional_float(text: str) -> float | None:
    if text == "":
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Encountered non-finite feature value: {text}")
    return value


def winsorize_feature_values(
    values: Sequence[float | None],
    *,
    halflife: float = WINSORIZATION_HALFLIFE,
    sigma_multiple: float = WINSORIZATION_SIGMA_MULTIPLE,
    adjust: bool = EWM_ADJUST,
    bias: bool = EWM_BIAS,
) -> list[float | None]:
    if halflife <= 0:
        raise ValueError(f"halflife must be positive: {halflife}")
    if sigma_multiple < 0:
        raise ValueError(f"sigma_multiple must be non-negative: {sigma_multiple}")
    if not values:
        return []

    series = pd.Series(list(values), dtype="float64")
    mean_series = series.ewm(halflife=halflife, adjust=adjust).mean()
    std_series = series.ewm(halflife=halflife, adjust=adjust).std(bias=bias)

    winsorized_values: list[float | None] = []
    for original, mean_value, std_value in zip(
        series.tolist(),
        mean_series.tolist(),
        std_series.tolist(),
    ):
        if pd.isna(original):
            winsorized_values.append(None)
            continue

        original_value = float(original)
        if pd.isna(mean_value) or pd.isna(std_value):
            winsorized_values.append(original_value)
            continue

        mean_float = float(mean_value)
        std_float = float(std_value)
        if not math.isfinite(mean_float) or not math.isfinite(std_float) or std_float < 0.0:
            winsorized_values.append(original_value)
            continue

        upper_bound = mean_float + sigma_multiple * std_float
        lower_bound = mean_float - sigma_multiple * std_float
        winsorized_values.append(min(max(original_value, lower_bound), upper_bound))
    return winsorized_values


def build_base_feature_rows(
    joined_rows: Sequence[dict[str, str]],
    *,
    halflife: float = WINSORIZATION_HALFLIFE,
    sigma_multiple: float = WINSORIZATION_SIGMA_MULTIPLE,
) -> list[dict[str, str]]:
    feature_values = {
        column_name: [
            _parse_optional_float(row[column_name]) for row in joined_rows
        ]
        for column_name in FEATURE_COLUMNS
    }
    winsorized_columns = {
        column_name: [
            serialize_optional_float(value)
            for value in winsorize_feature_values(
                values,
                halflife=halflife,
                sigma_multiple=sigma_multiple,
            )
        ]
        for column_name, values in feature_values.items()
    }

    output_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(joined_rows):
        output_row = {
            "timestamp": row["timestamp"],
            "asset_id": row["asset_id"],
        }
        for column_name in FEATURE_COLUMNS:
            output_row[column_name] = winsorized_columns[column_name][row_index]
        output_rows.append(output_row)
    return output_rows


def write_base_features_csv(
    rows: Sequence[dict[str, str]],
    output_path: Path | str,
) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(T12_OUTPUT_HEADER),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_asset_path_map(
    input_dir: Path | str,
    *,
    suffix: str,
) -> dict[str, Path]:
    input_dir_path = Path(input_dir)
    asset_map: dict[str, Path] = {}
    for path in sorted(input_dir_path.glob(f"*{suffix}")):
        if not path.is_file():
            continue
        asset_id = path.name[: -len(suffix)]
        if asset_id in asset_map:
            raise ValueError(f"Duplicate input asset_id for suffix {suffix}: {asset_id}")
        asset_map[asset_id] = path
    return asset_map


def collect_input_file_pairs(input_dir: Path | str) -> list[tuple[str, Path, Path]]:
    normalized_map = _build_asset_path_map(
        input_dir,
        suffix=NORMALIZED_RETURNS_SUFFIX,
    )
    macd_map = _build_asset_path_map(
        input_dir,
        suffix=MACD_FEATURES_SUFFIX,
    )
    if not normalized_map:
        raise ValueError(f"No normalized-return inputs found under {Path(input_dir)}")
    if not macd_map:
        raise ValueError(f"No MACD inputs found under {Path(input_dir)}")

    normalized_assets = set(normalized_map)
    macd_assets = set(macd_map)
    if normalized_assets != macd_assets:
        only_normalized = sorted(normalized_assets - macd_assets)
        only_macd = sorted(macd_assets - normalized_assets)
        raise ValueError(
            "Input asset mismatch between normalized returns and MACD features: "
            f"only_normalized={only_normalized}, only_macd={only_macd}"
        )

    return [
        (asset_id, normalized_map[asset_id], macd_map[asset_id])
        for asset_id in sorted(normalized_assets)
    ]


def write_feature_provenance_report(
    report_path: Path | str,
    *,
    asset_count: int,
) -> None:
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# Feature Provenance Report",
            "",
            "## Summary",
            "",
            f"- Generated asset count: {asset_count}",
            "- Output artifact family: `artifacts/features/base/<asset_id>_base_features.csv`",
            "- Output schema: `timestamp`, `asset_id`, five normalized returns, three MACD features",
            "",
            "## Upstream Dependencies",
            "",
            "- Arithmetic returns use close-to-close arithmetic returns from T-09, not log returns.",
            "- Volatility uses the T-09 60-day EWM estimate with `span=60`, `adjust=False`, `min_periods=60`, and `bias=False`.",
            "- Normalized-return inputs come from T-10.",
            "- MACD inputs come from T-11.",
            "",
            "## Feature Definitions",
            "",
            "- Normalized return horizons are exactly `{1,21,63,126,256}`.",
            "- MACD pairs are exactly `{(8,24),(16,28),(32,96)}`.",
            "- Final non-CPD feature set contains exactly 8 features per asset-date.",
            "",
            "## Winsorization Contract",
            "",
            "- Winsorization applies only to the 8 non-CPD features.",
            "- Raw close prices are not winsorized.",
            "- CPD features are not winsorized and are not appended here.",
            "- Causal clipping uses trailing EWM mean plus/minus `5.0` trailing EWM standard deviations.",
            "- Winsorization half-life is `252`.",
            "- EWM implementation uses Pandas with `halflife=252`, `adjust=False`, and `std(bias=False)`.",
            "- Missing upstream feature values remain blank; no imputation is performed.",
            "",
            "## Output Columns",
            "",
            f"- `{', '.join(T12_OUTPUT_HEADER)}`",
            "",
        ]
    )
    report_file.write_text(content + "\n", encoding="utf-8")


def build_t12_outputs(
    input_dir: Path | str = default_input_dir(),
    output_dir: Path | str = default_output_dir(),
    report_path: Path | str = default_report_path(),
) -> list[Path]:
    file_pairs = collect_input_file_pairs(input_dir)
    output_dir_path = Path(output_dir)

    output_paths: list[Path] = []
    for asset_id, normalized_path, macd_path in file_pairs:
        normalized_rows = load_normalized_returns_csv(
            normalized_path,
            expected_asset_id=asset_id,
        )
        macd_rows = load_macd_features_csv(
            macd_path,
            expected_asset_id=asset_id,
        )
        joined_rows = join_feature_rows(normalized_rows, macd_rows)
        output_rows = build_base_feature_rows(joined_rows)
        output_path = output_dir_path / f"{asset_id}{BASE_FEATURES_SUFFIX}"
        write_base_features_csv(output_rows, output_path)
        output_paths.append(output_path)

    for asset_id, _, _ in file_pairs:
        expected_output_path = output_dir_path / f"{asset_id}{BASE_FEATURES_SUFFIX}"
        if not expected_output_path.exists():
            raise ValueError(f"Missing base-feature output for {asset_id}")

    write_feature_provenance_report(report_path, asset_count=len(output_paths))
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build winsorized non-CPD base feature files."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--report-path", type=Path, default=default_report_path())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_paths = build_t12_outputs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
    )
    print(f"Wrote {len(output_paths)} base-feature files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
