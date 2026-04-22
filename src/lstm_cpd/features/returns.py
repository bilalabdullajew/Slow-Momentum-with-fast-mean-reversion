from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CANONICAL_DAILY_CLOSE_HEADER = ("timestamp", "asset_id", "close")
RETURNS_VOLATILITY_HEADER = (
    "timestamp",
    "asset_id",
    "close",
    "arithmetic_return",
    "sigma_t",
)


@dataclass(frozen=True)
class CanonicalDailyCloseRecord:
    timestamp: str
    asset_id: str
    close_text: str
    close_value: float


def load_canonical_daily_close_csv(
    path: Path | str,
    expected_asset_id: str | None = None,
) -> list[CanonicalDailyCloseRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANONICAL_DAILY_CLOSE_HEADER:
            raise ValueError(f"Canonical daily-close header mismatch: {csv_path}")

        records: list[CanonicalDailyCloseRecord] = []
        for index, row in enumerate(reader):
            timestamp = row["timestamp"]
            asset_id = row["asset_id"]
            close_text = row["close"]
            if not timestamp:
                raise ValueError(f"Canonical row {index} missing timestamp: {csv_path}")
            if not asset_id:
                raise ValueError(f"Canonical row {index} missing asset_id: {csv_path}")
            if expected_asset_id is not None and asset_id != expected_asset_id:
                raise ValueError(
                    f"Canonical row {index} asset_id mismatch for {csv_path}: {asset_id}"
                )
            if close_text == "":
                raise ValueError(f"Canonical row {index} missing close: {csv_path}")

            close_value = float(close_text)
            if not math.isfinite(close_value):
                raise ValueError(f"Canonical row {index} has non-finite close: {csv_path}")

            records.append(
                CanonicalDailyCloseRecord(
                    timestamp=timestamp,
                    asset_id=asset_id,
                    close_text=close_text,
                    close_value=close_value,
                )
            )
    return records


def compute_arithmetic_returns(
    canonical_rows: Sequence[CanonicalDailyCloseRecord],
) -> list[float | None]:
    if not canonical_rows:
        return []

    arithmetic_returns: list[float | None] = [None]
    for previous_row, current_row in zip(canonical_rows, canonical_rows[1:]):
        previous_close = previous_row.close_value
        current_close = current_row.close_value
        if previous_close == 0.0:
            arithmetic_returns.append(None)
            continue
        arithmetic_returns.append((current_close - previous_close) / previous_close)
    return arithmetic_returns


def serialize_optional_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = format(value, ".17g")
    if text == "-0":
        return "0"
    return text

