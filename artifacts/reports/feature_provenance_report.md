# Feature Provenance Report

## Summary

- Generated asset count: 126
- Output artifact family: `artifacts/features/base/<asset_id>_base_features.csv`
- Output schema: `timestamp`, `asset_id`, five normalized returns, three MACD features

## Upstream Dependencies

- Arithmetic returns use close-to-close arithmetic returns from T-09, not log returns.
- Volatility uses the T-09 60-day EWM estimate with `span=60`, `adjust=False`, `min_periods=60`, and `bias=False`.
- Normalized-return inputs come from T-10.
- MACD inputs come from T-11.

## Feature Definitions

- Normalized return horizons are exactly `{1,21,63,126,256}`.
- MACD pairs are exactly `{(8,24),(16,28),(32,96)}`.
- Final non-CPD feature set contains exactly 8 features per asset-date.

## Winsorization Contract

- Winsorization applies only to the 8 non-CPD features.
- Raw close prices are not winsorized.
- CPD features are not winsorized and are not appended here.
- Causal clipping uses trailing EWM mean plus/minus `5.0` trailing EWM standard deviations.
- Winsorization half-life is `252`.
- EWM implementation uses Pandas with `halflife=252`, `adjust=False`, and `std(bias=False)`.
- Missing upstream feature values remain blank; no imputation is performed.

## Output Columns

- `timestamp, asset_id, normalized_return_1, normalized_return_21, normalized_return_63, normalized_return_126, normalized_return_256, macd_8_24, macd_16_28, macd_32_96`

