# Validation Evaluation Report

- candidate_id: `C-031`
- lbw: `63`
- model_path: `artifacts/training/candidates/candidate_030/best_model.keras`
- dataset_registry: `artifacts/manifests/dataset_registry.json`

## Raw Validation Metrics

- annualized_return: `0.21438763520238976`
- annualized_volatility: `0.21679880913575342`
- annualized_downside_deviation: `0.15688449960167444`
- sharpe_ratio: `0.9888782879252171`
- sortino_ratio: `1.3665316570261195`
- maximum_drawdown: `-0.27992979502165105`
- calmar_ratio: `0.7658621519220847`
- percentage_positive_daily_returns: `53.68956743002544`

## Rescaled Validation Metrics

- annualized_return: `0.14833174318878264`
- annualized_volatility: `0.15`
- annualized_downside_deviation: `0.1085461448522794`
- sharpe_ratio: `0.9888782879252177`
- sortino_ratio: `1.3665316570261203`
- maximum_drawdown: `-0.1995505539651704`
- calmar_ratio: `0.7433291476337995`
- percentage_positive_daily_returns: `53.68956743002544`
- rescaling_factor: `0.6918857192895103`

## Artifacts

- raw_validation_returns: `artifacts/evaluation/raw_validation_returns.csv`
- raw_validation_metrics: `artifacts/evaluation/raw_validation_metrics.json`
- rescaled_validation_returns: `artifacts/evaluation/rescaled_validation_returns.csv`
- rescaled_validation_metrics: `artifacts/evaluation/rescaled_validation_metrics.json`

## Scope Note

FTMO evaluation is model-faithful but not a literal reproduction of the paper's 50-futures universe.
