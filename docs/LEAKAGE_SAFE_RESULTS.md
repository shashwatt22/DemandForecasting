# Leakage-aware forecasting benchmark — October 10, 2026

**Source data:** synthetic CSV datasets already included in the repository.
**Execution:** https://github.com/shashwatt22/DemandForecasting/actions/runs/38071152281
**Script:** `src/evaluation/leakage_safe_benchmark.py`

This is a separate research/evaluation pipeline, **not** the original deployed saved models. It assumes one-step-ahead forecasts with past realized target values available and uses 70% chronological training / 10% chronological interval calibration / 20% final test data per domain.

## Held-out synthetic-data results

| Metric | Airline | E-commerce | Payments |
|---|---:|---:|---:|
| Train / calibration / test rows | 827 / 118 / 237 | 2506 / 358 / 716 | 6092 / 871 / 1741 |
| 7-observation seasonal-naive RMSE | 41.080 | 98.600 | 350.156 |
| Ridge stack RMSE | 13.604 | 30.288 | 75.160 |
| Ridge stack MASE (scale from training history) | 0.344 | 0.241 | 0.303 |
| Reduction vs 7-observation seasonal naive | 66.88% | 69.28% | 78.54% |
| Reduction vs best tested simple baseline | 57.79% | 58.96% | 65.48% |
| Raw P10–P90 coverage | 54.01% | 71.65% | 68.06% |
| Calibrated P10–P90 coverage | 75.95% | 74.72% | 71.74% |

The purported **7-observation seasonal naive is *not* necessarily last week's value**, particularly for multiseries data. Do not call it a 7-day seasonal-naive benchmark. In this separate experiment the additional ETS-labeled comparator is only a simple exponential-level proxy (`0.5 * latest_observation + 0.5 * historical_7_mean`), not a fitted ETS model.

### Leakage audit

- **Target lags:** `groupby(...).shift(k)`; no current target is copied into a predictor.
- **Rolling features:** `shift(1).rolling(window)`; calculated exclusively using previous target observations.
- **Ridge meta-learner:** trained only on base-model *out-of-fold* predictions from `TimeSeriesSplit`, not fitted base predictions.
- **Encoders/scalers:** new research pipeline consumes numeric exogenous features and historical targets only; there is no full-dataset fitted scaling or category encoder. The original deployed pipeline uses different feature code and still requires retraining.
- **Calibration:** quantile models fit on initial training window; residual-based conformal adjustment learned on next 10%; last 20% used for evaluation only.

### Model ablation, fold error analysis and importances

Workflow artifact `leakage-safe-forecast-diagnostics` includes all four test configurations (XGBoost, LightGBM, their average, Ridge stack), 3 expanding training-fold RMSEs for each base model, and top XGBoost feature importances per domain. **Do not assert the Ridge stack always wins:** it was worse than LightGBM alone on e-commerce in this run.

### Limitations / next

- All three datasets are synthetic; these results are not evidence of production performance.
- The interval coverage target is nominally 80%; payments reaches only 71.74% even after calibration.
- Historical one-step-ahead lag features assume recent actual values are already available. Multistep forecasting would require a different design.
- The existing production/deployed artifacts **have not been retrained** using this research feature pipeline. Do not quote these evaluation numbers as deployed-app performance.
- Next: use actual time-aware seasonality (24/168 hours for hourly series), group-specific naive baselines and fitted ETS; retrain artifact pipeline only after test compatibility checks.
