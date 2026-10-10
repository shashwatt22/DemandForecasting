# Evaluation audit and next steps (October 2026)

## Implemented metrics

`src/evaluation/metrics.py` provides **MASE(7)** calculated from the training set, **empirical P10–P90 coverage** and mean interval width. `src/models/train_all.py` compares ensemble RMSE to the **existing weekly seasonal-naive baseline** and saves all diagnostic columns when retrained. `tests/test_evaluation_additions.py` checks the formulas.

## Critical pre-existing risk: target leakage

The original feature engineering uses the contemporaneous target inside predictors:

- `src/features/common.py:add_rolling_features` computes rolling means, maxima and standard deviations on the **current** target rather than a shifted historical series.
- `add_trend_direction` similarly includes the current target.
- Airline `booking_velocity` and `load_factor` depend on current bookings.
- Payments `avg_txn_value = value / volume` depends on current volume.

This risks substantial look-ahead leakage for real forecasting. The quoted very large reductions on synthetic datasets **must not be interpreted as evidence of operational forecasting performance** until the forecast horizon and feature availability are defined, target-dependent features are shifted or removed, and models are retrained and re-evaluated chronologically.

## Safe retraining work required

1. Define whether predictions are one-step-ahead or multi-step-ahead and what data are known at forecast time.
2. Shift target-derived rolling/trend features by at least one interval for one-step-ahead predictions; drop or recursively forecast any features unavailable over multi-step horizons.
3. Remove same-period target derivatives (booking velocity/load factor, avg transaction value) or replace with lagged, genuinely available measures.
4. Group and order time series appropriately and split chronologically; maintain an untouched final test period.
5. Retrain the ensemble and quantile models. Existing saved model artifacts are **not** valid evidence for a redesigned feature pipeline.
6. Report seasonal naive, moving average, individual XGBoost/LightGBM, stacked RMSE, MASE, interval coverage and actual sample sizes.
7. Never claim that the newly added metric formulas have produced measured values until a successful retraining/evaluation run generates them.

These changes are deliberately **not applied directly to the deployed feature definitions**, because doing so without retraining would silently make saved model predictions inconsistent with their training data.
