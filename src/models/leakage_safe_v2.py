"""Leakage-safe v2 forecasting service: chronological OOF stack and multi-step intervals.

TRAINING NEVER EVALUATES OR ACCESSES THE FINAL 20% TARGETS. Calibration
uses the next chronological development window, not the sealed test window.

Input/serving semantics: single aggregated time series per domain. This v2
research model cannot use the legacy API's exogenous-feature request format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "v2"
PARAMS = {
    "airline": {"period": 12, "freq": "MS", "horizon": 3},
    "ecommerce": {"period": 7, "freq": "D", "horizon": 7},
    "payment": {"period": 168, "freq": "h", "horizon": 24},
}

def known_time_features(timestamp: pd.Timestamp) -> dict[str, float]:
    return {
        "month_sin": float(np.sin(2 * np.pi * timestamp.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * timestamp.month / 12)),
        "weekday_sin": float(np.sin(2 * np.pi * timestamp.dayofweek / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * timestamp.dayofweek / 7)),
        "hour_sin": float(np.sin(2 * np.pi * timestamp.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * timestamp.hour / 24)),
    }

def feature_row(history: list[float], timestamp: pd.Timestamp, period: int) -> dict[str, float]:
    """Past-only features. NO contemporaneous target, no future target."""
    if period < 1 or len(history) < 2 * period:
        raise ValueError("At least two complete seasons of history required")
    if not np.isfinite(np.asarray(history, dtype=float)).all():
        raise ValueError("Forecast history must contain only finite values")
    row = {
        "lag_1": float(history[-1]),
        "lag_season": float(history[-period]),
        "lag_two_seasons": float(history[-2 * period]),
        "mean_3": float(np.mean(history[-3:])),
        "mean_season": float(np.mean(history[-period:])),
        "std_season": float(np.std(history[-period:], ddof=0)),
    }
    row.update(known_time_features(timestamp))
    return row

def to_supervised(history: pd.Series, period: int) -> tuple[pd.DataFrame, pd.Series]:
    """At time i only [0, ..., i-1] are used as features."""
    values = history.to_numpy(dtype=float).tolist()
    columns, indices, targets = [], [], []
    for i in range(2 * period, len(values)):
        columns.append(feature_row(values[:i], history.index[i], period))
        indices.append(history.index[i])
        targets.append(values[i])
    if not columns:
        raise ValueError("Insufficient series length")
    return pd.DataFrame(columns, index=indices), pd.Series(targets, index=indices)

def base_models():
    return (
        XGBRegressor(n_estimators=110, max_depth=3, learning_rate=0.06,
                     n_jobs=2, random_state=42, verbosity=0),
        LGBMRegressor(n_estimators=110, num_leaves=12, learning_rate=0.06,
                      n_jobs=2, random_state=42, verbose=-1),
    )

def one_step(bundle: dict[str, Any], history: list[float], stamp: pd.Timestamp,
             estimator: str = "ridge") -> float:
    inputs = pd.DataFrame([feature_row(history, stamp, bundle["period"])])
    xgb = float(bundle["xgb"].predict(inputs)[0])
    lgb = float(bundle["lgb"].predict(inputs)[0])
    if estimator == "xgb":
        result = xgb
    elif estimator == "lgb":
        result = lgb
    elif estimator == "ridge":
        result = float(bundle["ridge"].predict(np.array([[xgb, lgb]]))[0])
    else:
        raise ValueError("Unknown forecasting estimator")
    return max(0.0, result)

def recursive_forecast(bundle: dict[str, Any], history: list[float],
                       last_time: pd.Timestamp, steps: int,
                       estimator: str = "ridge") -> list[float]:
    if not 1 <= steps <= bundle["max_horizon"]:
        raise ValueError("Requested horizon outside calibrated range")
    values = list(map(float, history))
    future = pd.date_range(last_time, periods=steps + 1, freq=bundle["freq"])[1:]
    predicted = []
    for time in future:
        value = one_step(bundle, values, time, estimator=estimator)
        values.append(value)
        predicted.append(value)
    return predicted

def calibrated_forecast(bundle: dict[str, Any], horizon: int) -> dict:
    values = recursive_forecast(bundle, bundle["history"], pd.Timestamp(bundle["last_time"]), horizon)
    radii = bundle["radii"][:horizon]
    return {
        "domain": bundle["domain"],
        "origin": bundle["last_time"],
        "frequency": bundle["freq"],
        "forecast": values,
        "lower_bound": [max(0.0, p-r) for p,r in zip(values,radii)],
        "upper_bound": [p+r for p,r in zip(values,radii)],
        "horizon": horizon,
        "test_split_used": False,
        "note": "Historical research forecast from end of development data; "
                "nominal 80% intervals calibrated by horizon on development data.",
    }

def fit_domain(domain: str, series: pd.Series) -> dict:
    if domain not in PARAMS:
        raise ValueError("Unknown domain")
    cfg = PARAMS[domain]
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex required")
    series = series.sort_index().astype(float)
    if not series.index.is_unique:
        raise ValueError("Duplicate timestamps are not allowed")
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError("Training series must contain finite observations")
    n = len(series)
    dev_end = int(n * 0.8)
    train_end = int(n * 0.7)
    # Target values at indices >= dev_end are a SEALED TEST and never read.
    development = series.iloc[:dev_end]
    training = development.iloc[:train_end]
    X, y = to_supervised(training, cfg["period"])
    if len(X) < 32:
        raise ValueError("Too few supervised training samples")
    oof = np.full((len(X), 2), np.nan)
    for a, b in TimeSeriesSplit(n_splits=3).split(X):
        model1, model2 = base_models()
        model1.fit(X.iloc[a], y.iloc[a])
        model2.fit(X.iloc[a], y.iloc[a])
        oof[b,0] = model1.predict(X.iloc[b])
        oof[b,1] = model2.predict(X.iloc[b])
    valid = np.isfinite(oof).all(axis=1)
    ridge = Ridge(alpha=1.0).fit(oof[valid], y.iloc[np.flatnonzero(valid)])
    xgb,lgb = base_models()
    xgb.fit(X,y)
    lgb.fit(X,y)
    bundle = {
        "domain": domain, "period": cfg["period"],
        "freq": cfg["freq"], "max_horizon": cfg["horizon"],
        "xgb": xgb, "lgb": lgb, "ridge": ridge,
    }
    # Nonconformity computed at multiple rolling origins entirely WITHIN
    # the calibration period. At each origin, future labels are used for
    # CALIBRATION ONLY, never as model features at that origin.
    horizon = cfg["horizon"]
    origin_indices = list(range(train_end,dev_end-horizon+1,max(1,horizon)))
    if not origin_indices:
        raise ValueError("Insufficient calibration window")
    residuals = [[] for _ in range(horizon)]
    dev_values = development.to_numpy(dtype=float).tolist()
    for i in origin_indices:
        predictions = recursive_forecast(
            bundle, dev_values[:i], development.index[i-1], horizon,
        )
        for h,prediction in enumerate(predictions):
            residuals[h].append(abs(dev_values[i+h]-prediction))
    radii=[]
    for errors in residuals:
        level=min(1.,np.ceil((len(errors)+1)*0.8)/len(errors))
        radii.append(float(np.quantile(errors,level,method="higher")))
    bundle.update({
        "radii": radii,
        "history": dev_values,
        "last_time": str(development.index[-1]),
        "sealed_test_rows": n-dev_end,
        "development_rows": dev_end,
        "train_rows": train_end,
        "calibration_origins": len(origin_indices),
        "sealed_evaluated": False,
    })
    return bundle

def get_payment_series() -> pd.Series:
    frame = pd.read_csv(ROOT/"data"/"payment_volume.csv")
    times = pd.to_datetime(frame["timestamp"]).dt.floor("h")
    return (pd.DataFrame({"timestamp":times,"volume":frame["volume"]})
            .groupby("timestamp")["volume"].sum().sort_index()
            .asfreq("h",fill_value=0))

def train_all() -> None:
    from src.evaluation.public_backtesting import air_passengers, retail_demand

    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    datasets = {
        "airline": air_passengers(),
        "ecommerce": retail_demand(),
        "payment": get_payment_series(),
    }
    for domain,series in datasets.items():
        bundle=fit_domain(domain,series)
        joblib.dump(bundle, MODEL_DIR/f"{domain}.joblib",compress=3)
        # Only development calibration metadata is printed, no holdout score.
        print(json.dumps({
            "domain":domain,"training":bundle["train_rows"],
            "development":bundle["development_rows"],
            "sealed_test_rows":bundle["sealed_test_rows"],
            "sealed_evaluated":False,
            "calibration_origins":bundle["calibration_origins"],
            "nominal_coverage":0.8,
        }))

if __name__=="__main__":
    train_all()
