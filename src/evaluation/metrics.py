"""Evaluation metrics for demand forecasting."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1 - ss_res / ss_tot)


def percentile_error(y_true: np.ndarray, y_pred: np.ndarray, p: int = 95) -> float:
    return float(np.percentile(np.abs(y_true - y_pred), p))


def rmse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_actual = np.mean(np.abs(y_true))
    if mean_actual == 0:
        return 0.0
    return rmse(y_true, y_pred) / mean_actual * 100


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1) -> float:
    """Mean absolute scaled error. Denominator uses TRAINING observations only."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    history = np.asarray(y_train, dtype=float)
    if seasonality < 1 or len(history) <= seasonality:
        raise ValueError("Training history must exceed positive seasonality")
    scale = np.mean(np.abs(history[seasonality:] - history[:-seasonality]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(truth - prediction)) / scale)


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical fraction of test targets inside the predicted P10-P90 band."""
    truth, lo, hi = [np.asarray(x, dtype=float) for x in (y_true, lower, upper)]
    if truth.shape != lo.shape or lo.shape != hi.shape or truth.size == 0:
        raise ValueError("Expected equally shaped, non-empty arrays")
    if np.any(lo > hi):
        raise ValueError("Interval lower bound exceeds upper bound")
    return float(np.mean((truth >= lo) & (truth <= hi)))


def interval_mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.shape != hi.shape or lo.size == 0 or np.any(lo > hi):
        raise ValueError("Invalid prediction intervals")
    return float(np.mean(hi - lo))
