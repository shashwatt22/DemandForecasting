"""v2 temporal invariants: no future target enters features or calibration."""
import numpy as np
import pandas as pd
import pytest

from src.models.leakage_safe_v2 import feature_row, to_supervised, recursive_forecast

def test_future_target_change_cannot_modify_earlier_features():
    dates=pd.date_range("2024-01-01",periods=65,freq="D")
    values=np.arange(65.,dtype=float)
    first=pd.Series(values,index=dates)
    altered=first.copy()
    altered.iloc[45:]=999999.
    Xa,ya=to_supervised(first,period=7)
    Xb,yb=to_supervised(altered,period=7)
    pd.testing.assert_frame_equal(Xa.iloc[:31],Xb.iloc[:31])
    assert ya.iloc[:31].equals(yb.iloc[:31])

def test_step_features_only_use_past():
    timestamp=pd.Timestamp("2024-02-01")
    history=list(range(30))
    r=feature_row(history,timestamp,7)
    assert r["lag_1"]==29
    assert r["lag_season"]==23
    assert r["lag_two_seasons"]==16
    assert r["mean_3"]==pytest.approx(28)

class DummyModel:
    def predict(self,frame):
        return np.array([20.],dtype=float)

class DummyRidge:
    def predict(self,frame):
        return np.array([20.],dtype=float)

def test_recursive_prediction_does_not_need_future_truths():
    bundle={"period":7,"freq":"D","max_horizon":5,
            "xgb":DummyModel(),"lgb":DummyModel(),"ridge":DummyRidge()}
    result=recursive_forecast(bundle,list(range(20)),pd.Timestamp("2024-01-20"),5)
    assert result==[20.]*5

def test_invalid_horizon_rejected():
    bundle={"period":7,"freq":"D","max_horizon":3}
    with pytest.raises(ValueError):
        recursive_forecast(bundle,list(range(20)),pd.Timestamp("2024-01-20"),4)
