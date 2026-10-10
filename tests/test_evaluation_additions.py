"""Tests for leakage-safe scaled errors and interval diagnostics."""
import numpy as np
import pytest
from src.evaluation.metrics import mase, interval_coverage, interval_mean_width

def test_mase_uses_training_scale():
    train=np.array([10.,20.,30.,40.])
    truth=np.array([50.,60.])
    pred=np.array([40.,50.])
    assert mase(truth,pred,train,seasonality=1)==pytest.approx(1.)

def test_mase_rejects_short_history():
    with pytest.raises(ValueError):
        mase(np.array([1.]),np.array([1.]),np.array([1.]),seasonality=1)

def test_interval_coverage():
    y=np.array([1.,2.,3.,4.])
    lo=np.array([0.,0.,4.,3.])
    hi=np.array([2.,1.,5.,5.])
    assert interval_coverage(y,lo,hi)==pytest.approx(0.5)
    assert interval_mean_width(lo,hi)==pytest.approx(1.5)

def test_invalid_interval_order():
    with pytest.raises(ValueError):
        interval_coverage(np.array([1.]),np.array([2.]),np.array([1.]))
