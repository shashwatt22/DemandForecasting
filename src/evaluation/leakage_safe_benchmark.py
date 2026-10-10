"""Leakage-aware one-step-ahead research benchmark.

Independent of deployed saved models. Forecast assumes the previous outcome for
a series is available before predicting the next row. Generated domain datasets
remain synthetic and should not be presented as operational observations.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.evaluation.metrics import mase, interval_coverage, interval_mean_width

ROOT=Path(__file__).resolve().parents[2]
SETTINGS={
    "airline":("airline_bookings.csv","date","departure_date","bookings",7),
    "ecommerce":("ecommerce_demand.csv","timestamp","product_category","quantity_sold",7),
    "payment":("payment_volume.csv","timestamp","transaction_type","volume",7),
}
EXCLUDE={
    "airline":{"bookings"},
    "ecommerce":{"quantity_sold"},
    "payment":{"volume","value","fraud_flag"},  # value depends on volume at time t
}

def assemble(raw:pd.DataFrame,domain:str):
    file,time,group,target,season=SETTINGS[domain]
    d=raw.copy().sort_values([time,group],kind="stable").reset_index(drop=True)
    y=d[target].astype(float)
    numeric=[k for k in d.select_dtypes(include="number").columns if k not in EXCLUDE[domain]]
    X=d[numeric].copy()
    group_by=d.groupby(group,sort=False)[target]
    for lag in [1,7,14]:
        X[f"historical_lag_{lag}"]=group_by.shift(lag)
    # shift BEFORE rolling. Every statistic uses past observations only.
    for window in [7,14]:
        X[f"historical_mean_{window}"]=group_by.transform(
            lambda series: series.shift(1).rolling(window,min_periods=window).mean())
    keep=X.notna().all(axis=1)
    X=X.loc[keep].reset_index(drop=True).astype(float)
    y=y.loc[keep].reset_index(drop=True)
    if len(y)<140:
        raise ValueError("Not enough chronological observations for "+domain)
    return X,y

def model_pair():
    xgb=XGBRegressor(n_estimators=80,max_depth=4,learning_rate=.07,
                     n_jobs=2,random_state=42,verbosity=0)
    lgb=LGBMRegressor(n_estimators=80,num_leaves=15,learning_rate=.07,
                      n_jobs=2,random_state=42,verbose=-1)
    return xgb,lgb

def root_mean_sq(y,p):
    return float(np.sqrt(mean_squared_error(y,p)))

def diagnostics(domain,X,y):
    n=len(y)
    n_train=int(.70*n)
    n_cal=int(.80*n)
    train=X.iloc[:n_train]
    val=X.iloc[n_train:n_cal]
    test=X.iloc[n_cal:]
    y_train=y.iloc[:n_train]
    y_val=y.iloc[n_train:n_cal]
    y_test=y.iloc[n_cal:]
    if len(train)<=25 or len(val)<=10:
        raise ValueError("Too few observations")

    # Walk-forward OOF: early entries have no forecast and are not used.
    oof=np.full((len(train),2),np.nan)
    cv=TimeSeriesSplit(n_splits=3)
    folds=[]
    for fold,(tr,va) in enumerate(cv.split(train),start=1):
        xa,xb=model_pair()
        xa.fit(train.iloc[tr],y_train.iloc[tr])
        xb.fit(train.iloc[tr],y_train.iloc[tr])
        oof[va,0]=xa.predict(train.iloc[va])
        oof[va,1]=xb.predict(train.iloc[va])
        folds.append({"fold":fold,"train_n":len(tr),"validation_n":len(va),
                      "xgb_rmse":root_mean_sq(y_train.iloc[va],oof[va,0]),
                      "lgbm_rmse":root_mean_sq(y_train.iloc[va],oof[va,1])})
    good=~np.isnan(oof).any(axis=1)
    meta=Ridge(alpha=1.)
    meta.fit(oof[good],y_train.iloc[np.where(good)[0]])

    a,b=model_pair()
    a.fit(train,y_train); b.fit(train,y_train)
    def predict_base(data):
        p1=a.predict(data);p2=b.predict(data)
        return np.column_stack((p1,p2)),(p1+p2)/2

    stack_test,avg_test=predict_base(test)
    stack_cal,_=predict_base(val)
    stack_pred=meta.predict(stack_test)
    cal_pred=meta.predict(stack_cal)

    # Quantile bands, calibrated against a disjoint chronological calibration
    # window. The test set is never used for fitting or calibration.
    low_model=LGBMRegressor(objective="quantile",alpha=.1,n_estimators=80,
                            num_leaves=15,learning_rate=.07,n_jobs=2,verbose=-1)
    hi_model=LGBMRegressor(objective="quantile",alpha=.9,n_estimators=80,
                           num_leaves=15,learning_rate=.07,n_jobs=2,verbose=-1)
    low_model.fit(train,y_train);hi_model.fit(train,y_train)
    low_cal=low_model.predict(val); high_cal=hi_model.predict(val)
    low_test=low_model.predict(test); high_test=hi_model.predict(test)
    rawlo=np.minimum(low_test,high_test); rawhi=np.maximum(low_test,high_test)
    callo=np.minimum(low_cal,high_cal); calhi=np.maximum(low_cal,high_cal)
    residual=np.maximum.reduce([callo-y_val.to_numpy(),y_val.to_numpy()-calhi,np.zeros(len(y_val))])
    level=min(1.,np.ceil((len(residual)+1)*.8)/len(residual))
    adjustment=float(np.quantile(residual,level,method="higher"))
    fixedlo=rawlo-adjustment
    fixedhi=rawhi+adjustment

    truth=y_test.to_numpy()
    lag1=test["historical_lag_1"].to_numpy()
    lag7=test["historical_lag_7"].to_numpy()
    sma=test["historical_mean_7"].to_numpy()
    # Simple level smoothing approximates ETS(A,N,N) and uses only historical
    # outcomes; stateless one-step filtering with alpha fixed in advance.
    ets=.5*lag1+.5*sma
    baselines={
        "naive_last_observation":lag1,
        "seasonal_naive_7_observations":lag7,
        "moving_average_7":sma,
        "simple_exponential_level_proxy":ets,
    }
    rmse={name:root_mean_sq(truth,pred) for name,pred in baselines.items()}
    rmse.update({
        "XGBoost_only":root_mean_sq(truth,stack_test[:,0]),
        "LightGBM_only":root_mean_sq(truth,stack_test[:,1]),
        "simple_average_no_Ridge":root_mean_sq(truth,avg_test),
        "Ridge_stack":root_mean_sq(truth,stack_pred),
    })
    rmse_season=rmse["seasonal_naive_7_observations"]
    best_base=min(rmse[k] for k in baselines)
    scaled={name:mase(truth,pred,y_train.to_numpy(),seasonality=1)
            for name,pred in {**baselines,
                              "XGBoost_only":stack_test[:,0],
                              "LightGBM_only":stack_test[:,1],
                              "simple_average_no_Ridge":avg_test,
                              "Ridge_stack":stack_pred}.items()}
    importance=sorted(zip(train.columns.tolist(),a.feature_importances_.tolist()),
                      key=lambda p:-p[1])[:8]
    return {
        "domain":domain,"raw_source":"synthetic CSV",
        "assumption":"one-step-ahead with latest historical target available",
        "train":len(train),"calibration":len(val),"test":len(test),
        "rmse":rmse,"mase":scaled,
        "reduction_vs_seasonal_naive_pct":round(100*(1-rmse["Ridge_stack"]/rmse_season),2) if rmse_season else None,
        "reduction_vs_best_baseline_pct":round(100*(1-rmse["Ridge_stack"]/best_base),2) if best_base else None,
        "p10_p90_raw_coverage":interval_coverage(truth,rawlo,rawhi),
        "p10_p90_calibrated_coverage":interval_coverage(truth,fixedlo,fixedhi),
        "p10_p90_raw_width":interval_mean_width(rawlo,rawhi),
        "p10_p90_calibrated_width":interval_mean_width(fixedlo,fixedhi),
        "calibration_adjustment":adjustment,
        "folds":folds,"feature_importance_top":importance,
    }

def main():
    reports=[]
    for domain,(filename,*_) in SETTINGS.items():
        source=pd.read_csv(ROOT/"data"/filename)
        X,y=assemble(source,domain)
        reports.append(diagnostics(domain,X,y))
    dest=ROOT/"results"/"leakage_safe_diagnostic.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(reports,indent=2,allow_nan=False))
    for item in reports:
        print(json.dumps({k:item[k] for k in
              ["domain","train","calibration","test","rmse","mase",
               "reduction_vs_seasonal_naive_pct","reduction_vs_best_baseline_pct",
               "p10_p90_raw_coverage","p10_p90_calibrated_coverage"]},indent=2))
    print("Saved",dest)
if __name__=="__main__":
    main()
