"""Development-only rolling origin study of public demand datasets.

All model fitting and calibration at each origin uses observations strictly
before that origin. The final 20% of each series is NEVER scored or accessed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from src.models.leakage_safe_v2 import to_supervised, base_models, recursive_forecast
from src.evaluation.public_backtesting import (air_passengers, retail_demand,
                                                forecast_baselines, rmse)

ROOT=Path(__file__).resolve().parents[2]

def fit_history(history:pd.Series,season:int,horizon:int)->dict:
    X,y=to_supervised(history,season)
    oof=np.full((len(X),2),np.nan)
    for a,b in TimeSeriesSplit(n_splits=3).split(X):
        m1,m2=base_models()
        m1.fit(X.iloc[a],y.iloc[a]);m2.fit(X.iloc[a],y.iloc[a])
        oof[b,0]=m1.predict(X.iloc[b])
        oof[b,1]=m2.predict(X.iloc[b])
    valid=np.isfinite(oof).all(axis=1)
    meta=Ridge(alpha=1.).fit(oof[valid],y.iloc[np.flatnonzero(valid)])
    xgb,lgb=base_models()
    xgb.fit(X,y);lgb.fit(X,y)
    return {"period":season,"freq":pd.infer_freq(history.index),
            "max_horizon":horizon,"xgb":xgb,"lgb":lgb,"ridge":meta}

def walk(series:pd.Series,season:int,horizon:int,origins:int=6)->dict:
    n=len(series); dev=series.iloc[:int(.8*n)]
    earliest=max(4*season,int(.55*n))
    candidates=list(range(earliest,len(dev)-horizon+1,horizon))
    if len(candidates)<3: raise ValueError("Insufficient development data")
    # Calibration origins first, evaluation origins after. Both inside dev.
    selected=[candidates[i] for i in np.linspace(0,len(candidates)-1,
             min(origins,len(candidates)),dtype=int)]
    calibration=selected[:max(2,len(selected)//3)]
    evaluation=selected[len(calibration):]
    predictions={}
    for origin in selected:
        prior=dev.iloc[:origin]
        bundle=fit_history(prior,season,horizon)
        vals=prior.to_numpy(dtype=float).tolist()
        models={}
        for key in ("ridge","xgb","lgb"):
            models[key]=np.asarray(recursive_forecast(
                bundle,vals,prior.index[-1],horizon,estimator=key))
        baseline=forecast_baselines(prior,horizon,season)
        truth=dev.iloc[origin:origin+horizon].to_numpy(dtype=float)
        scale=float(np.mean(np.abs(np.diff(prior.to_numpy(dtype=float)))))
        predictions[origin]=(truth,{**baseline,**models},scale)
    models=list(next(iter(predictions.values()))[1])
    # Residual quantiles from disjoint past calibration origins ONLY.
    residuals={key:np.concatenate([np.abs(predictions[i][0]-predictions[i][1][key])
                                    for i in calibration]) for key in ("ridge","xgb","lgb")}
    radii={key:float(np.quantile(err,min(1.,np.ceil((len(err)+1)*.8)/len(err)),
                                  method="higher")) for key,err in residuals.items()}
    rows=[]
    for origin in evaluation:
        truth,pred,scale=predictions[origin]
        record={"origin":str(dev.index[origin]),"samples":len(truth)}
        for key in models:
            record[key+"_rmse"]=rmse(truth,pred[key])
            record[key+"_mase"]=float(np.mean(np.abs(truth-pred[key]))/scale) if scale else None
            if key in radii:
                record[key+"_coverage"]=float(np.mean(
                    (truth>=pred[key]-radii[key])&(truth<=pred[key]+radii[key])))
        rows.append(record)
    return {"total_rows":n,"sealed_rows":n-len(dev),"sealed_evaluated":False,
            "calibration_origins":len(calibration),"evaluation_origins":len(evaluation),
            "interval_nominal":.8,"models":{
                key:{"rmse":round(float(np.mean([row[key+"_rmse"] for row in rows])),3),
                     "mase":round(float(np.mean([row[key+"_mase"] for row in rows])),3),
                     "coverage":round(float(np.mean([row[key+"_coverage"] for row in rows])),3)
                                 if key in radii else None}
                for key in models},"per_origin":rows}

def main():
    output={}
    for name,(series,season,horizon) in {
       "AirPassengers":(air_passengers(),12,3),
       "UCI_Online_Retail":(retail_demand(),7,7)
    }.items():
        output[name]=walk(series,season,horizon)
        print(name,json.dumps(output[name]["models"]))
    out=ROOT/"results"/"public_stack_development.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output,indent=2))
if __name__=="__main__":main()
