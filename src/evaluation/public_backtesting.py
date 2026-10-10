"""Rolling-origin public-data forecasting evaluation. NEVER evaluates sealed 20%.

Datasets:
- International AirPassengers monthly totals (historical real passenger counts).
- UCI Online Retail (2010-2011 real transaction quantity, aggregated daily).

These are related real demand data, not the same synthetic booking/payment domains.
"""
from __future__ import annotations
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT=Path(__file__).resolve().parents[2]
AIR_URL="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
RETAIL_URL="https://archive.ics.uci.edu/static/public/352/online+retail.zip"

def fetch_bytes(url: str)->bytes:
    request=Request(url,headers={"User-Agent":"DemandForecasting-research/1.0"})
    with urlopen(request,timeout=60) as response:
        return response.read()

def air_passengers()->pd.Series:
    data=pd.read_csv(BytesIO(fetch_bytes(AIR_URL)))
    months=pd.to_datetime(data["Month"],format="%Y-%m")
    s=pd.Series(data["Passengers"].to_numpy(dtype=float),index=months,name="passengers")
    return s.asfreq("MS")

def retail_demand()->pd.Series:
    with ZipFile(BytesIO(fetch_bytes(RETAIL_URL))) as z:
        book=next((n for n in z.namelist() if n.lower().endswith(".xlsx")),None)
        if book is None:
            raise ValueError("Official UCI archive has no XLSX")
        df=pd.read_excel(BytesIO(z.read(book)),engine="openpyxl")
    # Cancellations/returns and negative quantities are not demand.
    invoice=df["InvoiceNo"].astype(str)
    quantity=pd.to_numeric(df["Quantity"],errors="coerce")
    dates=pd.to_datetime(df["InvoiceDate"],errors="coerce")
    valid=(~invoice.str.upper().str.startswith("C")) & quantity.gt(0) & dates.notna()
    data=pd.DataFrame({"date":dates.loc[valid].dt.floor("D"),"qty":quantity.loc[valid]})
    daily=data.groupby("date")["qty"].sum().sort_index()
    # Missing trading days explicitly zero, not imputed from the future.
    return daily.asfreq("D",fill_value=0).rename("retail_daily_quantity")

def rmse(actual,predicted)->float:
    y=np.asarray(actual,dtype=float)
    p=np.asarray(predicted,dtype=float)
    return float(np.sqrt(np.mean((y-p)**2)))

def forecast_baselines(history:pd.Series,horizon:int,season:int)->dict[str,np.ndarray]:
    y=history.to_numpy(dtype=float)
    if len(y)<2*season:
        raise ValueError("Insufficient history for seasonal evaluation")
    seasonal=np.array([y[len(y)-season+(i%season)] for i in range(horizon)])
    moving=np.full(horizon,float(y[-season:].mean()))
    fitted=ExponentialSmoothing(
        y,trend="add",seasonal="add",seasonal_periods=season,
        initialization_method="estimated",
    ).fit(optimized=True)
    ets=np.asarray(fitted.forecast(horizon),dtype=float)
    return {"seasonal_naive":seasonal,"moving_average":moving,"ETS":ets}

def evaluate_development(series:pd.Series,season:int,horizon:int)->dict:
    """Expanding rolling origins entirely before the last 20% sealed test."""
    n=len(series)
    sealed_start=int(n*.8)
    development=series.iloc[:sealed_start]
    min_train=max(season*3,int(n*.55))
    possible=list(range(min_train,len(development)-horizon+1,horizon))
    if not possible:
        raise ValueError("Insufficient development period")
    records=[]
    for origin in possible:
        history=development.iloc[:origin]
        truth=development.iloc[origin:origin+horizon].to_numpy()
        baseline=forecast_baselines(history,horizon,season)
        records.append({"origin":str(history.index[-1].date()),
                        "horizon":horizon,
                        **{f"{k}_rmse":rmse(truth,p) for k,p in baseline.items()}})
    models=("seasonal_naive","moving_average","ETS")
    return {"total_rows":n,"development_rows":sealed_start,
            "sealed_test_rows":n-sealed_start,
            "sealed_evaluated":False,
            "rolling_origins":len(records),
            "seasonal_period":season,
            "horizon":horizon,
            "mean_dev_rmse":{k:round(float(np.mean([row[f"{k}_rmse"] for row in records])),4)
                             for k in models},
            "per_origin":records}

def main():
    datasets={"air_passengers":(air_passengers(),12,3),
              "uci_online_retail":(retail_demand(),7,7)}
    results={}
    for name,(series,season,horizon) in datasets.items():
        result=evaluate_development(series,season,horizon)
        results[name]=result
        print(name,"sealed test untouched",result["mean_dev_rmse"])
    path=ROOT/"results"/"public_development_backtest.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
