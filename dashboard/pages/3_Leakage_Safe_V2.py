"""Leakage-safe v2 research forecasts; never plots sealed test target values."""
from __future__ import annotations
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))
from src.models.leakage_safe_v2 import PARAMS, calibrated_forecast

st.title("Leakage-safe multi-step forecasting")
st.caption(
    "V2 research models use only lagged observations and calendar features. "
    "The final 20% of each dataset is sealed and never displayed or evaluated here. "
    "Source series include historical airline passengers, UCI retail demand, "
    "and synthetic payment transactions."
)
domain=st.selectbox("Domain",list(PARAMS),format_func=lambda x:{
    "airline":"Historical airline passengers (real)",
    "ecommerce":"UCI retail quantity (real)",
    "payment":"Hourly payment transactions (synthetic)",
}[x])
path=PROJECT_ROOT/"models"/"v2"/f"{domain}.joblib"
if not path.is_file():
    st.warning("V2 model artifacts are not available in this deployment. "
               "Run the v2 training workflow; the legacy demo is unchanged.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model(location:str):
    return joblib.load(location)

bundle=load_model(str(path))
horizon=st.slider("Forecast horizon",1,int(bundle["max_horizon"]),min(3,int(bundle["max_horizon"])))
selector=st.selectbox("Forecast method",["Stacked XGBoost + LightGBM + Ridge",
                                          "Calendar seasonal naive","Moving average"])
base=calibrated_forecast(bundle,horizon)
history=np.asarray(bundle["history"],dtype=float)
period=int(bundle["period"])
if selector=="Calendar seasonal naive":
    projected=np.array([history[-period+(k%period)] for k in range(horizon)])
elif selector=="Moving average":
    projected=np.full(horizon,float(np.mean(history[-period:])))
else:
    projected=np.asarray(base["forecast"])

last_time=pd.Timestamp(bundle["last_time"])
dates=pd.date_range(last_time,periods=horizon+1,freq=bundle["freq"])[1:]
plot_history=pd.date_range(
    end=last_time,periods=min(len(history),max(3*period,24)),freq=bundle["freq"]
)
y_history=history[-len(plot_history):]
fig=go.Figure()
fig.add_trace(go.Scatter(x=plot_history,y=y_history,mode="lines",
                         name="Historical development observations"))
fig.add_trace(go.Scatter(x=dates,y=projected,mode="lines+markers",
                         name="Selected forecast"))
if selector=="Stacked XGBoost + LightGBM + Ridge":
    lo=np.asarray(base["lower_bound"])
    hi=np.asarray(base["upper_bound"])
    fig.add_trace(go.Scatter(x=dates,y=hi,mode="lines",
                             name="Upper bound",line=dict(width=0)))
    fig.add_trace(go.Scatter(x=dates,y=lo,mode="lines",
                             name="Calibrated interval",fill="tonexty",
                             line=dict(width=0),opacity=0.3))
fig.update_layout(height=440,xaxis_title="Historical time",
                  yaxis_title="Observed quantity or passenger count")
st.plotly_chart(fig,use_container_width=True)
st.metric("Development calibration origins",bundle["calibration_origins"])
st.caption(
    "Intervals have a nominal 80% target with horizon-specific development "
    "calibration; no coverage guarantee for the sealed test period. "
    "Historical research origins are not current business predictions."
)
st.dataframe(pd.DataFrame({"Timestamp":dates,"Forecast":projected}),hide_index=True)
