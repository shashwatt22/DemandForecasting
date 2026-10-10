"""Streamlit dashboard for multi-domain demand forecasting."""

import streamlit as st

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/3_Leakage_Safe_V2.py", title="Leakage-safe forecasting",
            icon=":material/trending_up:", default=True),
    st.Page("pages/0_Forecasting.py", title="Legacy forecasting demo",
            icon=":material/history:"),
    st.Page("pages/1_Model_Internals.py", title="Model Internals",
            icon=":material/architecture:"),
    st.Page("pages/2_Evaluation.py",      title="Evaluation (legacy)",
            icon=":material/verified:"),
])

pg.run()
