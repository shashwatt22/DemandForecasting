# DemandForecasting — Multi-Domain Demand Prediction

A machine learning platform for forecasting airline bookings, e-commerce demand, and payment transactions using stacked ensemble learning.

**Live Demo:** [DemandForecasting — Streamlit](https://demandforecasting-shashwat.streamlit.app/)

## Tech Stack

Python · XGBoost · LightGBM · Scikit-learn · FastAPI · Streamlit · Pandas · NumPy · Plotly · Docker

## Key Features

- Demand forecasting across **3 business domains**.
- Stacked ensemble using XGBoost, LightGBM, and Ridge Regression.
- 5-fold time-series cross-validation.
- P10–P90 prediction intervals for uncertainty estimation.
- Interactive dashboards for forecasting, model analysis, and evaluation.
- REST API with validated prediction endpoints.
- Automated testing and CI/CD support.

## Model Performance

| Domain | RMSE | R² | RMSE Reduction |
|---|---:|---:|---:|
| Airline Bookings | 1.24 | 0.998 | 95.3% |
| E-commerce | 24.1 | 0.953 | 78.7% |
| Payment Transactions | 51.0 | 0.975 | 87.6% |

*Reported performance on held-out synthetic datasets against naive forecasting baselines.*

## Architecture

Data → Feature Engineering → XGBoost + LightGBM → Ridge Stacking → Forecast + Prediction Intervals → Dashboard/API

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Testing

```bash
pytest -q
```
