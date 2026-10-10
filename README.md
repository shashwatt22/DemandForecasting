# DemandForecasting — Multi-Domain Demand Prediction

[![CI](https://github.com/shashwatt22/DemandForecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/shashwatt22/DemandForecasting/actions/workflows/ci.yml)
[![Docker Compose smoke](https://github.com/shashwatt22/DemandForecasting/actions/workflows/docker-smoke.yml/badge.svg)](https://github.com/shashwatt22/DemandForecasting/actions/workflows/docker-smoke.yml)

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

*Original synthetic-data results; the original feature engineering has target-leakage risks and these headline improvements should not be interpreted as production forecast quality. For leakage-aware research metrics, see [evaluation results](docs/LEAKAGE_SAFE_RESULTS.md).*

## Architecture

\`\`\`mermaid
flowchart LR
    A[Synthetic observations] --> B[Historical features]
    B --> C[XGBoost]
    B --> D[LightGBM]
    C --> E[Ridge stack]
    D --> E
    E --> F[Prediction and P10-P90 intervals]
    F --> G[Streamlit dashboard and FastAPI]
\`\`\`

> The published live app serves existing saved models. The [leakage-aware research benchmark](docs/LEAKAGE_SAFE_RESULTS.md) runs a separate train/calibration/test workflow; it has not replaced those saved models.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Run API + Dashboard with Docker Compose

\`\`\`bash
docker compose up --build
\`\`\`

- Dashboard: http://localhost:8501
- API: http://localhost:8000
- Compose smoke test: [GitHub Actions](https://github.com/shashwatt22/DemandForecasting/actions/workflows/docker-smoke.yml)

## Evaluation

Leakage-safe research results (synthetic test periods, **not** live models):

| Domain | Ridge RMSE | Reduction vs 7-observation seasonal naive | MASE | Raw P10–P90 coverage | After conformal adjustment |
|---|---:|---:|---:|---:|---:|
| Airline | 13.604 | 66.88% | 0.344 | 54.01% | 75.95% |
| E-commerce | 30.288 | 69.28% | 0.241 | 71.65% | 74.72% |
| Payments | 75.160 | 78.54% | 0.303 | 68.06% | 71.74% |

Important: “7 observations” is not necessarily seven calendar days for grouped data. See [full audit and limitations](docs/LEAKAGE_SAFE_RESULTS.md). The Ridge stack did not beat standalone LightGBM for every domain.

## Next steps

Replace the legacy deployed models with leak-safe retrained models only after defining production-time feature availability, validating time-specific seasonality and re-running container and API compatibility tests. Investigate payment interval undercoverage.

## Testing

```bash
pytest -q
```
