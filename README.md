# DemandForecasting — Multi-Domain Demand Prediction

[![CI](https://github.com/shashwatt22/DemandForecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/shashwatt22/DemandForecasting/actions/workflows/ci.yml)
[![Docker Compose smoke](https://github.com/shashwatt22/DemandForecasting/actions/workflows/docker-smoke.yml/badge.svg)](https://github.com/shashwatt22/DemandForecasting/actions/workflows/docker-smoke.yml)

A two-generation forecasting application. The **default dashboard** now serves leakage-safe v2 historical research models, trained with past-only features and out-of-fold stacking. Legacy synthetic demonstrations remain available in separately labeled pages.

**Live Demo:** [DemandForecasting — Streamlit](https://demandforecasting-shashwat.streamlit.app/)

## Tech Stack

Python · XGBoost · LightGBM · Scikit-learn · FastAPI · Streamlit · Pandas · NumPy · Plotly · Docker

## Key Features

- Demand forecasting across **3 business domains**.
- Stacked ensemble using XGBoost, LightGBM, and Ridge Regression.
- Walk-forward out-of-fold stacking for v2; legacy code uses a separate five-fold procedure.
- Horizon-specific conformal prediction intervals for v2 (nominal 80% target; unseen-test coverage unverified).
- Interactive dashboards for forecasting, model analysis, and evaluation.
- REST API with validated prediction endpoints.
- Automated testing and CI/CD support.

## Current model generations

| Route / view | Models | Data | Evaluation |
|---|---|---|---|
| Default Streamlit page, `POST /api/v2/forecast` | **V2:** past-only lagged XGBoost + LightGBM + OOF Ridge, calibrated recursively across horizons | Real historical airline passengers; real UCI online retail transactions; synthetic payment series | Last 20% sealed; **no unseen-test performance claim** |
| Legacy demo pages, `/forecast/*` | Original saved models with historical feature design | Synthetic airline/e-commerce/payments | Known target-leakage risks; should **not** be represented as reliable real forecasting performance |

The original 95.3% / 78.7% / 87.6% reduction claims were computed on synthetic data with leakage risks and are **not** performance claims for v2.

### Public development baseline comparison

Rolling origins on the first 80% of each real time series, with the last 20% intentionally withheld:

| Dataset | Seasonality | Seasonal naive RMSE | Fitted ETS RMSE |
|---|---|---:|---:|
| Historical international airline passengers | Same month previous year (12 months) | 38.04 | 18.33 |
| UCI Online Retail daily units | Same weekday previous week (7 days) | 7,084.34 | 6,559.78 |

The UCI source is [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail); airline monthly counts are the public [AirPassengers CSV](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv). The airline data measure passenger totals, **not flight bookings**. Do not compare their RMSE magnitudes across datasets.

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

> The default dashboard now uses **v2** artifacts in `models/v2`; legacy pages retain the original artifacts in `models/saved`. Both are research demonstrations, not real-time airline or payment production systems.

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

## Retrain V2 and run tests

```bash
python -m pip install -r requirements.txt
python -m pip install statsmodels openpyxl
python -m src.models.leakage_safe_v2
pytest -q
```

The [V2 training workflow](https://github.com/shashwatt22/DemandForecasting/actions/workflows/train-v2.yml) versions the trained artifacts. This training uses 70% chronological training data and the following 10% for horizon-specific calibration. The final **20% is sealed**; it is neither trained on nor used for tuning. The [public-data baseline workflow](https://github.com/shashwatt22/DemandForecasting/actions/workflows/public-backtests.yml) evaluates only development periods.

**Next:** add production-time external regressors that are genuinely observable at forecast creation, independently verify sealed-test coverage once after freezing the model, evaluate real payment datasets if suitable lawful data become available, and quantify uncertainty under abrupt regime changes.

## Testing

```bash
pytest -q
```
