# RT-Electricity-Forecast

**Real-Time Electricity Price Forecasting via Two-Stage Load Prediction with Simulated Streaming Infrastructure**

## Overview

This system predicts **PJME electricity load (MW)** one hour ahead using a LightGBM gradient-boosted tree model enhanced with **cross-regional features** from 5 neighboring PJM zones, then converts the load forecast into a **price estimate ($/MWh)** via a deterministic pricing engine.

### Two-Stage Architecture

```
LoadEvent(t) → [Stage 1: LightGBM Load Forecaster] → Load_hat(t+1)
                                                          ↓
              [Stage 2: Deterministic Pricing Engine] → Price_hat(t+1)
              P(L) = 15.0 + 0.005·L + 0.000002·L²
```

**Critical invariant:** The ML model predicts **Load**, never Price. Price is always derived downstream through a deterministic business logic layer.

### Key Features

- **65-feature pipeline** from 6 PJM regional load series (PJME + 5 auxiliary regions)
- **Walk-forward validation** with 5 expanding-window chronological folds (no data leakage)
- **Asyncio streaming simulation** with token-bucket rate limiting and logical time replay
- **FastAPI service** with REST + WebSocket endpoints
- **Streamlit dashboard** with real-time Plotly visualizations
- **Comprehensive evaluation** with MAE, RMSE, MAPE metrics for both Load and Price

## Dataset

- **Source:** `data/pjm_hourly_est.csv` — consolidated PJM Interconnection hourly load data
- **Regions used:** PJME (target), PJMW, AEP, DAYTON, DOM, DUQ
- **Complete rows:** ~116,187 (2005-12-31 to 2018-01-02)
- **Train/Stream split:** 80/20 chronological (92,950 / 23,237 rows)

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 2. Prepare Data

```bash
# Copy dataset to expected location
mkdir -p data
cp Dataset/pjm_hourly_est.csv data/
```

### 3. Train Model

```bash
make train
# or: python -m scripts.train
```

### 4. Run Streaming Simulation

```bash
make stream
# or: python -m scripts.stream
```

This launches:
- FastAPI server at `http://localhost:8000`
- Streaming pipeline processing events

### 5. Launch Dashboard (separate terminal)

```bash
streamlit run dashboard/app.py
```

Dashboard at `http://localhost:8501`

### 6. Run Batch Evaluation

```bash
make evaluate
# or: python -m scripts.evaluate
```

### 7. Run Tests

```bash
make test
# or: python -m pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Single-shot inference from load sequence |
| `/metrics` | GET | Current evaluation metrics |
| `/health` | GET | System health check |
| `/stream/status` | GET | Streaming simulation state |
| `/ws/stream` | WS | Real-time forecast streaming |

## Project Structure

```
├── config/          # Pydantic BaseSettings configuration
├── core/            # Data loading, features, pricing, metrics
├── models/          # LightGBM trainer, forecaster, baselines
├── streaming/       # Rate limiter, replay engine, feature buffer, pipeline
├── api/             # FastAPI app, routes, WebSocket, schemas
├── dashboard/       # Streamlit monitoring UI
├── scripts/         # CLI entrypoints (train, stream, evaluate)
├── tests/           # Unit and integration tests
├── artifacts/       # Generated models and reports
└── data/            # Dataset (pjm_hourly_est.csv)
```

## Configuration

All configuration is managed via `config/settings.py` with environment variable overrides (prefix: `RTEF_`). See `.env.example` for available options.

## License

This project is for educational and portfolio purposes.
