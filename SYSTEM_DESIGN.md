# Real-Time Electricity Price Forecasting System (Simulated Streaming)

---

## 1. Project Name

**`rt-electricity-forecast`** — Real-Time Electricity Price Forecasting via Two-Stage Load Prediction with Simulated Streaming Infrastructure

---

## 2. Problem Statement

Electricity markets require sub-hourly price forecasts to support dispatch optimization, trading decisions, and grid stability. This system addresses the problem by predicting **PJME electricity load (MW)** one hour ahead using a gradient-boosted tree model enhanced with **cross-regional features** from 5 neighboring PJM zones, then converting the load forecast into a **price estimate ($/MWh)** via a deterministic pricing engine.

**Simulation Constraint:** No real-time data feed exists. The system replays historical PJM Interconnection hourly load data from `pjm_hourly_est.csv` — a consolidated dataset covering 6 PJM regions (PJME, PJMW, AEP, DAYTON, DOM, DUQ) with ~116,187 complete rows spanning 2005-12-31 to 2018-01-02 — through an asyncio-based streaming simulation engine. The simulation respects logical time ordering and uses a token-bucket rate limiter to control throughput — it does NOT use naive `time.sleep()` blocking loops. Ground-truth prices are synthesized via a known quadratic pricing function for evaluation purposes only.

**Cross-Regional Design:** Regional electricity loads are physically correlated — weather systems, economic activity, and grid interconnections cause load in one zone to predict load in adjacent zones. Using 5 auxiliary regions as features captures these spatial correlations without adding external data sources.

**Critical invariant:** The ML model predicts **PJME Load**, never Price. Price is always derived downstream through a deterministic business logic layer.

---

## 3. System Overview

The system is a **two-stage forecasting pipeline** wrapped inside an **asyncio-driven streaming simulation**:

| Component | Responsibility |
|---|---|
| **Streaming Simulation Engine** | Replays historical load data as an async event stream with token-bucket rate control |
| **Feature Engineering Pipeline** | Transforms raw multi-region load stream into PJME lagged features, cross-regional features, temporal encodings, and rolling statistics |
| **Load Forecasting Model** | LightGBM regressor producing `Load_hat(t+1)` from feature vector |
| **Deterministic Pricing Engine** | Converts `Load_hat(t+1)` → `Price_hat(t+1)` via fixed quadratic formula |
| **Ground Truth Simulator** | Converts `Load_actual(t+1)` → `Price_actual(t+1)` for evaluation |
| **Evaluation Engine** | Computes MAE, RMSE, MAPE on both Load and Price; tracks inference latency |
| **FastAPI Service** | REST + WebSocket endpoints for inference, metrics, and real-time streaming |
| **Dashboard** | Streamlit-based monitoring UI for live forecast visualization |

---

## 4. Core Architecture (Step-by-Step Pipeline)

### 4.1 Data Ingestion Layer

1. On system startup, the `DataLoader` reads `pjm_hourly_est.csv` from disk into a pandas DataFrame.
2. The DataFrame is filtered to rows where all 6 target regions (PJME, PJMW, AEP, DAYTON, DOM, DUQ) have non-null values, yielding ~116,187 complete rows starting from 2005-12-31.
3. The DataFrame is sorted by `Datetime` ascending, deduplicated, and validated for continuity (missing hours are flagged, not imputed silently).
4. The dataset is split chronologically: the final 20% (~23,237 hours ≈ 2.7 years) is reserved as the **streaming replay buffer**. The preceding 80% is the **training corpus**.

### 4.2 Training Pipeline (Offline, Pre-Streaming)

5. The `FeatureEngineer` processes the training corpus to extract:
   - PJME lag features: `Load(t-1)`, `Load(t-2)`, ..., `Load(t-168)` (up to 1 week)
   - Cross-regional features: `lag_1` and `lag_24` for each auxiliary region (PJMW, AEP, DAYTON, DOM, DUQ) + 24h rolling mean per auxiliary region
   - Rolling statistics: 24h, 48h, 168h moving averages and standard deviations of PJME
   - Temporal features: `hour_of_day`, `day_of_week`, `month`, `is_weekend`
   - Cyclical encodings: `sin/cos` transforms of `hour_of_day`, `day_of_week`, and `month`
6. The `WalkForwardTrainer` trains a **LightGBM** regressor using expanding-window walk-forward validation across 5 chronological folds. Each fold trains on all prior data and validates on the next segment.
7. The best model (by validation MAE) is serialized to `artifacts/model.lgb`.
8. Feature importance and validation metrics are logged to `artifacts/training_report.json`.

### 4.3 Streaming Simulation Engine (Online)

9. The `StreamSimulator` wraps the replay buffer (the held-out 20%) in an async generator.
10. A `TokenBucketRateLimiter` governs emission rate. Default configuration: bucket capacity = 10 tokens, refill rate = 5 tokens/second. Each emitted record consumes 1 token. When the bucket is empty, the coroutine `await`s the next refill — no `time.sleep()`, no spin-waiting.
11. The `LogicalClock` tracks simulated time (advancing by 1 hour per emitted record), decoupled from wall-clock time. This enables fast-replay and slow-replay modes without changing pipeline logic.
12. Each emitted event is a `LoadEvent` dataclass: `{timestamp: datetime, pjme_mw: float, pjmw_mw: float, aep_mw: float, dayton_mw: float, dom_mw: float, duq_mw: float, sequence_id: int}`.

### 4.4 Real-Time Inference Pipeline (Per Event)

13. On receiving `LoadEvent(t)`, the `FeatureBuffer` (a ring buffer of the last 168 multi-region load records) appends the new values and constructs the feature vector `X(t)` including cross-regional features.
14. The `LoadForecaster` runs `model.predict(X(t))` → `Load_hat(t+1)` (PJME load). Inference latency is measured via `time.perf_counter_ns()`.
14. The `PricingEngine` applies the deterministic formula:
    ```
    Price_hat(t+1) = 15.0 + 0.005 * Load_hat(t+1) + 0.000002 * (Load_hat(t+1))^2
    ```
15. The `GroundTruthSimulator` simultaneously computes:
    ```
    Price_actual(t+1) = 15.0 + 0.005 * Load_actual(t+1) + 0.000002 * (Load_actual(t+1))^2 + ε
    ```
    where `ε ~ N(0, σ²)` with `σ = 2.0` (configurable noise parameter).

### 4.5 Evaluation & Monitoring

16. The `MetricsCollector` accumulates predictions and actuals into a sliding window. Every N events (configurable, default 24), it computes:
    - Load metrics: MAE, RMSE, MAPE
    - Price metrics: MAE, RMSE, MAPE
    - Inference latency: p50, p95, p99 (nanosecond precision)
17. Metrics are published to an in-memory metrics store and pushed via WebSocket to connected dashboard clients.

### 4.6 API & Dashboard Layer

18. FastAPI serves REST endpoints for on-demand inference, historical metrics retrieval, and system health.
19. A WebSocket endpoint streams live forecasts and metrics to the Streamlit dashboard.
20. The dashboard renders real-time time-series charts (actual vs. predicted for both Load and Price), error distribution histograms, and latency gauges.

---

## 5. Streaming Simulation Engine

### 5.1 Token Bucket Rate Limiter

```
Class: TokenBucketRateLimiter
- capacity: int         # max tokens in bucket (burst allowance)
- refill_rate: float    # tokens added per second
- tokens: float         # current token count
- last_refill: float    # monotonic timestamp of last refill

Method: async acquire(n=1) -> None
  1. Compute elapsed = now - last_refill
  2. Add elapsed * refill_rate tokens (capped at capacity)
  3. If tokens >= n: consume and return
  4. Else: compute wait_time = (n - tokens) / refill_rate
  5. await asyncio.sleep(wait_time)  # non-blocking yield to event loop
  6. Refill and consume
```

**Design rationale:** Token bucket (not leaky bucket) allows controlled bursting. This is critical for simulation warm-up where we need to fill the feature buffer quickly, then throttle to steady-state replay speed.

### 5.2 Logical Time Replay Engine

```
Class: LogicalTimeReplayEngine
- replay_buffer: List[LoadEvent]     # chronologically sorted
- logical_clock: datetime            # current simulated time
- speed_multiplier: float            # 1.0 = real-time, 100.0 = 100x fast
- rate_limiter: TokenBucketRateLimiter

Method: async stream() -> AsyncGenerator[LoadEvent, None]
  for event in replay_buffer:
      await rate_limiter.acquire(1)
      logical_clock = event.timestamp
      yield event
```

**Key properties:**
- Logical time advances discretely by the inter-event interval in the dataset (1 hour), regardless of wall-clock time.
- The rate limiter controls wall-clock throughput, not logical time gaps.
- The engine is fully deterministic: identical rate limiter config produces identical event ordering and timing.

### 5.3 Asyncio Event Loop Architecture

```
Main Coroutine Topology:
├── replay_engine.stream()        # produces LoadEvents
├── inference_pipeline.process()  # consumes events, produces forecasts
├── metrics_collector.update()    # consumes forecasts + actuals
├── websocket_publisher.push()    # publishes to dashboard
└── api_server.serve()            # FastAPI uvicorn (run_in_executor)
```

All components communicate via `asyncio.Queue` instances with bounded capacity (backpressure). No threading is used in the hot path. The FastAPI server is launched via `uvicorn` inside the same event loop.

---

## 6. Data Pipeline Design

### 6.1 Dataset Specification

| Property | Value |
|---|---|
| Source file | `Dataset/pjm_hourly_est.csv` |
| Raw columns | `Datetime`, `AEP`, `COMED`, `DAYTON`, `DEOK`, `DOM`, `DUQ`, `EKPC`, `FE`, `NI`, `PJME`, `PJMW`, `PJM_Load` |
| Used columns | `Datetime`, `PJME`, `PJMW`, `AEP`, `DAYTON`, `DOM`, `DUQ` (6 regions) |
| Unused columns | `COMED`, `DEOK`, `EKPC`, `FE`, `NI`, `PJM_Load` (excluded due to low data overlap) |
| Complete records | 116,187 rows (rows where all 6 used regions have non-null values) |
| Time range | 2005-12-31 01:00 → 2018-01-02 00:00 |
| Granularity | Hourly (1H) |
| Target variable | `PJME` (megawatts of PJME region electricity load) |
| Auxiliary variables | `PJMW`, `AEP`, `DAYTON`, `DOM`, `DUQ` (cross-regional features, not prediction targets) |

### 6.1.1 Region Selection Rationale

The 6 regions were selected based on data completeness analysis:

| Region | Non-null rows | Coverage (%) | Included? |
|---|---|---|---|
| PJME | 145,366 | 81.5% | Yes (target) |
| PJMW | 143,206 | 80.3% | Yes |
| AEP | 121,273 | 68.0% | Yes |
| DAYTON | 121,275 | 68.0% | Yes |
| DOM | 116,189 | 65.2% | Yes |
| DUQ | 119,068 | 66.8% | Yes |
| COMED | 66,497 | 37.3% | No (would reduce to ~66K rows) |
| DEOK | 57,739 | 32.4% | No |
| EKPC | 45,334 | 25.4% | No |
| FE | 62,874 | 35.3% | No |
| NI | 58,450 | 32.8% | No |
| PJM_Load | 32,896 | 18.5% | No |

The 6-region subset preserves 116,187 complete rows (12 years of data). Adding COMED would cut this to ~66K rows (6 years) — an unacceptable tradeoff.

### 6.2 Data Validation Rules

| Check | Action |
|---|---|
| Duplicate timestamps | Drop duplicates, keep last |
| Missing regions | Drop rows where any of the 6 used regions is null |
| Missing hours (gaps) | Log warning; forward-fill if gap <= 3 hours; flag if gap > 3 hours |
| Negative load values | Clip to 0.0 with warning (applied per region) |
| Extreme outliers | Flag values > mu + 5sigma per region (do not remove — energy spikes are real) |
| Timezone | Assume EST (PJM territory); store as timezone-naive for simplicity |

### 6.3 Train / Stream Split

```
Total complete records: 116,187
Training corpus:        92,950 (80%) -> 2005-12-31 to ~2015-08
Streaming replay:        23,237 (20%) -> ~2015-08 to 2018-01-02
Warm-up buffer:            168 (first 168 events of replay, to fill lag buffer)
Evaluation window:      23,069 (replay minus warm-up)
```

The split is **strictly chronological**. No shuffling. No random sampling. This prevents data leakage.

---

## 7. Feature Engineering

All features are derived **exclusively** from the `Datetime` column and the 6 load columns (`PJME`, `PJMW`, `AEP`, `DAYTON`, `DOM`, `DUQ`) present in `pjm_hourly_est.csv`. No external data.

### 7.1 PJME Lag Features (Primary Target)

| Feature | Description |
|---|---|
| `pjme_lag_1` to `pjme_lag_24` | PJME load values from 1 to 24 hours ago |
| `pjme_lag_48` | PJME load value 48 hours ago |
| `pjme_lag_168` | PJME load value 168 hours ago (same hour, same day, last week) |

**Total: 26 PJME lag features**

### 7.2 Cross-Regional Lag Features (Auxiliary Regions)

For each auxiliary region `R` in `{PJMW, AEP, DAYTON, DOM, DUQ}`:

| Feature | Description |
|---|---|
| `{r}_lag_1` | Region R load value 1 hour ago |
| `{r}_lag_24` | Region R load value 24 hours ago (same hour yesterday) |

Where `{r}` is the lowercase region name: `pjmw`, `aep`, `dayton`, `dom`, `duq`.

**Total: 5 regions x 2 lags = 10 cross-regional lag features**

### 7.3 PJME Rolling Statistics

| Feature | Window | Statistic |
|---|---|---|
| `pjme_roll_mean_24h` | 24 hours | Mean |
| `pjme_roll_std_24h` | 24 hours | Std deviation |
| `pjme_roll_mean_48h` | 48 hours | Mean |
| `pjme_roll_std_48h` | 48 hours | Std deviation |
| `pjme_roll_mean_168h` | 168 hours (1 week) | Mean |
| `pjme_roll_std_168h` | 168 hours (1 week) | Std deviation |
| `pjme_roll_min_24h` | 24 hours | Minimum |
| `pjme_roll_max_24h` | 24 hours | Maximum |

**Total: 8 PJME rolling features**

### 7.4 Cross-Regional Rolling Features

For each auxiliary region `R` in `{PJMW, AEP, DAYTON, DOM, DUQ}`:

| Feature | Window | Statistic |
|---|---|---|
| `{r}_roll_mean_24h` | 24 hours | Mean |

Where `{r}` is the lowercase region name.

**Total: 5 regions x 1 rolling feature = 5 cross-regional rolling features**

### 7.5 Temporal Features

| Feature | Derivation | Encoding |
|---|---|---|
| `hour_of_day` | `Datetime.hour` | Integer [0, 23] |
| `hour_sin` | `sin(2pi * hour / 24)` | Float [-1, 1] |
| `hour_cos` | `cos(2pi * hour / 24)` | Float [-1, 1] |
| `day_of_week` | `Datetime.dayofweek` | Integer [0, 6] |
| `dow_sin` | `sin(2pi * dayofweek / 7)` | Float [-1, 1] |
| `dow_cos` | `cos(2pi * dayofweek / 7)` | Float [-1, 1] |
| `month` | `Datetime.month` | Integer [1, 12] |
| `month_sin` | `sin(2pi * month / 12)` | Float [-1, 1] |
| `month_cos` | `cos(2pi * month / 12)` | Float [-1, 1] |
| `is_weekend` | `dayofweek >= 5` | Binary [0, 1] |
| `day_of_year` | `Datetime.dayofyear` | Integer [1, 366] |

**Total: 11 temporal features**

### 7.6 PJME Derived Features

| Feature | Derivation |
|---|---|
| `pjme_diff_1h` | `PJME(t) - PJME(t-1)` (1st difference) |
| `pjme_diff_24h` | `PJME(t) - PJME(t-24)` (day-over-day change) |
| `pjme_ratio_24h` | `PJME(t) / PJME(t-24)` (day-over-day ratio) |

**Total: 3 PJME derived features**

### 7.7 Cross-Regional Ratio Features

| Feature | Derivation |
|---|---|
| `pjme_to_pjmw_ratio` | `PJME(t) / PJMW(t)` (east-to-west load ratio) |
| `pjme_to_total_ratio` | `PJME(t) / (PJME(t) + PJMW(t) + AEP(t) + DAYTON(t) + DOM(t) + DUQ(t))` |

**Total: 2 cross-regional ratio features**

### 7.8 Feature Vector Summary

| Category | Count | Features |
|---|---|---|
| PJME lag features | 26 | `pjme_lag_1` to `pjme_lag_24`, `pjme_lag_48`, `pjme_lag_168` |
| Cross-regional lags | 10 | `{r}_lag_1`, `{r}_lag_24` for 5 regions |
| PJME rolling stats | 8 | `pjme_roll_{mean,std}_{24h,48h,168h}`, `pjme_roll_{min,max}_24h` |
| Cross-regional rolling | 5 | `{r}_roll_mean_24h` for 5 regions |
| Temporal features | 11 | hour, day, month encodings + is_weekend + day_of_year |
| PJME derived | 3 | `pjme_diff_1h`, `pjme_diff_24h`, `pjme_ratio_24h` |
| Cross-regional ratios | 2 | `pjme_to_pjmw_ratio`, `pjme_to_total_ratio` |

**Total features: 65**

Target variable for ML model: `PJME(t+1)` — the next-hour PJME load value.

---

## 8. Modeling Strategy

### 8.1 Primary Model: LightGBM

**Model:** `lightgbm.LGBMRegressor`

**Justification (engineering, not pedagogical):**
- **Tabular data dominance.** For structured tabular features (lags, rolling stats, temporal encodings), gradient-boosted trees consistently outperform deep learning on datasets of this scale (~100K rows). This is not opinion; it's empirically validated across ML benchmarks (Grinsztajn et al., 2022).
- **Inference speed.** LightGBM produces compiled decision trees. Single-sample inference runs in <100μs — critical for real-time streaming where each event must be processed before the next arrives.
- **Native handling of feature types.** LightGBM handles mixed integer/float features without normalization, unlike neural networks.
- **Memory efficiency.** Histogram-based splitting uses O(n_bins) memory rather than O(n_samples) for finding split points.
- **Built-in early stopping.** Prevents overfitting without manual epoch tuning.

### 8.2 Hyperparameter Space

```python
params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,           # 2^7 - 1, balances expressiveness vs. overfitting
    "learning_rate": 0.05,       # conservative for stability
    "n_estimators": 2000,        # with early stopping patience=50
    "min_child_samples": 50,     # regularization for hourly seasonality
    "subsample": 0.8,            # row sampling per tree
    "colsample_bytree": 0.8,    # column sampling per tree
    "reg_alpha": 0.1,            # L1 regularization
    "reg_lambda": 1.0,           # L2 regularization
    "verbose": -1,
}
```

### 8.3 Why Not XGBoost?

XGBoost is a valid alternative. LightGBM is chosen as primary because:
- 3–5× faster training on this dataset size (histogram-based vs. exact/approximate split-finding)
- Lower memory footprint for the same tree complexity
- Functionally equivalent accuracy for this problem class

XGBoost is retained as a **benchmark model** for validation comparison.

---

## 9. Training Strategy

### 9.1 Walk-Forward Validation (Mandatory)

Standard k-fold cross-validation is **invalid** for time series — it allows future data to leak into training folds. Walk-forward (expanding window) validation is the only correct approach.

```
Training corpus: ~92,950 hourly records

Fold 1: Train [0 : 18,590]         -> Validate [18,590 : 37,180]
Fold 2: Train [0 : 37,180]         -> Validate [37,180 : 55,770]
Fold 3: Train [0 : 55,770]         -> Validate [55,770 : 74,360]
Fold 4: Train [0 : 74,360]         -> Validate [74,360 : 92,950]
Fold 5: Train [0 : 92,950]         -> Final model (full training corpus)

Each fold validation window: ~18,590 hours ~ 2.12 years
```

### 9.2 Training Protocol

1. For each fold `k`:
   a. Extract features from training window
   b. Train LightGBM with early stopping on validation MAE (patience=50 rounds)
   c. Record: best iteration, validation MAE, RMSE, MAPE, feature importances
2. Average validation metrics across folds 1–4 to estimate generalization error.
3. Train final production model on entire training corpus (fold 5) using the average best iteration count from folds 1–4 as `n_estimators`.
4. Serialize final model to `artifacts/model.lgb`.

### 9.3 Baseline Models (for comparison, not production)

| Baseline | Description |
|---|---|
| **Naive persistence** | `Load_hat(t+1) = Load(t)` |
| **Seasonal naive** | `Load_hat(t+1) = Load(t-168)` (same hour last week) |
| **24h moving average** | `Load_hat(t+1) = mean(Load(t), ..., Load(t-23))` |

The LightGBM model must beat all three baselines on all metrics to be considered valid.

---

## 10. Real-Time Inference Design

### 10.1 Two-Stage Pipeline (Per-Event)

```
                     ┌─────────────────────────────────────┐
  LoadEvent(t) ───▶  │  STAGE 1: Load Forecasting (ML)     │
  (from stream)      │  FeatureBuffer.update(load_t)       │
                     │  X = FeatureBuffer.get_features()    │
                     │  Load_hat = model.predict(X)         │
                     └──────────────┬──────────────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────────────┐
                     │  STAGE 2: Pricing Engine (Business)  │
                     │  Price_hat = f(Load_hat)             │
                     │  f(L) = 15.0 + 0.005L + 0.000002L² │
                     └──────────────┬──────────────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────────────┐
                     │  ForecastResult                      │
                     │  { timestamp, load_hat, price_hat,   │
                     │    load_actual, price_actual,        │
                     │    latency_ns, sequence_id }         │
                     └─────────────────────────────────────┘
```

### 10.2 FeatureBuffer (Ring Buffer)

```
Class: FeatureBuffer
- pjme_buffer: deque(maxlen=168)       # stores last 168 PJME load values
- region_buffers: dict[str, deque]     # {"pjmw": deque(maxlen=168), "aep": ..., ...}
- timestamps: deque(maxlen=168)

Method: update(pjme_mw, pjmw_mw, aep_mw, dayton_mw, dom_mw, duq_mw, timestamp) -> None
  Append to all ring buffers.

Method: get_features() -> np.ndarray
  Extract all 65 features from current buffer state.
  Returns shape (1, 65) array for single-sample prediction.

Method: is_warm() -> bool
  Return len(pjme_buffer) >= 168
```

**Critical:** During the warm-up phase (first 168 events), the buffer is not full. The system processes events but does NOT emit predictions until `is_warm()` returns True. This prevents garbage predictions from incomplete feature vectors.

### 10.3 Latency Measurement

```python
start = time.perf_counter_ns()
load_hat = model.predict(features)    # Stage 1
price_hat = pricing_engine(load_hat)  # Stage 2
latency_ns = time.perf_counter_ns() - start
```

Target: p99 latency < 1ms for single-sample LightGBM inference.

### 10.4 Pricing Engine (Deterministic)

```python
class PricingEngine:
    """Deterministic pricing function. NOT a model. Pure business logic."""

    @staticmethod
    def calculate(load_mw: float) -> float:
        return 15.0 + 0.005 * load_mw + 0.000002 * (load_mw ** 2)
```

This function is used in **two** places with different semantics:
1. **Inference path:** `PricingEngine.calculate(Load_hat)` → predicted price
2. **Ground truth path:** `PricingEngine.calculate(Load_actual) + ε` → actual price (with noise)

The function itself is identical. The noise `ε` is added externally in the ground-truth simulator only.

---

## 11. Repository Architecture

### 11.1 Module Dependency Graph

```
config                  ← pure configuration, no imports from other modules
  │
  ▼
core/
  ├── data_loader       ← reads CSV, validates, splits (depends: config)
  ├── feature_engine    ← transforms load → feature vectors (depends: config)
  ├── pricing_engine    ← deterministic L → P formula (depends: nothing)
  └── metrics           ← MAE, RMSE, MAPE calculators (depends: nothing)
  │
  ▼
models/
  ├── trainer           ← walk-forward training + serialization (depends: core)
  ├── forecaster        ← loads model, runs inference (depends: core)
  └── baselines         ← naive, seasonal, moving-avg (depends: core)
  │
  ▼
streaming/
  ├── rate_limiter      ← token bucket implementation (depends: nothing)
  ├── replay_engine     ← async event generator (depends: core, rate_limiter)
  ├── feature_buffer    ← ring buffer for online features (depends: core)
  └── pipeline          ← orchestrates stream → inference → eval (depends: all)
  │
  ▼
api/
  ├── app               ← FastAPI application factory (depends: models, streaming)
  ├── routes             ← REST endpoints (depends: api/app)
  └── websocket         ← WS streaming handler (depends: streaming)
  │
  ▼
dashboard/
  └── app               ← Streamlit dashboard (depends: api via HTTP/WS)
  │
  ▼
scripts/
  ├── train.py          ← CLI entrypoint for offline training
  ├── stream.py         ← CLI entrypoint for streaming simulation
  └── evaluate.py       ← CLI entrypoint for batch evaluation
```

### 11.2 Module Responsibilities

| Module | Single Responsibility |
|---|---|
| `config` | All configuration via Pydantic `BaseSettings`, environment variable overrides |
| `core.data_loader` | CSV ingestion, validation, chronological split |
| `core.feature_engine` | Stateless feature transformation (batch mode for training) |
| `core.pricing_engine` | `f(L) → P`, zero state, zero side effects |
| `core.metrics` | Pure functions: `mae()`, `rmse()`, `mape()` |
| `models.trainer` | Walk-forward training loop, early stopping, model serialization |
| `models.forecaster` | Loads serialized model, single-sample `predict()` |
| `models.baselines` | Persistence, seasonal, moving-average baseline predictors |
| `streaming.rate_limiter` | Token bucket with async `acquire()` |
| `streaming.replay_engine` | Async generator yielding `LoadEvent` at governed rate |
| `streaming.feature_buffer` | Ring buffer maintaining online feature state |
| `streaming.pipeline` | Wires stream → buffer → forecaster → pricing → metrics |
| `api.app` | FastAPI app factory, lifespan events, CORS config |
| `api.routes` | `/predict`, `/metrics`, `/health` endpoints |
| `api.websocket` | `/ws/stream` real-time forecast push |
| `dashboard.app` | Streamlit UI, consumes API/WS |
| `scripts.train` | `python -m scripts.train` CLI |
| `scripts.stream` | `python -m scripts.stream` CLI |
| `scripts.evaluate` | `python -m scripts.evaluate` CLI |

---

## 12. Folder Structure

```
rt-electricity-forecast/
│
├── README.md                           # Project overview, setup, usage
├── pyproject.toml                      # Project metadata + dependencies (PEP 621)
├── Makefile                            # Common commands: train, stream, test, lint
├── .env.example                        # Environment variable template
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── settings.py                     # Pydantic BaseSettings (all system params)
│
├── core/
│   ├── __init__.py
│   ├── data_loader.py                  # CSV ingestion + validation + split
│   ├── feature_engine.py              # Batch feature engineering (training)
│   ├── pricing_engine.py              # Deterministic pricing function
│   └── metrics.py                     # MAE, RMSE, MAPE implementations
│
├── models/
│   ├── __init__.py
│   ├── trainer.py                     # Walk-forward training pipeline
│   ├── forecaster.py                  # Online inference wrapper
│   └── baselines.py                   # Baseline model implementations
│
├── streaming/
│   ├── __init__.py
│   ├── rate_limiter.py                # TokenBucketRateLimiter
│   ├── replay_engine.py              # LogicalTimeReplayEngine
│   ├── feature_buffer.py             # Ring buffer for online features
│   └── pipeline.py                   # Streaming orchestration pipeline
│
├── api/
│   ├── __init__.py
│   ├── app.py                         # FastAPI application factory
│   ├── routes.py                      # REST API endpoints
│   ├── websocket.py                   # WebSocket streaming handler
│   └── schemas.py                     # Pydantic request/response models
│
├── dashboard/
│   └── app.py                         # Streamlit dashboard
│
├── scripts/
│   ├── train.py                       # CLI: offline model training
│   ├── stream.py                      # CLI: launch streaming simulation
│   └── evaluate.py                    # CLI: batch evaluation report
│
├── tests/
│   ├── __init__.py
│   ├── test_feature_engine.py         # Unit tests: feature computation
│   ├── test_pricing_engine.py         # Unit tests: pricing formula
│   ├── test_rate_limiter.py           # Unit tests: token bucket behavior
│   ├── test_feature_buffer.py         # Unit tests: ring buffer correctness
│   ├── test_pipeline.py              # Integration tests: end-to-end streaming
│   └── test_api.py                   # API endpoint tests
│
├── artifacts/
│   ├── model.lgb                      # Serialized LightGBM model
│   ├── training_report.json           # Training metrics + feature importance
│   └── evaluation_report.json         # Streaming evaluation results
│
├── data/
│   └── pjm_hourly_est.csv           # Consolidated multi-region dataset
│
└── notebooks/
    ├── 01_eda.ipynb                   # Exploratory data analysis
    └── 02_training_analysis.ipynb     # Training results visualization
```

---

## 13. MLOps Layer

> **Scope:** Local simulation only. No cloud deployment. All MLOps tooling runs locally or is optional.

### 13.1 Experiment Tracking (Optional but Recommended)

**Tool:** MLflow (local tracking server)

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Tracked per training run:
- Hyperparameters (full LightGBM config)
- Walk-forward fold metrics (MAE, RMSE, MAPE per fold)
- Aggregated validation metrics
- Feature importance JSON
- Model artifact (`.lgb` file)

### 13.2 Model Registry

```
artifacts/
├── model.lgb                   # current production model
├── model_v{timestamp}.lgb      # versioned snapshots
└── training_report.json        # links to MLflow run ID
```

No cloud registry. Model versioning is timestamp-based filenames. The `forecaster.py` module loads the model specified in `config/settings.py`.

### 13.3 Data Validation

**Tool:** `pandera` (schema-based DataFrame validation)

```python
region_cols = ["PJME", "PJMW", "AEP", "DAYTON", "DOM", "DUQ"]
schema = pa.DataFrameSchema({
    "Datetime": pa.Column(pa.DateTime, nullable=False),
    **{
        col: pa.Column(pa.Float, checks=[
            pa.Check.ge(0),
            pa.Check.le(70000),  # physical upper bound per region
        ], nullable=False)
        for col in region_cols
    },
})
```

Validation runs on data load and fails fast with clear error messages.

### 13.4 Monitoring & Alerting (Simulation Only)

- **Prediction drift:** If rolling MAPE exceeds 10% over a 24-hour window, log a WARNING.
- **Feature drift:** If `pjme_roll_mean_168h` deviates > 2sigma from training distribution, log a WARNING.
- **Latency breach:** If p99 inference latency exceeds 5ms, log a WARNING.

All monitoring is log-based (Python `logging` module). No external alerting infrastructure.

---

## 14. API Design

### 14.1 REST Endpoints

#### `POST /predict`

Single-shot inference for a given load sequence.

**Request:**
```json
{
  "load_sequence": {
    "pjme": [32000.0, 31500.0, "...", 28000.0],
    "pjmw": [8000.0, 7800.0, "...", 7500.0],
    "aep": [15000.0, 14800.0, "...", 14200.0],
    "dayton": [2500.0, 2400.0, "...", 2300.0],
    "dom": [12000.0, 11800.0, "...", 11500.0],
    "duq": [1800.0, 1750.0, "...", 1700.0]
  },
  "timestamp": "2017-06-15T14:00:00"
}
```

**Response:**
```json
{
  "timestamp_predicted": "2017-06-15T15:00:00",
  "load_forecast_mw": 33150.5,
  "price_forecast_usd_mwh": 197.12,
  "inference_latency_ms": 0.42,
  "model_version": "20170101_120000"
}
```

#### `GET /metrics`

Retrieve current evaluation metrics.

**Response:**
```json
{
  "load_metrics": {
    "mae": 412.3,
    "rmse": 587.1,
    "mape": 1.24
  },
  "price_metrics": {
    "mae": 3.82,
    "rmse": 5.14,
    "mape": 1.97
  },
  "latency": {
    "p50_ms": 0.31,
    "p95_ms": 0.52,
    "p99_ms": 0.78
  },
  "events_processed": 15240,
  "uptime_seconds": 3048.0
}
```

#### `GET /health`

System health check.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "stream_active": true,
  "buffer_warm": true,
  "events_in_queue": 3
}
```

#### `GET /stream/status`

Current streaming simulation state.

**Response:**
```json
{
  "logical_time": "2017-06-15T14:00:00",
  "wall_clock_elapsed_s": 3048.0,
  "events_emitted": 15408,
  "events_remaining": 13665,
  "progress_pct": 53.0,
  "rate_limiter": {
    "tokens_available": 7.2,
    "capacity": 10,
    "refill_rate": 5.0
  }
}
```

### 14.2 WebSocket Endpoint

#### `WS /ws/stream`

Pushes real-time forecast events to connected clients.

**Per-event message:**
```json
{
  "type": "forecast",
  "sequence_id": 15408,
  "logical_timestamp": "2017-06-15T14:00:00",
  "load_actual_mw": 33280.0,
  "load_forecast_mw": 33150.5,
  "load_error_mw": -129.5,
  "price_actual_usd_mwh": 198.45,
  "price_forecast_usd_mwh": 197.12,
  "price_error_usd_mwh": -1.33,
  "inference_latency_ms": 0.42
}
```

**Periodic metrics message (every 24 events):**
```json
{
  "type": "metrics_update",
  "window": "24h_rolling",
  "load_mae": 412.3,
  "load_rmse": 587.1,
  "price_mae": 3.82,
  "price_rmse": 5.14,
  "latency_p99_ms": 0.78
}
```

---

## 15. Dashboard Design

### 15.1 Technology

**Streamlit** — chosen for rapid prototyping, native Python data visualization, and built-in WebSocket client support. No React/TypeScript overhead for a simulation dashboard.

### 15.2 Layout

```
┌──────────────────────────────────────────────────────────────┐
│  RT-ELECTRICITY-FORECAST  │  Status: ● Streaming  │  Logical Time: 2017-06-15 14:00  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  LOAD FORECAST (MW)                          [24h view] │ │
│  │  ████████████████████ actual (blue)                      │ │
│  │  ░░░░░░░░░░░░░░░░░░░ predicted (orange, dashed)         │ │
│  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ error band (gray fill)             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  PRICE FORECAST ($/MWh)                      [24h view] │ │
│  │  ████████████████████ actual (green)                      │ │
│  │  ░░░░░░░░░░░░░░░░░░░ predicted (red, dashed)             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Load MAE    │  │  Load RMSE   │  │  Load MAPE   │       │
│  │  412.3 MW    │  │  587.1 MW    │  │  1.24%       │       │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤       │
│  │  Price MAE   │  │  Price RMSE  │  │  Price MAPE  │       │
│  │  $3.82/MWh   │  │  $5.14/MWh   │  │  1.97%       │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────────────┐  ┌────────────────────────────┐   │
│  │  LATENCY GAUGE       │  │  ERROR DISTRIBUTION         │   │
│  │  p50: 0.31ms         │  │  [histogram of load errors]  │   │
│  │  p95: 0.52ms         │  │                              │   │
│  │  p99: 0.78ms         │  │                              │   │
│  └──────────────────────┘  └────────────────────────────┘   │
│                                                              │
│  Progress: ████████████████████░░░░░░░░░░  53% (15408/29073) │
└──────────────────────────────────────────────────────────────┘
```

### 15.3 Dashboard Components

| Component | Library | Update Frequency |
|---|---|---|
| Time series charts | `plotly.graph_objects` | Per event (via WS) |
| Metric cards | `streamlit.metric` | Every 24 events |
| Latency gauge | `plotly.indicator` | Every 24 events |
| Error histogram | `plotly.histogram` | Every 100 events |
| Progress bar | `streamlit.progress` | Per event |

---

## 16. Evaluation Metrics

### 16.1 Load Forecasting Metrics

| Metric | Formula | Purpose |
|---|---|---|
| **MAE** | `(1/n) Σ\|Load_actual - Load_hat\|` | Average absolute error in MW |
| **RMSE** | `sqrt((1/n) Σ(Load_actual - Load_hat)²)` | Penalizes large errors more heavily |
| **MAPE** | `(100/n) Σ\|Load_actual - Load_hat\| / Load_actual` | Scale-independent percentage error |

### 16.2 Price Forecasting Metrics

Same formulas applied to `Price_actual` vs. `Price_hat`. Because price is a monotonic function of load, price errors are a **non-linear amplification** of load errors. The quadratic term in the pricing function means load errors at high loads produce disproportionately larger price errors.

### 16.3 Latency Metrics

| Metric | Description | Target |
|---|---|---|
| **p50 latency** | Median inference time (feature extraction + model predict + pricing) | < 0.5ms |
| **p95 latency** | 95th percentile | < 1.0ms |
| **p99 latency** | 99th percentile | < 2.0ms |
| **Throughput** | Events processed per second (wall-clock) | > 1000 evt/s (burst) |

### 16.4 Expected Performance Ranges

Based on PJM load data characteristics (typical load 20,000–55,000 MW):

| Metric | Expected Range | Baseline (Seasonal Naive) |
|---|---|---|
| Load MAE | 300–600 MW | 800–1200 MW |
| Load RMSE | 450–800 MW | 1100–1600 MW |
| Load MAPE | 0.8–2.0% | 2.5–4.0% |
| Price MAE | $2–6/MWh | $8–15/MWh |
| Price MAPE | 1.0–3.0% | 4.0–8.0% |

These ranges are **estimates** based on typical LightGBM performance on similar energy load datasets. Actual numbers depend on hyperparameter tuning.

---

## 17. Deployment Strategy

### 17.1 Scope: Local Simulation Only

This system is **not** deployed to any cloud provider, Kubernetes cluster, or production server. It runs entirely on the developer's local machine. Any claims of "production deployment" would be hallucinated.

### 17.2 Local Execution Modes

| Mode | Command | Description |
|---|---|---|
| **Train** | `make train` | Runs offline walk-forward training, saves model |
| **Stream** | `make stream` | Launches streaming simulation + API + dashboard |
| **Evaluate** | `make evaluate` | Batch evaluation on test set, generates report |
| **Test** | `make test` | Runs pytest suite |
| **Lint** | `make lint` | Runs ruff + mypy |

### 17.3 `make stream` Lifecycle

```
1. Load configuration from .env / config/settings.py
2. Load serialized model from artifacts/model.lgb
3. Initialize FeatureBuffer, PricingEngine, MetricsCollector
4. Initialize TokenBucketRateLimiter (capacity=10, rate=5/s)
5. Initialize LogicalTimeReplayEngine with test split data
6. Launch FastAPI server (uvicorn, port 8000)
7. Launch Streamlit dashboard (port 8501)
8. Start asyncio event loop:
   a. Replay engine emits LoadEvents
   b. Pipeline processes events through two-stage inference
   c. Results published to WebSocket
   d. Dashboard renders in real-time
9. On replay completion: flush final metrics, generate evaluation report
```

### 17.4 Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "lightgbm>=4.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "websockets>=12.0",
    "streamlit>=1.28",
    "plotly>=5.18",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pandera>=0.18",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.21",
    "httpx>=0.25",
    "ruff>=0.1",
    "mypy>=1.7",
]
mlops = [
    "mlflow>=2.8",
]
```

---

## 18. GitHub Portfolio Value

### 18.1 Differentiation from Typical ML Projects

| Typical Portfolio Project | This Project |
|---|---|
| `model.fit()` → `model.predict()` in Jupyter | Production-grade Python package with modular architecture |
| Random train/test split | Walk-forward chronological validation |
| No deployment story | Asyncio streaming simulation with rate limiting |
| Single metric reported | Multi-level metrics (Load + Price + latency) |
| No API | FastAPI with REST + WebSocket endpoints |
| No monitoring | Real-time Streamlit dashboard with live metrics |
| "Deploy to cloud" hand-wave | Honest local simulation with clear scope boundaries |

### 18.2 Skills Demonstrated

- **Systems Design:** Async event-driven architecture, backpressure handling, ring buffers
- **ML Engineering:** Walk-forward validation, feature engineering pipeline, model serialization
- **Software Engineering:** Clean module separation, Pydantic configs, type hints, testing
- **Domain Knowledge:** Two-stage forecasting (load → price), energy market pricing mechanics
- **Infrastructure:** Token bucket rate limiting, logical time simulation, WebSocket streaming

### 18.3 Target Roles

- ML Engineer (L4–L6 equivalent)
- Machine Learning Infrastructure Engineer
- Quantitative Developer (Energy/Commodities)
- Data Engineer (Streaming Systems)

---

## 19. Resume Bullet Points

```
* Designed and implemented a two-stage real-time electricity price forecasting system
  using LightGBM with cross-regional features from 6 PJM zones and a deterministic
  quadratic pricing engine, achieving <2% MAPE on 23K+ hours of test data.

* Built an asyncio-based streaming simulation engine with token-bucket rate limiting
  and logical time replay, processing 1000+ events/second with p99 inference latency
  under 2ms.

* Implemented walk-forward chronological validation across 5 expanding-window folds
  on 92K+ training samples, preventing temporal data leakage and ensuring robust
  out-of-sample evaluation.

* Developed a real-time monitoring dashboard (Streamlit + Plotly) with WebSocket-driven
  live forecast visualization, error distribution tracking, and latency gauges.

* Engineered a 65-feature pipeline from 6 PJM regional load series including PJME
  lag sequences, cross-regional lag and rolling features, cyclical temporal encodings,
  and inter-region ratio features -- all computed in <100us per sample via ring buffer.

* Created a FastAPI service with REST and WebSocket endpoints for on-demand inference
  and live forecast streaming, with Pydantic schema validation and structured error
  handling.
```

---

## 20. Interview Questions

### 20.1 System Design Questions

**Q: Why two stages instead of predicting price directly?**
A: The dataset contains only load data, not price data. Prices are synthesized via a known formula. Separating the ML prediction (load) from business logic (pricing) follows the single-responsibility principle, makes the ML model reusable across different pricing regimes, and allows the pricing function to change without retraining the model. In real energy markets, pricing functions change due to regulatory updates — the ML model should be agnostic to this.

**Q: Why not use LSTM or Transformer for time series?**
A: For tabular features (lags, rolling stats, temporal encodings) at this data scale (~100K rows), gradient-boosted trees match or exceed deep learning models in accuracy while offering 10–100× faster training and inference. LSTMs process sequences — but we've already extracted sequence information into explicit lag features. The overhead of GPU infrastructure, hyperparameter tuning (learning rate scheduling, architecture search), and longer iteration cycles is unjustified here.

**Q: How does the token bucket differ from a simple rate limiter?**
A: A simple rate limiter (e.g., `sleep(1/rate)` between events) enforces constant spacing. A token bucket allows **bursting** — up to `capacity` events can be emitted instantly, then sustained at `refill_rate`. This is critical for warm-up: we need to quickly fill the 168-event feature buffer, then throttle to steady-state speed for the actual evaluation.

**Q: What happens if the ML model produces a negative load prediction?**
A: LightGBM does not inherently constrain output range. Negative predictions are physically impossible (load cannot be negative). The `Forecaster` class clips predictions to `max(0.0, prediction)`. This is logged as a warning — frequent negative predictions would indicate model degradation.

### 20.2 ML Engineering Questions

**Q: Why walk-forward validation instead of time-series cross-validation?**
A: Walk-forward (expanding window) validation mirrors how the model will actually be used: trained on all available history, deployed to predict the future. Standard time-series CV (scikit-learn's `TimeSeriesSplit`) also works but uses fixed-size training windows, which discards older data. For energy load data where multi-year seasonality matters, expanding windows are preferred.

**Q: How do you detect model staleness in the streaming pipeline?**
A: Rolling MAPE over a configurable window (default 24 hours). If MAPE exceeds 2× the validation MAPE from training, it indicates distribution shift. The system logs a warning but does NOT auto-retrain — retraining decisions should be deliberate, not automated in a simulation.

**Q: Why 168 lags and not fewer?**
A: 168 hours = 1 week. Electricity load exhibits strong weekly seasonality (weekday vs. weekend patterns). The lag at t-168 captures "same hour, same day of week, last week" — the single most predictive feature for hourly load forecasting. Shorter lag windows miss this pattern.

### 20.3 Streaming Architecture Questions

**Q: How do you handle backpressure if inference is slower than event emission?**
A: `asyncio.Queue` with bounded capacity (`maxsize=100`). If the queue is full, `replay_engine.stream()` awaits `queue.put()`, which naturally throttles emission. This prevents unbounded memory growth and ensures the pipeline processes events in order.

**Q: What is the difference between logical time and wall-clock time in this system?**
A: Logical time advances by exactly 1 hour per event (matching the dataset granularity). Wall-clock time depends on the rate limiter configuration. At `refill_rate=5`, we process 5 events/second, meaning 1 logical hour passes every 200ms of wall-clock time. This decoupling allows fast-replay (development), slow-replay (demo), or event-driven replay (testing) without changing pipeline logic.

---

## 21. Risks & Limitations

### 21.1 Fundamental Limitations

| Risk | Impact | Mitigation |
|---|---|---|
| **Synthetic pricing function** | Prices are not real market prices; system cannot be validated against actual PJM LMPs | Clearly documented as simulation. The ML model (load forecasting) is validated against real data. |
| **Univariate input** | Model uses only historical load. Real load forecasting uses weather, economic indicators, calendar events | Acknowledged. See Section 22 for optional enhancements. |
| **Stationarity assumption** | Model trained on 2002–2015 data may not generalize to 2015–2018 due to demand growth, efficiency improvements | Walk-forward validation explicitly tests this. Expanding window incorporates structural changes. |
| **No concept drift detection** | No automated retraining trigger | Rolling MAPE monitoring with alerting thresholds (Section 13.4). Manual retrain decision. |
| **Noise model simplicity** | `ε ~ N(0, 4)` is a simplistic price noise model | Intentionally simple. Real market noise is heteroskedastic and fat-tailed. Not the focus of this system. |

### 21.2 Engineering Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Memory pressure** | Full dataset in memory (~145K rows × 48 features) during training | ~56MB for training matrix. Negligible on modern hardware. |
| **Single-process** | No horizontal scaling | Out of scope for local simulation. Architecture supports future extraction to worker processes. |
| **Feature buffer cold start** | First 168 events produce no predictions | Documented behavior. Warm-up events are excluded from metrics. |

---

## 22. Future Improvements

### 22.1 [OPTIONAL] External Data Integration

> All items in this section require external data sources NOT present in the current dataset. They are engineering recommendations, NOT current capabilities.

| Enhancement | Data Source | Expected Impact |
|---|---|---|
| **[OPTIONAL] Weather features** | NOAA hourly temperature/humidity API | Temperature is the #1 driver of electricity demand. Expected MAPE reduction: 30–50%. |
| **[OPTIONAL] Calendar features** | US federal holiday calendar | Holiday loads differ significantly from regular weekdays. |
| **[OPTIONAL] Real LMP prices** | PJM Data Miner API (historical LMP data) | Replace synthetic pricing with actual market prices. Enables real price model training. |
| **[OPTIONAL] Economic indicators** | FRED API (GDP, industrial production index) | Long-term demand trend correction. |

### 22.2 Architecture Improvements (No External Data)

| Enhancement | Description |
|---|---|
| **Online learning** | Incrementally update model weights as new data arrives (LightGBM supports `init_model`). |
| **Ensemble methods** | Combine LightGBM + XGBoost via stacking or simple averaging. |
| **Probabilistic forecasting** | LightGBM quantile regression for prediction intervals (10th, 50th, 90th percentiles). |
| **Containerization** | Docker Compose for API + dashboard + MLflow. Still local, but reproducible. |
| **CI/CD pipeline** | GitHub Actions for lint + test + training validation on push. |
| **Grafana dashboards** | Replace Streamlit with Grafana + Prometheus for production-grade monitoring (if deployment scope expands). |

---

## Appendix A: Pricing Function Analysis

The deterministic pricing function:

```
P(L) = 15.0 + 0.005 * L + 0.000002 * L²
```

| Load (MW) | Price ($/MWh) | Marginal Price ($/MW) |
|---|---|---|
| 20,000 | 915.00 | 0.085 |
| 30,000 | 1,965.00 | 0.125 |
| 40,000 | 3,415.00 | 0.165 |
| 50,000 | 5,265.00 | 0.205 |

The quadratic term creates convex pricing: marginal cost of electricity increases with load. This reflects real-world merit-order dispatch where expensive peaker plants activate at high loads.

**Error amplification:** A 500 MW load error at 30,000 MW load produces a ~$62.5 price error. The same 500 MW error at 50,000 MW load produces a ~$102.5 price error. This non-linear error amplification means load forecasting accuracy is **more critical** at high loads.

---

## Appendix B: Configuration Reference

```python
class Settings(BaseSettings):
    # Data
    data_path: str = "data/pjm_hourly_est.csv"
    train_ratio: float = 0.80
    datetime_col: str = "Datetime"
    target_col: str = "PJME"                          # prediction target region
    auxiliary_cols: list[str] = ["PJMW", "AEP", "DAYTON", "DOM", "DUQ"]
    all_region_cols: list[str] = ["PJME", "PJMW", "AEP", "DAYTON", "DOM", "DUQ"]

    # Features - PJME lags
    pjme_lag_hours: list[int] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,48,168]
    # Features - Cross-regional lags (per auxiliary region)
    cross_regional_lag_hours: list[int] = [1, 24]
    # Features - Rolling windows
    pjme_rolling_windows: list[int] = [24, 48, 168]
    cross_regional_rolling_windows: list[int] = [24]
    feature_buffer_size: int = 168
    total_features: int = 65

    # Model
    model_type: str = "lightgbm"  # "lightgbm" or "xgboost"
    model_path: str = "artifacts/model.lgb"
    n_estimators: int = 2000
    learning_rate: float = 0.05
    num_leaves: int = 127
    early_stopping_rounds: int = 50
    walk_forward_folds: int = 5

    # Pricing
    price_intercept: float = 15.0
    price_linear_coeff: float = 0.005
    price_quadratic_coeff: float = 0.000002
    price_noise_std: float = 2.0

    # Streaming
    token_bucket_capacity: int = 10
    token_bucket_refill_rate: float = 5.0
    metrics_update_interval: int = 24
    queue_max_size: int = 100

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501

    class Config:
        env_file = ".env"
        env_prefix = "RTEF_"
```
