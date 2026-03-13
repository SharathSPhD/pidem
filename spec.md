# Fuel Pricing Intelligence Lab — Technical Specification

> **Version:** 1.0 | **Status:** Final
> **Companion Document:** PRD.md (product requirements and business context)
> **Stack:** Python, FastAPI, Plotly Dash (React/Next.js migration path), PyTorch

---

## 1. System Architecture Overview

### 1.1 Architecture Diagram

```
                           +-----------------------+
                           |    Browser Client      |
                           |  (Dash / React+Plotly) |
                           +----------+------------+
                                      |
                                      | HTTP / WebSocket
                                      v
                           +----------+------------+
                           |   FastAPI Application  |
                           |  (Routers + Middleware) |
                           +--+-------+----------+--+
                              |       |          |
               +--------------+   +---+---+   +--+-----------+
               |                  |       |   |              |
     +---------v--------+  +-----v--+  +--v--v--------+  +--v-----------+
     | Synthetic Data    |  | Model  |  | Async Task   |  | LLM Client   |
     | Engine + Cache    |  | Layer  |  | Queue        |  | (Claude API)  |
     | (generator.py,    |  | (ML    |  | (Background  |  | (llm_client.py|
     |  cache.py)        |  |  models)|  |  Tasks /     |  |  + RAG)      |
     +-------------------+  +--------+  |  Celery+Redis)|  +--------------+
                                        +--------------+
```

### 1.2 Data Flow

```
User Interaction (slider/button/dropdown)
    |
    v
Frontend Callback / React Event Handler
    |
    v
HTTP Request to FastAPI Router (POST /api/v1/models/...)
    |
    +---> [Fast Model] --> Synchronous response
    |         Train model, generate Plotly JSON, return immediately
    |
    +---> [Slow Model] --> Async response
              Create background task, return run_id
              Client polls GET /api/v1/runs/{run_id}/status
              On completion: GET /api/v1/runs/{run_id}/results
    |
    v
Response: { "figure": "<plotly_json>", "metrics": {...}, ... }
    |
    v
Frontend renders Plotly chart from JSON
```

### 1.3 API Versioning

All endpoints are prefixed with `/api/v1/`. This enables future breaking changes under `/api/v2/` without disrupting existing frontends.

---

## 2. Synthetic Data Engine

### 2.1 Design Goals

The synthetic data must be realistic enough to produce non-trivial model behavior: clustering should find meaningful station groups, anomaly detection should surface plausible events, time series models should discover genuine seasonality. Toy data produces toy insights.

The data engine is geography-agnostic and configurable. All station names, competitor identities, and geographic coordinates are synthetic.

### 2.2 Station Network

**Default configuration: 80 stations, 3-year hourly data**

| Attribute | Values / Distribution |
|---|---|
| Station ID | `STN_001` to `STN_080` (configurable count) |
| Type | Motorway: 25, Urban: 40, Rural: 15 |
| Region | Region A through Region E (configurable count and names) |
| Capacity (L/day) | Motorway: N(120k, 15k), Urban: N(55k, 12k), Rural: N(28k, 6k) |
| Competitors within 1km | Poisson(lambda): Motorway=1.2, Urban=2.8, Rural=0.6 |
| Lat/Long | Configurable geographic spread; default generates a realistic national station network clustered along major highway corridors |

All station network parameters are configurable via the data generation API to support any national fuel retail geography.

### 2.3 Price Series Generation

**Our price** is modelled as:

```
P_our(t) = P_reference(t)
          + strategic_offset(station_type)
          + competitive_response(t)
          + manager_discretion(t)
          + epsilon_price(t)
```

Where:

| Component | Specification |
|---|---|
| `P_reference(t)` | Synthetic wholesale price derived from a crude oil base via geometric Brownian motion. Weekly volatility ~0.3ct/day std. Includes a configurable spike event (e.g., energy crisis: +30ct over 6 weeks) |
| `strategic_offset` | Motorway: +4 to +8ct vs. Urban; Rural: -1 to +2ct vs. Urban. Configurable per station type |
| `competitive_response(t)` | Probabilistic reaction function — 70% chance of partial match within 4h on Motorway, 40% within 24h Urban. Parameterized by station type |
| `manager_discretion` | +/-2ct random walk at station level, weekly reset |
| `epsilon_price(t)` | N(0, 0.4ct) observation noise |

**Competitor price generation** (per competitor per station):

| Competitor Archetype | Behavior |
|---|---|
| Competitor A (Tracker) | Tracks our price with lag and independent noise |
| Competitor B (Aggressive) | Mean-reverts to -1ct vs. area average; more reactive |
| Competitor C (Stable) | Low volatility, less reactive to short-term moves |
| Competitor D (Independent) | High variance, occasional deep discounting events |

### 2.4 Volume Series Generation

**True demand model** (generating process — not revealed to learners, they must discover it):

```python
log(V_t) = alpha_station
         + beta1 * (P_our - P_min_competitor)    # cross-price gap (key driver)
         + beta2 * P_our                           # own price level
         + beta3 * log(V_{t-1})                    # autoregressive (habit)
         + gamma1 * hour_of_day_spline(t)          # intraday pattern
         + gamma2 * day_of_week_dummies(t)         # weekly seasonality
         + gamma3 * month_spline(t)                # annual seasonality
         + gamma4 * is_public_holiday(t)           # holiday effect
         + gamma5 * temp_deviation(t)              # weather effect on driving
         + delta * highway_traffic_index(t)        # motorway-specific traffic
         + epsilon_volume(t)                       # N(0, sigma_station)
```

**Elasticity parameters** (true values, hidden from learners):

| Parameter | Motorway | Urban | Rural |
|---|---|---|---|
| beta1 (cross-price gap) | -1.8 | -1.3 | -0.9 |
| beta2 (own price level) | -0.4 | -0.4 | -0.4 |

Seasonality effects: +35% summer motorway, -15% January, +20% Friday afternoon.

**Realism features:**

| Feature | Specification |
|---|---|
| Station fixed effects | `alpha_station` has genuine variation (+/-40% from mean) |
| Heteroscedasticity | Higher variance at low-volume stations |
| Outliers | 0.3% of observations are genuine anomalies (competitor closure, road event, data error) |
| Regime change | A configurable crisis period shifts demand elasticity temporarily |

### 2.5 Exogenous Inputs

| Variable | Generation Method |
|---|---|
| Crude oil price (ct/L) | Geometric Brownian Motion with configurable spike events |
| Wholesale / COGS (ct/L) | Crude * conversion_factor + refinery margin noise |
| Temperature (deg C) | Realistic annual cycle for configurable latitude + day-to-day noise |
| Highway traffic index | Weekly/seasonal synthetic pattern (configurable national traffic profile) |
| Public holidays | Configurable holiday calendar (national + regional) |
| Event flags | Configurable discrete events: competitor closures, road infrastructure events |

### 2.6 Data API Contracts

Six core DataFrames exposed via the API:

```python
df_stations:  [station_id, type, region, lat, lon, n_competitors, capacity_l_day]

df_prices:    [station_id, datetime, our_price, cogs, gross_margin,
               comp_a_price, comp_b_price, comp_c_price, comp_d_price,
               min_comp_price]

df_volume:    [station_id, datetime, volume_litres, gross_profit_eur]

df_market:    [datetime, crude_eur, cpi_index, temperature, highway_index]

df_daily:     [station_id, date, ...]    # daily aggregated for most models

df_hourly:    [station_id, datetime, ...]  # full granularity for time series
```

All schemas are defined as Pydantic models in `data/schemas.py` and serve as the API contract for any frontend.

---

## 3. Module Technical Specifications

---

### M0 — The Data Mindset & ML Taxonomy

**Scenario:** "We have 3 years of data across 80 stations — millions of rows. What questions can the data answer that a human analyst cannot? And which questions require human judgment that no amount of data can replace?"

**Core Concepts:**
- Supervised vs. Unsupervised vs. Reinforcement Learning (taxonomy with fuel pricing examples for each)
- The bias-variance tradeoff: why complex models overfit and simple models underfit
- Train / validation / test split — the "peek test" for honest evaluation
- What is a feature? What is a target? Why does the choice matter enormously?

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Polynomial degree | Slider | 1–12 | Fit progressively complex curves to price-volume scatter; watch train error decrease while test error U-curves |
| Train/test split ratio | Slider | 60%–90% | See how performance estimates become unreliable with small test sets |

**Visualizations:**
- `polynomial_fit_chart`: scatter of price vs. volume + fitted curve, with two RMSE readouts (train/test) that diverge as degree increases

**Quality Gate:** N/A (conceptual module)

**Break It:** Set polynomial degree = 12 — watch test MSE explode while train MSE approaches zero.

**Key Insight:** "More data is not always better than better features. Knowing that a station is on a major highway near an airport is more predictive than 10 more weeks of noisy volume data."

**API Endpoints:**
- `POST /api/v1/models/foundations/polyfit` — fit polynomial of given degree to selected station data
- `GET /api/v1/data/stations/{station_id}/price_volume` — raw scatter data

---

### M1 — Regression & Price Elasticity

**Scenario:** "If we cut diesel price 4ct below the nearest competitor at a motorway station, we expect a volume uplift. But how much? And is that enough to compensate the margin hit? We need a number — with an honest confidence interval."

**Model Formulation:**

The log-log demand model for elasticity estimation:

```
log(Volume_it) = alpha_i + beta1 * (P_our - P_min_comp)_it + beta2 * P_our_it
               + beta3 * log(Volume_{i,t-1}) + SUM(gamma_k * Controls_kit) + epsilon_it
```

- `alpha_i` = station fixed effect (absorbs permanent station-level differences)
- `beta1` = cross-price elasticity — the key decision number
- `exp(beta1 * delta_gap) - 1` = expected % volume change for `delta_gap` cent change
- Standard errors yield confidence intervals on the elasticity

**Interactive Controls:**

| Control | Type | Range/Options | Effect |
|---|---|---|---|
| Feature selector | Checkboxes | hour_of_day, day_of_week, weather, competitor_gap, ... | Toggle controls included in model; watch coefficients and R-squared change |
| Station type selector | Dropdown | Motorway / Urban / Rural / All | Separate elasticity estimates by station type |
| Time period selector | Date range | Full dataset range | Estimate on pre-crisis vs. crisis data — regime change visible |
| Regularization (Ridge lambda) | Slider | 0–100 | Show overfitting reduction with regularization |

**Visualizations:**
- **Elasticity coefficient plot:** bar chart of beta1 with 95% CI across station types — error bars are the business message
- **Partial response curve:** volume change as price gap varies from -8ct to +8ct, holding all else equal; fan chart showing prediction intervals
- **Residual diagnostics:** actual vs. predicted scatter, residuals vs. fitted (should be flat band), QQ-plot for normality
- **Leverage/influence plot (Cook's Distance):** identify outlier stations disproportionately driving elasticity estimates

**Diagnostics to Expose:**
- Heteroscedasticity (variance increases with volume) — suggest log transform
- Autocorrelation in residuals — "your model doesn't know about yesterday"
- Multicollinearity (our_price and comp_price correlated) — VIF table

**Quality Gate:** R-squared > 0.5 on test set; residual ACF not significantly autocorrelated

**Break It:** Include only `our_price` without competitor gap — watch multicollinearity inflate standard errors and destabilize coefficients.

**Key Insight:** "The elasticity on Motorway stations is -1.8, but the confidence interval is [-1.4, -2.2]. At -1.4, the margin break-even requires +28% volume. At -2.2, only +18%. That uncertainty changes the decision."

**API Endpoints:**
- `POST /api/v1/models/regression/train` — train elasticity model with selected features and station type
- `GET /api/v1/models/regression/{run_id}/diagnostics` — residual plots, VIF table, Cook's distance

---

### M2 — Classification & Competitive Threat Detection

**Scenario:** "Every hour, 80 stations face a decision: has a competitor just triggered a price war that requires immediate response, or is this routine noise? A classification model flags genuine threats — so the pricing team focuses on what matters."

**Tasks Modeled:**
1. **Binary:** P(volume_loss > 15% in next 4h | current features) — does a price gap warrant action?
2. **Multiclass:** Optimal strategy = {aggressive, parity, premium} given station state

**Models Covered:**
- Logistic Regression (interpretable baseline — coefficients readable as log-odds)
- Decision Tree (actual rules learned — pricing managers can read these directly)
- Gradient Boosting (XGBoost) — best performance

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Decision threshold | Slider | 0.1–0.9 | Precision/recall tradeoff live — at 0.3: catch 90% of threats but 40% false alerts; at 0.8: only 50% caught but 5% false alerts |
| Feature importance ranking | Drag-to-reorder | All available features | Include/exclude features; see accuracy response |
| Tree depth | Slider | 1–15 | Watch decision tree from interpretable (depth 3: 7 leaf rules) to overfit (depth 12: memorizes training data) |

**Visualizations:**
- **Decision tree diagram (depth <= 5):** actual pricing rules — "IF price_gap > 2.3ct AND station_type = Motorway AND hour in [7-9] THEN aggressive"
- **Confusion matrix (interactive threshold):** annotated with business cost per cell (false negative = missed competitive threat, estimated revenue loss; false positive = unnecessary price cut, estimated margin cost)
- **ROC curve + AUC:** all models together for comparison
- **Precision-recall curve:** more honest for the imbalanced case (9% of hours are genuine threats)
- **SHAP waterfall for single prediction:** "Why did the model flag this station at 08:22 on Tuesday?"

**Quality Gate:** ROC-AUC > 0.72; calibration error < 0.05

**Break It:** Set tree depth = 15 — watch train accuracy reach 99% while test accuracy drops below 70%.

**Key Insight:** "The decision tree at depth=4 achieves 83% of XGBoost's accuracy but every pricing manager can read the rules. That interpretability is worth 4 percentage points."

**API Endpoints:**
- `POST /api/v1/models/classification/train` — train classifier with model type, features, threshold
- `GET /api/v1/models/classification/{run_id}/tree` — decision tree structure as renderable JSON
- `GET /api/v1/models/classification/{run_id}/shap/{observation_id}` — SHAP waterfall for specific observation

---

### M3 — Ensemble Methods & Explainability

**Scenario:** "Single decision trees are fragile — tiny data changes flip predictions. Random Forest averages hundreds of trees for robustness. XGBoost builds them sequentially, correcting errors. But a 'black box' in pricing is a compliance problem — so we need SHAP."

**Concepts:**
- Bagging (Random Forest) vs. Boosting (XGBoost/LightGBM) — the fundamental difference
- Feature importance: Gini impurity vs. permutation importance vs. SHAP — why they disagree
- SHAP (SHapley Additive exPlanations): game-theoretic, locally accurate, honest

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| n_estimators | Slider | 10–500 | Validation MAE improves then plateaus (marginal returns) |
| learning_rate x n_estimators | Dual slider | lr: 0.01–0.5, n: 10–500 | Iso-performance curves showing GBM tradeoff |
| Feature selection | Checkboxes | All available features | Include/exclude; see which features move the needle |
| SHAP sample selector | Dropdown/search | Any station-hour observation | Individual SHAP explanation for selected observation |

**Visualizations:**
- **Learning curves:** train + validation MAE vs. training data size — "How much data does this model actually need?"
- **SHAP beeswarm plot (global):** distribution of SHAP values per feature — shows which features help which predictions and in which direction
- **SHAP dependence plot:** SHAP(price_gap) vs. price_gap — should show non-linear kink at ~0 (competitive parity breakpoint)
- **SHAP interaction plot:** price_gap x station_type interaction — motorway stations respond differently to the same gap

**Quality Gate:** Validation MAE improvement over single tree > 15%

**Break It:** Set learning_rate = 0.5 with n_estimators = 500 — watch overfitting as boosting memorizes noise.

**Key Insight:** "Feature importance says `hour_of_day` matters. SHAP tells you *how* it matters: rush hour amplifies price sensitivity by 40%. These are different business insights."

**API Endpoints:**
- `POST /api/v1/models/ensemble/train` — train RF or XGBoost with hyperparameters
- `GET /api/v1/models/ensemble/{run_id}/shap/global` — beeswarm plot data
- `GET /api/v1/models/ensemble/{run_id}/shap/{observation_id}` — individual waterfall
- `GET /api/v1/models/ensemble/{run_id}/shap/dependence/{feature}` — dependence plot

---

### M4 — Clustering & Station Segmentation

**Scenario:** "We have 80 stations but one pricing strategy. Are there natural groups of stations that should be managed differently? K-Means finds these groups from the data — but we decide what to do with them."

**Clustering Target:** Station-level features: `[elasticity_beta, avg_volume, n_competitors, station_type_encoded, urban_index, motorway_proximity, weekend_uplift_factor, morning_peak_ratio]`

**Concepts:**
- The unsupervised problem: no labels, just structure
- K-Means: centroid-based, Euclidean distance, convergence
- Choosing K: Elbow method (inertia), Silhouette score, business interpretation
- PCA for visualization: collapsing 8 dimensions to 2

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| K (number of clusters) | Slider | 2–10 | Watch cluster assignments change on the map; inertia and silhouette update |
| Feature selector | Checkboxes | 8 clustering features | Which dimensions to cluster on — a business judgment, not statistical |
| PCA component toggle | Buttons | PC1 vs. PC2 vs. PC3 | Explore variance structure |
| Cluster inspector | Click on map/chart | Any cluster | See profile card: avg elasticity, typical volume, competitor density |

**Visualizations:**
- **Elbow + Silhouette plot** side by side: "The statistics suggest 4-5 clusters. Here's how to choose."
- **Station map:** color-coded by cluster assignment, hover for station profile
- **PCA biplot:** stations in PC1-PC2 space, colored by cluster, with feature vectors showing which original dimensions each PC captures
- **Cluster profile heatmap:** clusters x features, z-scored — "Cluster 2 = high-elasticity, high-competition urban stations"
- **Silhouette diagram per cluster:** identifies poorly assigned stations (stations that don't fit any pricing strategy cleanly)

**Quality Gate:** Silhouette score > 0.35 for recommended K

**Break It:** Set K = 20 with only 80 stations — silhouette collapses, clusters become meaningless single-station assignments.

**Key Insight:** "K=5 produces archetypes: [Motorway-competitive], [Urban-congested], [Urban-captive], [Rural-stable], [Motorway-isolated]. These five have meaningfully different optimal pricing strategies — one size does not fit all."

**Gotcha:** "K-Means finds spherical clusters in Euclidean space. If your station types are genuinely different, the algorithm may just rediscover station_type. Check: does your clustering add information beyond the labels you already have?"

**API Endpoints:**
- `POST /api/v1/models/clustering/train` — run K-Means with selected K and features
- `GET /api/v1/models/clustering/{run_id}/profiles` — cluster profile data
- `GET /api/v1/models/clustering/{run_id}/pca` — PCA coordinates and loadings

---

### M5 — Anomaly Detection & Market Surveillance

**Scenario:** "Three times last year, a competitor station went temporarily offline. Volume surged 40% at nearby stations — but the pricing team didn't know until the next business day. An anomaly detection model runs 24/7."

**Three Detection Problems:**
1. **Price anomalies in competitor feed:** genuine price drop, data error, or temporary promotion? (Isolation Forest)
2. **Volume anomalies:** genuine demand spike/crash or sensor malfunction? (Statistical control chart + Local Outlier Factor)
3. **Concurrent competitor closure events:** spatial-temporal clustering of volume anomalies across nearby stations (DBSCAN)

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Isolation Forest contamination | Slider | 0.01–0.15 | Anomaly threshold — watch false positive rate vs. detection rate tradeoff |
| Rolling window size | Dropdown | 24h, 48h, 7d | Sensitivity vs. stability for statistical control charts |
| DBSCAN epsilon (spatial) and min_samples | Dual slider | eps: 0.5–5.0, min: 2–10 | Find genuine multi-station events vs. individual noise |
| Inject anomaly button | Button | — | Synthetically plant an anomaly (competitor closure, price error) — can the model catch it? |

**Visualizations:**
- **Anomaly score time series:** rolling score for price and volume, detected events flagged (hover shows station + magnitude)
- **Control chart (Shewhart X-bar):** upper/lower control limits, UCL/LCL breaches highlighted
- **Spatial anomaly cluster map:** when multiple stations show simultaneous volume surges, flag the geographic cluster
- **Anomaly event log table:** sortable, filterable — production monitoring dashboard format

**Quality Gate:** On labeled test set, F1 > 0.65 at configured contamination level

**Break It:** Set contamination = 0.15 — watch the model flag routine Friday afternoon volume spikes as anomalies.

**API Endpoints:**
- `POST /api/v1/models/anomaly/train` — train Isolation Forest with contamination parameter
- `POST /api/v1/models/anomaly/inject` — inject synthetic anomaly event
- `GET /api/v1/models/anomaly/{run_id}/events` — detected anomaly event log
- `GET /api/v1/models/anomaly/{run_id}/spatial` — DBSCAN spatial clusters

---

### M6 — Classical Time Series (Decomposition, ARIMA, Prophet)

**Scenario:** "Next week's diesel demand at a motorway station. We need a point forecast plus an 80% prediction interval to plan staffing and inventory."

**Concepts (in discovery order):**
1. **Decomposition (STL):** Trend + Seasonality + Residual
2. **Autocorrelation (ACF/PACF):** "Yesterday predicts today." — the lag structure of fuel volume
3. **ARIMA:** AR(1) -> ARIMA(p,d,q). Focus on parameter meaning, not estimation internals
4. **SARIMA:** Add seasonal component — crucial for fuel demand
5. **Prophet:** Decomposable model — changepoints, holiday effects, interpretable components

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| STL period selector | Dropdown | daily (24), weekly (168), annual | Force decomposition to find specific seasonality |
| AR/MA order (p, q) | Sliders | p: 0–5, q: 0–5, d: 0–2 | Watch AIC/BIC change; show white noise test on residuals |
| Prophet changepoint sensitivity | Slider | 0.01–0.5 | How readily Prophet detects trend changes |
| Prophet holiday toggle | Checkbox | on/off | Include/exclude holiday effects |
| Forecast horizon | Slider | 1 day – 6 weeks | Watch prediction intervals widen (uncertainty visualized honestly) |
| Station selector | Dropdown | All stations | Compare time series structure across station types |

**Visualizations:**
- **STL decomposition plot:** stacked original + trend + seasonal + residual (interactive zoom)
- **ACF/PACF correlograms:** with significance bands
- **Forecast fan chart:** quantiles (10th, 25th, 50th, 75th, 90th) for honest uncertainty
- **Residual diagnostics:** Ljung-Box test result, residual ACF, histogram
- **Walk-forward validation chart:** rolling 1-week-ahead forecasts over 3 months of test data vs. actuals
- **Component overlay (Prophet):** individual contributions of trend, weekly, annual, holiday

**Quality Gate:** ARIMA residuals pass Ljung-Box test (p > 0.05); walk-forward MAPE < 12%

**Break It:** Forecast 6 weeks ahead with ARIMA(1,0,0) — watch prediction intervals balloon to uselessness.

**Key Concept — Walk-Forward Validation:** Side-by-side comparison of in-sample fit (overly optimistic) vs. walk-forward MAE (honest). The gap is often shocking.

**API Endpoints:**
- `POST /api/v1/models/timeseries/decompose` — STL decomposition for station
- `POST /api/v1/models/timeseries/arima` — fit ARIMA with specified orders (async)
- `POST /api/v1/models/timeseries/prophet` — fit Prophet with configuration (async)
- `GET /api/v1/models/timeseries/{run_id}/forecast` — forecast results with intervals
- `GET /api/v1/models/timeseries/{run_id}/walkforward` — walk-forward validation results

---

### M7 — Sequence Models (Lag Features + LSTM)

**Scenario:** "ARIMA uses only the target variable's own history. But demand depends on yesterday's competitor price, last week's traffic, this morning's temperature. A supervised model over lag-engineered features can handle all of this."

**Two Approaches:**
1. **LightGBM on lag features:** Feature engineering as a skill. Lag-1 to lag-168 (hours), rolling means, rolling std, EWMA — show which lags matter via SHAP.
2. **LSTM:** Learns its own lag structure. No feature engineering needed, but less interpretable.

**Concepts:**
- Lag features: turning a sequence problem into a supervised problem
- The look-back window: how much history the model needs
- LSTM intuition: cell state = long-term memory, forget gate, input gate, output gate
- Teacher forcing vs. autoregressive inference
- Sequence-to-sequence vs. direct multi-step forecasting

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Lag feature selector | Checkboxes | Lag groups (1h, 24h, 168h, rolling_7d, ewma) | Watch SHAP importance update |
| LSTM units x layers | Dual slider | Units: 32–256, Layers: 1–3 | Architecture exploration |
| Look-back window | Dropdown | 24h, 48h, 168h, 336h | Impact on model accuracy and training time |
| Training progress | Live display | — | Loss per epoch — make training process visible |
| Multi-step forecast horizon | Slider | 1h – 7 days | Watch LSTM degrade gracefully vs. LightGBM |

**Visualizations:**
- **SHAP lag importance plot:** which lag matters most (t-1, t-24, t-168)? Rediscovers ACF structure via a different lens
- **Training/validation loss curves:** bias-variance tradeoff made dynamic — watch overfitting happen in real time
- **Prediction comparison:** LightGBM vs. LSTM vs. ARIMA on the same test window — who wins, and where?
- **LSTM cell state visualization (simplified):** for one training sequence, show forget gate activation at weekend transitions

**Quality Gate:** Walk-forward MAPE < 8% for best model

**Break It:** Set LSTM to 3 layers, 256 units with only 2000 training samples — severe overfitting visible in loss curves.

**API Endpoints:**
- `POST /api/v1/models/timeseries/lightgbm_lag` — train LightGBM with lag features
- `POST /api/v1/models/timeseries/lstm` — train LSTM (async, returns run_id)
- `GET /api/v1/models/timeseries/{run_id}/training_progress` — epoch-by-epoch loss for live display

---

### M8 — Temporal Fusion Transformer (TFT)

**Scenario:** "We need a single model that: (a) forecasts volume 7 days ahead with honest intervals, (b) handles known future inputs (promotions, holidays), (c) tells us which past time steps drove today's forecast, and (d) works across all stations without retraining."

**Why TFT:**
- Combines LSTM (local patterns) + attention (long-range dependencies) + variable selection networks
- Quantile outputs for calibrated prediction intervals
- Built-in feature importance: Variable Selection Network (which inputs matter) + Temporal Attention (which past time steps matter)
- Static covariates: station_type, region, n_competitors embedded via entity embeddings

**Architecture Concepts:**
- Three input types: static (station attributes), observed past (historical volume, prices), future known (holidays, planned promotions)
- Gated Residual Network: gating mechanism lets the model ignore irrelevant inputs entirely
- Temporal attention heads: interpretable by design (unlike vanilla attention)
- Quantile regression loss (pinball loss) for prediction intervals

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Forecast start date | Date selector | Test period range | See TFT predictions update for different periods |
| Station selector + overlay | Dropdown + toggle | All stations | Compare forecasts and attention patterns across station archetypes |
| Quantile toggle | Checkboxes | P10, P25, P50, P75, P90 | Understand what "80% prediction interval" means in practice |
| Input type ablation | Toggles | Static / Future known / All | Toggle off input types — watch performance degrade |
| Attention head selector | Dropdown | Heads 1–8 | Explore what different heads focus on |

**Visualizations:**
- **Multi-horizon forecast with quantile fan:** 1h, 4h, 24h, 7d ahead — uncertainty growth
- **Variable Selection Network heatmap:** stations x input_variables — feature weights per station type
- **Temporal attention heatmap:** time_steps x attention_weight — "the model pays most attention to the same hour last week and 3 days ago"
- **Calibration plot:** does the P90 interval actually contain 90% of actuals?
- **Error decomposition:** MAPE by forecast horizon + by station type

**Quality Gate:** Walk-forward MAPE < 8%; P90 calibration error < 5%

**Break It:** Remove holiday features — watch holiday-period forecasts fail catastrophically.

**Key Message:** "TFT is more complex but earns it. The attention maps are not just performance — they're a new type of insight into what drives demand patterns."

**API Endpoints:**
- `POST /api/v1/models/timeseries/tft` — train TFT (async, longest training time ~5-20 min)
- `GET /api/v1/models/timeseries/{run_id}/attention` — temporal attention weights
- `GET /api/v1/models/timeseries/{run_id}/variable_selection` — variable selection network weights
- `GET /api/v1/models/timeseries/{run_id}/calibration` — quantile calibration data

---

### M9 — Linear Programming & Margin Maximization

**Scenario:** "Given our elasticity estimates, the LP asks: What price should each of the 80 stations set tomorrow to maximize total gross margin, subject to: price band constraints, competitive distance constraints, and minimum volume requirements?"

**Formulation:**

```
Maximize:    SUM_i [margin_rate_i(p_i) * volume_i(p_i)]
Subject to:
  (1)  P_floor_i <= p_i <= P_ceil_i              # price band (regulatory/brand)
  (2)  p_i - min_comp_price_i <= max_gap_i       # competitive constraint
  (3)  volume_i(p_i) >= min_volume_i             # operational minimum (staffing)
  (4)  SUM_i volume_i(p_i) >= total_volume_target # aggregate commitment
```

Note: `volume_i(p_i)` is non-linear (from elasticity model) — this is actually a nonlinear programme. Teach both LP (linearized demand) and NLP (actual demand curve). Show where linearization fails.

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Price band width | Slider | +/-1ct to +/-8ct | Tighter bands = less optimizer freedom = lower margin gain |
| Volume floor | Slider | 50%–90% of baseline | Watch LP solution change as volume protection increases |
| Aggregate volume target | Slider | % of no-action baseline | The core business tradeoff dial |
| Regional pricing constraint | Toggle + slider | on/off, Xct range | Force regional price consistency |
| Solver mode | Dropdown | LP / NLP / Heuristic greedy | Compare solution quality and solve time |

**Visualizations:**
- **Price recommendation map:** station map with recommended delta_price per station (color scale: dark red = deep cut, dark green = price up)
- **Margin frontier chart:** Pareto front of Total Margin vs. Total Volume — LP traces this frontier as volume constraint relaxes
- **Shadow price table:** which constraints are binding? A binding volume floor means: "an additional liter of volume requirement costs X EUR in foregone margin"
- **Sensitivity analysis tornado:** which input assumptions (elasticity estimates) most change the optimal solution?

**Quality Gate:** Optimizer converges within 30s; solution satisfies all constraints

**Break It:** Set all volume floors to 90% of current baseline — watch the optimizer produce near-zero margin improvement (no room to maneuver).

**Key Message:** "The LP doesn't decide. It reveals the tradeoff frontier so you can decide. The shadow prices tell you the monetary value of relaxing each constraint — powerful input for negotiating operational commitments."

**API Endpoints:**
- `POST /api/v1/optimization/solve` — solve LP/NLP with constraints
- `GET /api/v1/optimization/{run_id}/shadow_prices` — binding constraints and shadow prices
- `GET /api/v1/optimization/{run_id}/frontier` — Pareto frontier data points
- `GET /api/v1/optimization/{run_id}/sensitivity` — tornado chart data

---

### M10 — Multi-Armed Bandits & Pricing Experiments

**Scenario:** "We believe -2ct vs. competitor will outperform parity at urban stations. But we don't know. A/B testing is slow and wastes 50% of traffic on the worse option. Thompson Sampling learns and earns simultaneously."

**MAB Framing:**
- **Arms:** discrete price levels {-4ct, -2ct, parity, +2ct, +4ct} vs. nearest competitor
- **Reward:** observed gross profit per hour (noisy, non-stationary)
- **Challenge:** demand is non-stationary (seasonality, competitive moves) — use sliding-window Thompson Sampling

**Algorithms Compared:**
1. **epsilon-greedy (baseline):** explore randomly 10% of the time
2. **UCB1:** optimistic in the face of uncertainty — choose arm with highest upper confidence bound
3. **Thompson Sampling (Beta-Binomial for discretized profit):** Bayesian posterior over arm quality, sample to choose
4. **f-dsw Thompson Sampling:** sliding window variant for non-stationarity

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Stations / horizon | Sliders | 10–200 hours | Watch cumulative regret curves diverge |
| True elasticity | Slider | -3.0 to -0.5 | Change the underlying "truth" — does Thompson Sampling adapt? |
| Non-stationarity toggle | Button | on/off | Introduce competitor price drop mid-experiment |
| Prior strength | Slider | 1–100 | Encode existing pricing knowledge vs. let data speak |
| Run simulation | Button | — | Stochastic — run 20 times, show distribution of outcomes |

**Visualizations:**
- **Arm selection heatmap:** hours x arms — which algorithm explores which arms and when
- **Cumulative regret plot (log scale):** UCB and TS sub-linear; epsilon-greedy linear
- **Posterior distribution evolution (Thompson Sampling):** animated Beta distributions narrowing as evidence accumulates
- **Exploration-exploitation chart:** fraction of arms selected for explore vs. exploit at each time step
- **Revenue comparison table:** expected revenue under epsilon-greedy vs. Thompson Sampling over 30 days

**Quality Gate:** Thompson Sampling cumulative regret sub-linear; demonstrably outperforms epsilon-greedy in revenue

**Break It:** Set high non-stationarity with a fixed (non-sliding-window) Thompson Sampling — watch it fail to adapt to a mid-experiment competitor price drop.

**Key Insight:** "Thompson Sampling doesn't just find the best price — it finds it while losing less revenue during the search. The regret is sub-linear: the algorithm gets smarter faster."

**API Endpoints:**
- `POST /api/v1/bandit/simulate` — run bandit simulation with algorithm config
- `GET /api/v1/bandit/{run_id}/regret` — cumulative regret curves per algorithm
- `GET /api/v1/bandit/{run_id}/posteriors` — posterior distributions over time

---

### M11 — Markov Decision Processes & Q-Learning

**Scenario:** "Unlike the bandit, a pricing agent must think sequentially. If I cut price now, I may trigger a price war that costs me for months. MDP/Q-Learning frames pricing as a game against time and competitors."

**MDP Formulation:**

```
State S:   (price_gap_discretized,       # -4, -2, 0, +2, +4 vs. competitor
            volume_index_discretized,    # low/mid/high vs. 4-week average
            time_of_week_bucket,         # 9 buckets: hour x day-type
            competitor_trend)            # falling / stable / rising

Action A:  {keep, -2ct, -1ct, +1ct, +2ct}

Reward R:  gross_profit_t - alpha * |price_change_t|   # profit minus volatility penalty

Transition P(s'|s,a): deterministic base + stochastic competitor response
```

**State space:** 5 x 3 x 9 x 3 = 405 states — small enough for tabular Q-learning, large enough for interesting policies.

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Discount factor (gamma) | Slider | 0.5–0.99 | Myopic vs. far-sighted agent — watch policy change |
| Training episodes | Slider | 100–10,000 | Q-table convergence animation |
| Epsilon decay rate | Slider | 0.99–0.999 | Fast vs. slow exploration |
| Competitor response model | Dropdown | Cooperative / Neutral / Aggressive | How does optimal policy change? |
| State selector | Dropdown/grid | Any MDP state | See Q-values for all actions — understand learned policy |

**Visualizations:**
- **Q-value convergence animation:** selected state-action Q-values stabilizing over episodes
- **Policy heatmap:** volume_state x price_gap_state -> recommended action (color-coded) per time-of-week bucket
- **Reward trajectory:** episode reward over training — non-monotone but trending up
- **Value function surface (3D):** V(s) as surface over price_gap x volume_state

**Quality Gate:** Reward trajectory converging within 3000 episodes

**Break It:** Set gamma = 0.99 in a volatile competitor environment — agent learns an unstable greedy policy that triggers price wars.

**API Endpoints:**
- `POST /api/v1/rl/qlearning/train` — train tabular Q-learning (async)
- `GET /api/v1/rl/{run_id}/policy` — learned policy heatmap data
- `GET /api/v1/rl/{run_id}/convergence` — Q-value convergence data
- `GET /api/v1/rl/{run_id}/value_surface` — value function surface data

---

### M12 — Deep Q-Networks (DQN)

**Scenario:** "The tabular Q-table breaks when the state space becomes continuous or high-dimensional. DQN replaces the table with a neural network — enabling pricing policies over richer state representations."

**Why DQN Over Q-Learning:**
- State: raw price/volume/time features (continuous) instead of discretized buckets
- Neural network approximates Q(s,a) — generalizes across nearby states
- Experience replay: breaks temporal correlation in training data
- Target network: stabilizes training

**Concepts (business-appropriate depth):**
- Neural Q-function: "the network learns which states are valuable, without us defining the buckets"
- Experience replay buffer: "the agent learns from a shuffled sample of past experiences — like studying from a shuffled flashcard deck"
- Target network freeze: "we update the target every N steps to prevent circular learning"

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Network architecture | Sliders | Hidden layers: 1–3, Units: 32–256 | Capacity vs. stability |
| Replay buffer size | Slider | 500–50,000 | Effect on training stability |
| Target network update frequency | Slider | 10–500 steps | Stability vs. lag tradeoff |
| Environment difficulty | Dropdown | Easy (stable) / Hard (reactive) / Adversarial (memory) | Difficulty levels |

**Visualizations:**
- **DQN training curve:** episode rewards + smoothed trend (much noisier than Q-learning — address directly)
- **Q-value landscape evolution (2D state slice):** animated — neural network's value estimate sharpens over training
- **Policy comparison:** DQN policy vs. Q-learning policy vs. hand-crafted rule-based policy — quantified performance gap

**Quality Gate:** DQN reward trajectory converging; performance within 10% of tabular Q-learning on discretized version

**Break It:** Set replay buffer to 500 with target update every 10 steps — catastrophically unstable training.

**API Endpoints:**
- `POST /api/v1/rl/dqn/train` — train DQN (async, 30–300s)
- `GET /api/v1/rl/{run_id}/landscape` — Q-value landscape animation frames
- `GET /api/v1/rl/{run_id}/comparison` — DQN vs. Q-learning vs. rule-based performance

---

### M13 — Neural Network Architecture Explorer

**Scenario:** "Before LLMs, before attention, there was the feedforward neural network. Every modern AI system has MLP layers at its core. Understanding how activation functions, depth, and width interact is foundational."

**Concepts (interactive-first):**
- Activation functions: ReLU, sigmoid, tanh, GELU — as function plots, then as effects on decision boundaries
- Universal approximation theorem: a 2-layer network can approximate any function — but practical convergence requires depth
- Vanishing/exploding gradients: what happens with sigmoid at depth 10
- Batch normalization: why it helps
- Embeddings for categorical features: why one-hot encoding at scale fails

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Architecture builder | Visual layer controls | Add/remove layers, resize width | Parameter count updates in real time |
| Activation function per layer | Dropdowns | ReLU, GELU, Tanh, Sigmoid | Watch gradient magnitudes differ |
| Training visualizer | Animation | Epochs 0–200 | 2D decision boundary forms epoch by epoch |
| Embedding dimension | Slider | 2–64 | PCA of learned station embeddings after training |

**Visualizations:**
- **Decision boundary animation (2D):** price_gap x hour -> threat/no-threat classification — boundary sharpens over epochs
- **Gradient flow diagram:** color-coded backpropagation arrows — vanishing gradient made visible
- **Embedding PCA:** 2D PCA of learned station embeddings — "the network learned that motorway stations cluster together, without being told"
- **Activation distribution histograms per layer:** dead neurons (zeros) with ReLU, saturation with sigmoid

**Quality Gate:** N/A (exploration module)

**Break It:** Use sigmoid activation with 10+ layers — watch vanishing gradients kill learning entirely.

**API Endpoints:**
- `POST /api/v1/neural/mlp/train` — train MLP with specified architecture
- `GET /api/v1/neural/{run_id}/decision_boundary` — decision boundary animation frames
- `GET /api/v1/neural/{run_id}/gradients` — gradient magnitude per layer
- `GET /api/v1/neural/{run_id}/embeddings` — learned embedding PCA coordinates

---

### M14 — Self-Attention for Tabular Data (FT-Transformer)

**Scenario:** "In tabular pricing data, the interaction between `price_gap` and `station_type` is crucial — XGBoost discovers this through splits. FT-Transformer discovers it through attention: every feature attends to every other feature."

**Architecture (step by step):**
1. **Feature Tokenization:** Each feature (numeric + categorical) -> dense vector (the "token")
2. **[CLS] Token:** Learnable token appended to the sequence — aggregates all feature information
3. **Transformer layers:** Multi-head self-attention over feature tokens. Q, K, V matrices over features, not time steps
4. **Prediction head:** Linear layer on [CLS] embedding -> volume prediction

**Key Difference from Sequential Transformer:** "In NLP, attention is over word positions. Here, attention is over *features*. The model learns which features interact most."

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Number of attention heads | Slider | 1–8 | Watch feature interaction maps change |
| Attention visualization sample | Dropdown/search | Any station-hour observation | Full features x features attention matrix |
| Depth (transformer layers) | Slider | 1–6 | More layers -> more complex feature interactions |
| Compare mode | Toggle | FT-Transformer vs. XGBoost vs. MLP | Validation MAE + inference time comparison |

**Visualizations:**
- **Feature-Feature attention heatmap:** features x features — "price_gap strongly attends to station_type and hour_of_day"
- **CLS token attention over features:** which features does the final [CLS] token weight most?
- **Performance comparison chart:** FT-Transformer vs. XGBoost vs. MLP vs. Linear — with training data size on x-axis (FT-Transformer needs more data to win)
- **Calibration plots:** all models on same test set

**Quality Gate:** N/A (comparison module — the lesson is knowing when each model wins)

**Break It:** Train FT-Transformer on only 5k samples — watch XGBoost significantly outperform it.

**Key Message:** "FT-Transformer is not always better than XGBoost. It wins when feature interactions are complex and you have >50k training examples. Knowing when to use which is AI Fluency."

**API Endpoints:**
- `POST /api/v1/neural/ft_transformer/train` — train FT-Transformer (async)
- `GET /api/v1/neural/{run_id}/attention_map/{observation_id}` — feature-feature attention matrix
- `GET /api/v1/neural/{run_id}/comparison` — multi-model performance comparison

---

### M15 — Transformer Zoo: BERT, GPT, T5 — A Map

**Scenario:** "ChatGPT, Claude, Gemini, Llama — they're all 'transformers', but built for different purposes. This module builds the conceptual map for asking the right question: 'what type of model is right for this pricing task?'"

**Not a coding module — a conceptual one with rich visuals.**

**Three Transformer Families:**

| Type | Architecture | Training | Pricing Use Cases |
|---|---|---|---|
| **Encoder** (BERT) | Bidirectional attention | Masked language modeling | Classify competitor press releases; NER on pricing reports |
| **Decoder** (GPT) | Causal (left-to-right) attention | Next-token prediction | Generate pricing narratives, draft briefings |
| **Encoder-Decoder** (T5) | Full bidirectional + cross-attention | Seq2Seq (text -> text) | Summarize competitor reports, translate strategies |

**Interactive Visualization:** Animated attention pattern diagram toggling between:
- BERT: every token sees every other token (bidirectional arrows)
- GPT: each token only sees past tokens (one-directional arrows, masked future)
- T5: encoder arrows + cross-attention between encoder and decoder outputs

**Scale Timeline:** GPT-1 (117M params) to GPT-4 (~1.8T params) — parameter count on log scale, capability jumps annotated with pricing-relevant examples.

**Quality Gate:** N/A (conceptual module)

**API Endpoints:**
- `GET /api/v1/llm/transformer_zoo/architectures` — static architecture diagram data
- `GET /api/v1/llm/transformer_zoo/scale_timeline` — model scale timeline data

---

### M16 — LLM Architecture, Capabilities & Limits

**Scenario:** "Before you ask an LLM to analyze your pricing data, you need to know what it actually does when it processes your prompt. This module demystifies the LLM black box — and, crucially, where it breaks."

**Interactive Concepts:**
1. **Tokenization visualizer:** type a pricing prompt, see it tokenized live. Station IDs might be 4 tokens. Token counting helps predict cost and context limits.
2. **Context window:** sliding window visualization showing what the LLM "sees" at each generation step.
3. **Temperature and sampling:** interactive text generation with temperature 0 (deterministic) to temperature 2 (creative/chaotic). Applied to pricing: "What's the optimal price for this station tomorrow?" — responses vary.
4. **Hallucination mechanics:** deliberately obscure question about a fictional station. Confident-sounding wrong answers arise from token probability distribution.
5. **Knowledge cutoff:** LLM doesn't know about a competitor move from last week. RAG solves this.

**Capabilities/Limits Table:**

| Task | LLM Alone | LLM + Tools | Your ML Model |
|---|---|---|---|
| Predict volume at a station tomorrow | No — needs trained demand model | Yes, with API call to ML model | Best |
| Summarize last month's pricing performance | Yes, if data provided in context | Yes, with data retrieval | Not applicable |
| Write pricing strategy memo | Yes, native capability | Yes, same | Not applicable |
| Calculate elasticity from raw data | No — arithmetic unreliable | Yes, with Python execution tool | Best |
| Explain a SHAP plot in plain language | Yes, with image/data input | Yes | Not applicable |
| Detect competitor pricing anomaly in real-time | No — no live data | Yes, with monitoring feed | Best |

**Quality Gate:** N/A (conceptual module)

**API Endpoints:**
- `POST /api/v1/llm/tokenize` — tokenize input text, return tokens and count
- `POST /api/v1/llm/generate` — generate text with temperature parameter
- `GET /api/v1/llm/capabilities` — static capabilities/limits table data

---

### M17 — Prompt Engineering & RAG

**Scenario:** "You have years of pricing reports, competitor intelligence briefs, and strategy memos. A RAG system makes this knowledge instantly searchable and synthesizable — for anyone, in natural language."

**Prompt Engineering Patterns (with live playground):**
1. **Role + Context + Task + Constraints (RCTC):** type in playground, see quality difference
2. **Chain-of-thought:** add "Think step by step" — test if reasoning improves for pricing scenarios
3. **Few-shot examples:** 2 example input/outputs in the prompt — show quality jump for structured outputs
4. **Output format specification:** "Return as JSON with keys: recommendation, confidence, reasoning, risks"
5. **Self-consistency:** run same prompt 5x at temperature 0.8, show distribution of answers

**RAG Architecture:**

```
Pricing Reports (PDF/text documents)
        |
        v
  Text Chunking (500-token chunks with overlap)
        |
        v
  Embedding Model (sentence-transformers)
        |
        v
  Vector Store (FAISS)
        |
        v
  Query -> Embed -> Similarity Search -> Top-K Chunks
        |
        v
  LLM + Retrieved Context -> Answer
```

**Interactive Controls:**

| Control | Type | Range | Effect |
|---|---|---|---|
| Chunk size | Slider | 100–2000 tokens | Too small = no context; too large = diluted retrieval |
| Top-K retrieved chunks | Slider | 1–10 | More context helps up to a point |
| Query editor | Text input | Free text | Type pricing question, see which chunks are retrieved (highlighted), see final LLM answer |
| Embedding space visualizer | Toggle | on/off | UMAP 2D of document chunks, colored by source document |

**Quality Gate:** N/A (demonstration module)

**Break It:** Set chunk size = 100 tokens — watch retrieval return fragments too small to provide meaningful context.

**API Endpoints:**
- `POST /api/v1/llm/prompt/generate` — prompt playground with RCTC scaffolding
- `POST /api/v1/llm/rag/query` — RAG pipeline: embed query, retrieve, synthesize
- `GET /api/v1/llm/rag/embeddings` — UMAP coordinates of document chunks
- `POST /api/v1/llm/rag/configure` — set chunk size, top-K, overlap

---

### M18 — Synthesis: Pricing AI System Design

**Scenario:** "You've seen every building block. Now: what does a production-grade AI-augmented pricing system actually look like? What do you own, what does the machine own, and where does it break?"

**Full Stack Architecture Diagram:**

```
DATA LAYER
  +-- Real-time price feeds (public price APIs, internal point-of-sale)
  +-- Historical warehouse (3yr+ of prices, volumes, COGS)
  +-- External: weather, events, traffic, crude oil
  +-- Feature store (pre-computed lag features, elasticities, cluster assignments)

MODEL LAYER
  +-- Demand model (XGBoost/TFT): volume forecast + prediction intervals
  +-- Elasticity engine (panel regression): live coefficient estimates
  +-- Anomaly detector (Isolation Forest): real-time alert stream
  +-- Optimizer (NLP): margin-maximizing price recommendations

INTELLIGENCE LAYER
  +-- LLM narration: explain model outputs in plain language
  +-- RAG: retrieve relevant historical context for each decision
  +-- Agent orchestrator: monitor -> detect -> analyze -> recommend pipeline

HUMAN LAYER (non-negotiable)
  +-- Manager review dashboard: approve/reject/modify recommendations
  +-- Audit trail: every recommendation, its rationale, and human decision
  +-- Override mechanisms: human always has authority to override
  +-- Model monitoring: drift alerts, performance degradation flags
```

**Key Discussion Points:**
- **Model drift:** demand elasticity changes during market crises. How do you detect when your model is stale?
- **Data quality as the first failure mode:** anomaly detection exists partly to catch data errors before they corrupt the demand model
- **The governance gap:** regulatory frameworks increasingly require high-risk AI systems to have human oversight, audit trails, and incident reporting
- **Automation vs. judgment:** automate = data aggregation, pattern detection, forecast computation, option generation. Human judgment = strategy, exception handling, stakeholder communication

**Quality Gate:** N/A (synthesis module)

**API Endpoints:**
- `GET /api/v1/synthesis/architecture` — full system architecture diagram data
- `GET /api/v1/synthesis/governance_checklist` — governance requirements checklist

---

## 4. Backend Architecture

### 4.1 Project Structure

```
backend/
+-- main.py                    # FastAPI app, CORS, router mounting
+-- config.py                  # settings, paths, model params
+-- requirements.txt
|
+-- data/
|   +-- generator.py           # synthetic data generation engine
|   +-- cache.py               # in-memory dataset registry
|   +-- schemas.py             # Pydantic models for all data responses
|
+-- routers/
|   +-- data.py               # /api/v1/data/*
|   +-- eda.py                # /api/v1/eda/*
|   +-- regression.py         # /api/v1/models/regression/*
|   +-- classification.py     # /api/v1/models/classification/*
|   +-- clustering.py         # /api/v1/models/clustering/*
|   +-- timeseries.py         # /api/v1/models/timeseries/*
|   +-- optimization.py       # /api/v1/optimization/*
|   +-- bandit.py             # /api/v1/bandit/*
|   +-- rl.py                 # /api/v1/rl/*
|   +-- neural.py             # /api/v1/neural/*
|   +-- llm.py                # /api/v1/llm/*
|
+-- models/
|   +-- regression.py         # elasticity estimation, panel model
|   +-- classification.py     # threat detection, strategy classifier
|   +-- clustering.py         # K-Means, PCA, DBSCAN
|   +-- anomaly.py            # Isolation Forest, LOF
|   +-- timeseries.py         # ARIMA, Prophet, LightGBM-lag, LSTM
|   +-- tft.py                # Temporal Fusion Transformer
|   +-- optimization.py       # PuLP LP, scipy NLP
|   +-- bandit.py             # Thompson Sampling, UCB, epsilon-greedy
|   +-- rl_tabular.py         # Q-Learning
|   +-- rl_dqn.py             # DQN (PyTorch)
|   +-- mlp.py                # MLP architecture explorer
|   +-- ft_transformer.py     # FT-Transformer
|
+-- utils/
|   +-- chart_helpers.py      # Plotly figure factory functions
|   +-- metrics.py            # model evaluation utilities
|   +-- shap_helpers.py       # SHAP computation wrappers
|
+-- services/
|   +-- llm_client.py         # LLM provider abstraction (Claude API default)
|   +-- rag_pipeline.py       # document chunking, embedding, retrieval
|   +-- run_manager.py        # async run lifecycle (create, poll, retrieve)
|
+-- tests/
    +-- test_data.py
    +-- test_models.py
    +-- test_routers.py
```

### 4.2 API Design Patterns

**Async Training Pattern (all long-running models):**

```
POST /api/v1/models/{module}/{model_type}
Request:  { "features": [...], "hyperparams": {...}, "station_filter": "..." }
Response: { "run_id": "run_20240310_142301", "status": "training", "est_seconds": 45 }

GET /api/v1/runs/{run_id}/status
Response: { "run_id": "...", "status": "complete", "progress": 100, "metrics": {...} }

GET /api/v1/runs/{run_id}/results
Response: { "forecasts": [...], "metrics": {...}, "figures": {...} }
```

**Chart Response Pattern (all figure endpoints):**

```
GET /api/v1/eda/decompose?station_id=STN_042&period=168
Response: { "figure": "{...plotly json...}", "components": {"trend": [...], ...} }
```

All figures returned as Plotly JSON (`fig.to_json()`), ensuring frontend portability: Dash uses `dcc.Graph(figure=json.loads(response['figure']))` and React uses `<Plot data={...} layout={...} />` — identical data.

**Stateless Design:** All model state stored server-side by `run_id`. Clients hold only run IDs.

### 4.3 Performance Characteristics

| Model | Expected Training Time | Strategy |
|---|---|---|
| Linear Regression | < 1s | Synchronous endpoint |
| XGBoost (5k rows) | 2–5s | Synchronous, cached |
| K-Means | 1–3s | Synchronous |
| ARIMA (auto-grid) | 10–30s | Async + polling |
| Prophet | 15–45s | Async + polling |
| LSTM | 30–120s | Async + WebSocket progress |
| TFT | 5–20 min | Async + background task + WebSocket |
| DQN | 30–300s | Async + WebSocket (episode stream) |
| Q-Learning | 5–30s | Async + WebSocket |
| LP/NLP Optimizer | 1–30s | Synchronous for LP, async for NLP |

---

## 5. Frontend Architecture

### 5.1 Project Structure

```
frontend/
+-- app.py                    # Dash app initialization, routing
+-- requirements.txt
|
+-- pages/
|   +-- home.py               # Module Hub: overview, progress tracker
|   +-- m00_foundations.py
|   +-- m01_regression.py
|   +-- m02_classification.py
|   +-- m03_ensembles.py
|   +-- m04_clustering.py
|   +-- m05_anomaly.py
|   +-- m06_timeseries_classical.py
|   +-- m07_timeseries_sequence.py
|   +-- m08_tft.py
|   +-- m09_optimisation.py
|   +-- m10_bandits.py
|   +-- m11_rl_qlearning.py
|   +-- m12_dqn.py
|   +-- m13_nn_explorer.py
|   +-- m14_ft_transformer.py
|   +-- m15_transformer_zoo.py
|   +-- m16_llm.py
|   +-- m17_prompt_rag.py
|   +-- m18_synthesis.py
|
+-- components/
|   +-- navbar.py             # Module navigation + progress indicators
|   +-- story_card.py         # Pricing story frame (reusable dark card)
|   +-- metric_cards.py       # KPI metric display
|   +-- chart_panel.py        # Chart + explanation side-by-side
|   +-- control_panel.py      # Parameter sliders/dropdowns wrapper
|   +-- diagnostics_panel.py  # Residuals, metrics, alerts
|   +-- station_map.py        # Reusable geographic station map
|   +-- progress_tracker.py   # Module completion tracker
|   +-- training_status.py    # Async training progress bar
|   +-- assessment.py         # 3-question module assessment
|
+-- callbacks/
|   +-- regression_callbacks.py
|   +-- classification_callbacks.py
|   +-- clustering_callbacks.py
|   +-- ... (one file per module with complex callback logic)
|
+-- assets/
|   +-- custom.css            # Dash theme overrides
|   +-- colors.py             # Brand color constants
|
+-- api_client.py             # All FastAPI calls centralized here
```

### 5.2 Page Layout Template

Every module page follows this exact layout:

```
+-----------------------------------------------------------+
|  NAVBAR  [Module 0 | Module 1 | ... | Module 18]          |
+-----------------------------------------------------------+
|  STORY CARD  -- pricing scenario framing (dark card)       |
+----------------------+------------------------------------+
|  CONTROLS            |  PRIMARY CHART                     |
|  (sticky sidebar)    |                                    |
|  - sliders           |  (2/3 width, responsive)           |
|  - dropdowns         |                                    |
|  - checkboxes        +------------------------------------+
|  - Run button        |  SECONDARY CHARTS (tabs)           |
|  - Training status   |  Tab 1: Diagnostics                |
|                      |  Tab 2: Explainability             |
|  KPI CARDS           |  Tab 3: Comparison                 |
|  (below controls)    |                                    |
+----------------------+------------------------------------+
|  INSIGHTS PANEL  -- key takeaway + gotcha (collapsible)    |
+-----------------------------------------------------------+
|  ASSESSMENT  -- 3-question check (expandable section)      |
+-----------------------------------------------------------+
```

### 5.3 Callback Architecture

Callbacks are **thin** — they call `api_client.py` and update component IDs. All business logic lives in the FastAPI backend.

```python
@app.callback(
    Output("elasticity-chart", "figure"),
    Output("regression-metrics", "children"),
    Input("run-regression-btn", "n_clicks"),
    State("feature-selector", "value"),
    State("station-type-selector", "value"),
    prevent_initial_call=True
)
def run_regression(n_clicks, features, station_type):
    result = api_client.post("/models/regression/train", {
        "features": features,
        "station_type": station_type
    })
    return json.loads(result["elasticity_figure"]), render_metrics(result["metrics"])
```

This thin-callback pattern makes React migration straightforward: replace callback with `useEffect` + `fetch`.

### 5.4 State Management

`dcc.Store` components for cross-page state:

| Store ID | Purpose | Persistence |
|---|---|---|
| `store-dataset-params` | Current synthetic data configuration | Session |
| `store-run-ids` | Mapping of module -> latest run_id | Session |
| `store-progress` | Module completion states | Local storage (`storage_type="local"`) |

### 5.5 Async Training Handling

For long-running models (LSTM, TFT, DQN):

1. `dcc.Interval` polls `/runs/{run_id}/status` every 2 seconds
2. Progress bar component updates on each poll
3. Interval disabled once `status == "complete"`
4. Final results fetched from `/runs/{run_id}/results`

---

## 6. React/Next.js Portability

### 6.1 What Stays the Same

- All backend logic (FastAPI, models, data generation)
- All chart data (Plotly JSON format)
- All API contracts (Pydantic schemas)

### 6.2 Component Mapping

| Dash Component | React Equivalent |
|---|---|
| `dcc.Graph` | `react-plotly.js` `<Plot>` |
| `dcc.Slider` | `shadcn/ui` Slider |
| `dcc.Dropdown` | `shadcn/ui` Select |
| `dcc.Checklist` | `shadcn/ui` Checkbox group |
| `dcc.Tabs` | `shadcn/ui` Tabs |
| `dcc.Store` | React `useState` / Zustand |
| `dcc.Interval` | `setInterval` + `useEffect` |
| Callback graph | `useEffect` + `fetch` hooks |
| Multi-page routing | Next.js `app/` directory routing |

### 6.3 Migration Strategy

1. **Dash Phase:** Full functionality in Dash with clean thin-callback architecture
2. **Hybrid Phase:** Extract individual module pages as React components (start with M1 Regression as proof-of-concept). Both Dash and React call the same FastAPI backend.
3. **React Phase:** Full Next.js app with Plotly.js for charts, shadcn/ui for controls, Zustand for global state

---

## 7. Assessment & Quality Framework

### 7.1 Per-Module Assessment Format

Each module ends with a 3-question interactive check:

| Question Type | Format | Example (M1 Regression) |
|---|---|---|
| **Conceptual** | Multiple choice with instant feedback | "Why does the log-log regression form give you elasticity directly as a coefficient?" |
| **Applied** | Numeric input requiring calculation | "You estimate beta1 = -1.8 with CI [-1.4, -2.2]. The break-even for a -4ct price cut requires X% volume uplift. What range of uplifts is consistent with this estimate?" |
| **Critical Thinking** | Free text + model answer revealed | "A competitor just dropped price 5ct. Your model recommends matching immediately. What would make you NOT follow the recommendation?" |

### 7.2 Model Quality Gates

| Model | Minimum Quality Gate |
|---|---|
| Regression | R-squared > 0.5 on test set; residual ACF not significantly autocorrelated |
| Classification | ROC-AUC > 0.72; calibration error < 0.05 |
| Clustering | Silhouette score > 0.35 for recommended K |
| ARIMA | Ljung-Box p > 0.05 on residuals (white noise) |
| LSTM / TFT | Walk-forward MAPE < 8% |
| Q-Learning | Reward trajectory converging within 3000 episodes |

When a model fails a quality gate (due to user-selected hyperparameters), the UI displays a **yellow warning card**: "This model configuration may not generalize well. Here's what the diagnostic is telling you, and why it matters."

### 7.3 "Break It" Failure Modes Summary

| Module | Failure Mode | What Happens |
|---|---|---|
| M0 | Polynomial degree = 12 | Test MSE explodes while train MSE approaches zero |
| M1 | Omit competitor gap feature | Multicollinearity destabilizes coefficients |
| M2 | Tree depth = 15 | Train accuracy 99%, test accuracy < 70% |
| M3 | learning_rate = 0.5, n_estimators = 500 | Boosting memorizes noise |
| M4 | K = 20 on 80 stations | Silhouette collapses, single-station clusters |
| M5 | Contamination = 0.15 | Routine patterns flagged as anomalies |
| M6 | ARIMA(1,0,0) at 6-week horizon | Prediction intervals balloon to uselessness |
| M7 | 3-layer 256-unit LSTM on 2k samples | Severe overfitting in loss curves |
| M8 | Remove holiday features | Holiday-period forecasts fail catastrophically |
| M9 | All volume floors at 90% | Near-zero margin improvement |
| M10 | Fixed TS with high non-stationarity | Fails to adapt to mid-experiment changes |
| M11 | gamma = 0.99 in volatile environment | Unstable greedy policy, triggers price wars |
| M12 | Buffer = 500, target update = 10 | Catastrophically unstable training |
| M13 | Sigmoid at depth 10+ | Vanishing gradients kill learning |
| M14 | FT-Transformer on 5k samples | XGBoost significantly outperforms |
| M17 | Chunk size = 100 tokens | Retrieval returns useless fragments |

---

## 8. Technology Stack

### 8.1 Dependencies

```
# Core Framework
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.0.0

# Data
pandas>=2.0.0
numpy>=1.26.0

# ML - Supervised
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.3.0
shap>=0.44.0

# ML - Time Series
statsmodels>=0.14.0
prophet>=1.1.5

# Deep Learning
torch>=2.2.0
pytorch-forecasting>=1.0.0       # TFT
rtdl>=0.0.13                     # FT-Transformer

# Optimization
pulp>=2.7.0                      # LP solver
scipy>=1.12.0                    # NLP optimization

# LLM / RAG
anthropic>=0.18.0                # Claude API client
faiss-cpu>=1.8.0                 # Vector store
sentence-transformers>=2.6.0     # Embedding model

# Visualization
plotly>=5.19.0

# Async Task Queue (for heavy models)
celery>=5.3.0
redis>=5.0.0

# Frontend (Dash)
dash>=2.16.0
dash-bootstrap-components>=1.5.0
```

### 8.2 Resolved Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **LLM Backend** | Anthropic Claude API via `llm_client.py` abstraction | Best structured reasoning; interface allows provider swap |
| **Async Task Queue** | FastAPI BackgroundTasks (light) + Celery/Redis (heavy) | Avoids infra overhead for simple models; Celery reliable for LSTM/TFT/DQN |
| **Hosting** | Docker Compose (FastAPI + Redis containers) | Simplest self-hosted path; same containers work on any cloud platform |

### 8.3 Open Decisions (Recommendations)

| Decision | Recommendation | Alternative | When to Revisit |
|---|---|---|---|
| **Vector Store** | FAISS (in-memory) | Chroma (persistent), Weaviate (managed) | If RAG corpus grows beyond single-server memory |
| **Station Map Tiles** | Mapbox (free tier token) | OpenStreetMap, Plotly `scatter_geo` | If Mapbox token management is problematic |
| **TFT Library** | PyTorch Forecasting | Darts, custom | If PyTorch Forecasting maintenance stalls |
| **Authentication** | None (open access) | Session-based, OAuth | If deployed in restricted enterprise environment |

---

## Appendix: Key Formulas Reference

### A. Price Elasticity (Log-Log Model)

```
log(V) = alpha + beta1 * log(P_ratio) + beta2 * controls + epsilon
Elasticity = beta1 = (d_log_V / d_log_P) = (dV/V) / (dP/P)
```

### B. SHAP Values

```
phi_i(f, x) = SUM_{S subset F\{i}} [|S|!(|F|-|S|-1)!/|F|!] * [f(S union {i}) - f(S)]
```

Sum of marginal contributions across all feature coalitions. SHAP values sum to the model output minus the expected output.

### C. Thompson Sampling Update (Beta-Bernoulli)

```
Prior: theta_arm ~ Beta(alpha_0, beta_0)
After success (r=1): alpha <- alpha + 1
After failure (r=0): beta <- beta + 1
Sample: theta_hat = sample(Beta(alpha, beta)); choose arm with highest theta_hat
```

### D. Bellman Equation (Q-Learning)

```
Q(s,a) <- Q(s,a) + lr * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]
where lr = learning rate, gamma = discount factor
```

### E. FT-Transformer Feature Tokenization

```
For numerical feature x_j:    t_j = x_j * W_j + b_j       (linear projection)
For categorical feature x_j:  t_j = Embedding(x_j) + b_j
Input to transformer: T = [t_1, t_2, ..., t_k, t_CLS]
```

### F. TFT Quantile Loss (Pinball Loss)

```
L(q, y, y_hat) = q * max(y - y_hat, 0) + (1-q) * max(y_hat - y, 0)
```

---

*This specification defines how the Fuel Pricing Intelligence Lab is built. See PRD.md for product requirements and business context.*
