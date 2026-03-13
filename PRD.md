# Fuel Pricing Intelligence Lab — Product Requirements Document

> **Version:** 1.0 | **Status:** Final
> **Product Type:** Interactive ML/AI Learning Platform for Fuel Pricing Professionals
> **Stack:** Python, FastAPI, Plotly Dash (with React/Next.js migration path)

---

## 1. Vision & Problem Statement

### 1.1 The Gap

Fuel pricing professionals — analysts, managers, and directors — possess deep domain expertise in competitive dynamics, elasticity intuition, and margin management. However, they lack hands-on experience with the machine learning and AI tools increasingly used to augment pricing decisions.

Existing ML training falls into two traps:

| Trap | Symptom | Result |
|---|---|---|
| Too Academic | Derives OLS from first principles, uses toy data | Pricing professionals disengage — "this has nothing to do with my job" |
| Too Superficial | Sliders move, numbers change, no real model underneath | No genuine understanding — "I still don't know when to trust this" |

### 1.2 The Product

The **Fuel Pricing Intelligence Lab** is an interactive, web-based learning platform that bridges pricing expertise and ML/AI fluency. Users train real models on realistic synthetic fuel pricing data, interrogate their outputs, provoke their failures, and learn when to trust — and when to doubt — AI-driven pricing recommendations.

Every concept answers three questions the audience actually cares about:

1. **What problem in my pricing workflow does this solve?**
2. **What can go wrong — and how would I know?**
3. **When should I NOT use this?**

### 1.3 Scope Contracts

**What this platform IS:**
- A conceptual and practical bridge from pricing expertise to ML/AI fluency
- An interactive environment for building intuition through play and consequence
- A foundation for evaluating, commissioning, and governing AI-assisted pricing tools

**What this platform IS NOT:**
- A production pricing engine
- A substitute for data science training (no backpropagation derivation, no computational complexity theory)
- A code tutorial (Python is shown as reference; business users can ignore it entirely)

---

## 2. User Personas

### 2.1 Pricing Analyst

| Attribute | Detail |
|---|---|
| **Role** | Day-to-day pricing execution; monitors competitor moves, adjusts prices, reports on performance |
| **ML Background** | Minimal — uses Excel, understands correlation vs. causation at a high level |
| **Goal** | Understand what ML models produce, how to read their outputs, and when the numbers are unreliable |
| **Key Modules** | M0 (Foundations), M1 (Regression/Elasticity), M2 (Classification/Threat Detection), M5 (Anomaly Detection), M6 (Time Series) |
| **Success Looks Like** | "I can look at an elasticity estimate with a confidence interval and know whether this supports the price cut my manager is considering — or whether the uncertainty is too wide to act on." |

### 2.2 Pricing Manager

| Attribute | Detail |
|---|---|
| **Role** | Oversees pricing strategy for a region or station cluster; approves pricing recommendations; manages analyst team |
| **ML Background** | Conceptual awareness — has seen dashboards with ML outputs but doesn't know how they're built |
| **Goal** | Evaluate AI-driven pricing recommendations critically; know when to override; commission the right models for the right problems |
| **Key Modules** | M3 (Ensembles/SHAP), M4 (Clustering/Segmentation), M9 (Optimization), M10 (Bandits), M18 (Synthesis) |
| **Success Looks Like** | "When the optimization model recommends a price cut at 12 stations, I can read the SHAP explanation, check the constraint shadow prices, and make an informed approve/reject/modify decision." |

### 2.3 VP / Director

| Attribute | Detail |
|---|---|
| **Role** | Sets pricing strategy direction; accountable for margin targets and competitive positioning; evaluates AI investment |
| **ML Background** | Executive briefings — understands "AI can help" but not the mechanics, risks, or governance requirements |
| **Goal** | Assess organizational AI readiness; understand model risk, governance obligations, and the boundary between automation and human judgment |
| **Key Modules** | M0 (Foundations — bias-variance as a decision concept), M15 (Transformer Zoo — the landscape), M16 (LLM Capabilities/Limits), M18 (Synthesis — system design and governance) |
| **Success Looks Like** | "I can evaluate a vendor's AI pricing proposal and ask the right questions: What's the model drift detection strategy? Where is the human override? What does the audit trail look like?" |

---

## 3. Learning Objectives & Curriculum

### 3.1 Pedagogical Pattern

Every module follows the same five-beat structure:

```
FRAME  ->  EXPLORE  ->  BUILD  ->  INTERROGATE  ->  DECIDE
```

| Beat | Purpose | Example (Regression Module) |
|---|---|---|
| **FRAME** | A pricing story that sets the stakes | "If we cut price 4ct below the nearest competitor, what volume uplift do we expect — and is it enough to compensate the margin hit?" |
| **EXPLORE** | EDA of the relevant data slice — see before you model | Scatter of price gap vs. volume; distribution of competitor gaps by station type |
| **BUILD** | Train the model interactively — controls exposed, mechanics visible | Select features, set regularization, train a log-log elasticity model, see coefficients update |
| **INTERROGATE** | Diagnostics — what does the model not see? Where does it fail? | Residual plots, VIF for multicollinearity, autocorrelation in residuals, Cook's distance for outlier stations |
| **DECIDE** | The "so what?" — how does this output feed a pricing decision, and what human judgment is still required? | "The elasticity CI at this station type is [-1.4, -2.2]. At -1.4, break-even needs +28% volume. At -2.2, only +18%. That uncertainty changes the decision." |

### 3.2 Module Map

```
FOUNDATION
  M0   The Data Mindset & ML Taxonomy

SUPERVISED LEARNING
  M1   Regression & Price Elasticity
  M2   Classification & Competitive Threat Detection
  M3   Ensemble Methods (Random Forest, XGBoost) & Explainability

UNSUPERVISED LEARNING
  M4   Clustering & Station Segmentation
  M5   Anomaly Detection & Market Surveillance

TIME SERIES & FORECASTING
  M6   Classical Methods (Decomposition, ARIMA, Prophet)
  M7   Sequence Models (Lag Features, LSTM)
  M8   Advanced: Temporal Fusion Transformer (TFT)

OPTIMIZATION
  M9   Linear Programming & Margin Maximization
  M10  Multi-Armed Bandits & Pricing Experiments

REINFORCEMENT LEARNING
  M11  Markov Decision Processes & Q-Learning
  M12  Deep Q-Networks for Dynamic Pricing

NEURAL NETWORK DEEP DIVE
  M13  Architecture Explorer: MLPs, Activations, Embeddings

TRANSFORMERS & MODERN DEEP LEARNING
  M14  Self-Attention for Tabular Data (FT-Transformer)
  M15  The Transformer Zoo: BERT, GPT, T5 — A Map

LARGE LANGUAGE MODELS
  M16  LLM Architecture, Capabilities & Limits
  M17  Prompt Engineering & RAG for Pricing Intelligence

SYNTHESIS
  M18  Putting It Together: Pricing AI System Design
```

### 3.3 Per-Module Learning Objectives (Business Outcomes)

| Module | After completing this module, the learner can... |
|---|---|
| **M0** | Distinguish supervised, unsupervised, and reinforcement learning with fuel pricing examples; explain the bias-variance tradeoff as a business risk; construct honest train/test evaluations |
| **M1** | Read and interpret a price elasticity coefficient with confidence intervals; decide whether a price cut is margin-accretive given estimation uncertainty; identify when regression diagnostics signal unreliable estimates |
| **M2** | Evaluate a competitive threat classification model's precision-recall tradeoff in business terms (cost of missed threat vs. cost of false alarm); read a decision tree's pricing rules |
| **M3** | Interpret a SHAP explanation for any model prediction in plain business language; distinguish global feature importance from local prediction explanations; evaluate when ensemble complexity is justified |
| **M4** | Assess whether a station clustering is actionable (adds information beyond existing labels); evaluate cluster stability and assignment quality; map clusters to differentiated pricing strategies |
| **M5** | Configure anomaly detection sensitivity to match operational cost tradeoffs; distinguish genuine market events from data errors; understand spatial-temporal anomaly clustering |
| **M6** | Interpret time series decomposition components; evaluate forecast reliability by horizon length; understand why walk-forward validation produces honest performance estimates |
| **M7** | Compare feature-engineered (LightGBM) vs. learned (LSTM) approaches to sequence modeling; evaluate which lag structures drive demand forecasts |
| **M8** | Interpret TFT attention maps as demand-driver insights; evaluate prediction interval calibration; assess when TFT complexity is justified over simpler models |
| **M9** | Read an optimization solution's shadow prices as monetary values of constraint relaxation; evaluate the margin-volume tradeoff frontier; understand where linear approximations fail |
| **M10** | Explain the explore-exploit tradeoff in pricing experiments; evaluate Thompson Sampling's advantage over static A/B testing; assess when non-stationarity undermines bandit assumptions |
| **M11** | Read a learned Q-learning policy as pricing rules; evaluate how discount factor changes strategy horizon; understand when tabular RL is appropriate |
| **M12** | Assess when neural Q-functions improve over tabular methods; understand experience replay and target networks as stability mechanisms |
| **M13** | Identify activation function failure modes (vanishing gradients, dead neurons); interpret learned station embeddings; evaluate network capacity vs. overfitting |
| **M14** | Interpret feature-feature attention heatmaps as learned interaction structure; evaluate when transformer-based tabular models outperform tree-based methods |
| **M15** | Distinguish encoder (BERT), decoder (GPT), and encoder-decoder (T5) architectures by their appropriate pricing use cases; navigate the model landscape |
| **M16** | Identify LLM capability boundaries (what requires tools, what requires trained ML models); detect hallucination risk; evaluate temperature and sampling tradeoffs |
| **M17** | Construct effective prompts using RCTC pattern; evaluate RAG retrieval quality (chunk size, top-K); assess when RAG vs. fine-tuning is appropriate |
| **M18** | Design a production AI-augmented pricing system architecture; identify model drift risks; articulate governance requirements (audit trails, human override, regulatory compliance) |

---

## 4. Functional Requirements

### FR-1: Interactive Synthetic Data Environment

The platform generates and serves realistic synthetic fuel pricing data that produces non-trivial model behavior. The data engine is configurable for any station network topology and geography.

| Requirement | Detail |
|---|---|
| FR-1.1 | Generate a configurable station network (default: 80 stations) with Motorway, Urban, and Rural types |
| FR-1.2 | Produce hourly price series with reference price tracking, strategic offsets by station type, competitive response functions, and observation noise |
| FR-1.3 | Produce hourly volume series from a hidden log-linear demand model with cross-price elasticity, autoregressive components, intraday/weekly/annual seasonality, holiday effects, weather effects, and station fixed effects |
| FR-1.4 | Model 4 competitor archetypes with distinct behavioral profiles (tracking, aggressive, stable, high-variance) |
| FR-1.5 | Include realistic data imperfections: heteroscedasticity, outliers (0.3%), regime changes (e.g., energy crisis periods), and multicollinearity |
| FR-1.6 | Provide exogenous variable series: crude oil price (geometric Brownian motion), wholesale/COGS, temperature cycles, traffic indices, public holidays, event flags |
| FR-1.7 | Expose data through well-defined API contracts (stations, prices, volumes, market data) at both hourly and daily granularity |
| FR-1.8 | Cache generated datasets in memory to avoid regeneration across module interactions |

### FR-2: Per-Module Interactive Controls

Each module provides real-time interactive controls that train actual models (not pre-computed results).

| Requirement | Detail |
|---|---|
| FR-2.1 | Sliders, dropdowns, checkboxes, and buttons directly trigger model training or parameter updates |
| FR-2.2 | Controls include explicit ranges and defaults derived from the module's model formulation |
| FR-2.3 | Parameter changes produce immediate feedback for fast models (<5s) and progress-tracked async training for slow models |
| FR-2.4 | A "Run" button pattern for expensive operations; auto-update pattern for cheap operations (slider → chart) |

### FR-3: Real-Time Visualization

All model outputs are rendered as interactive Plotly charts with hover, zoom, and selection capabilities.

| Requirement | Detail |
|---|---|
| FR-3.1 | All chart data produced server-side as Plotly JSON, ensuring frontend portability |
| FR-3.2 | Each module has a primary chart (always visible) and secondary charts organized in tabs (Diagnostics, Explainability, Comparison) |
| FR-3.3 | SHAP explanations (beeswarm, waterfall, dependence, interaction) available for all tree-based and neural models |
| FR-3.4 | Geographic station map visualization with configurable base map, color-coded overlays, and hover station profiles |
| FR-3.5 | Animated visualizations for training progress (decision boundaries, Q-value convergence, posterior evolution) |

### FR-4: Diagnostics & Deliberate Failure Modes

Every module exposes at least one deliberately provocable failure mode — learning from breakage is more memorable than learning from success.

| Requirement | Detail |
|---|---|
| FR-4.1 | Each module has at least one "Break It" scenario documented and accessible via controls |
| FR-4.2 | When a model fails quality gates (user-selected hyperparameters produce poor generalization), the UI displays a warning card explaining the diagnostic signal and its business meaning |
| FR-4.3 | Diagnostic panels include residual plots, calibration plots, learning curves, and test-set performance — not just training metrics |

### FR-5: Assessment System

Each module concludes with a 3-question interactive validation.

| Requirement | Detail |
|---|---|
| FR-5.1 | Question 1 (Conceptual): multiple choice with instant feedback and explanation |
| FR-5.2 | Question 2 (Applied): numeric input requiring calculation from the module's model output |
| FR-5.3 | Question 3 (Critical Thinking): free-text response with a model answer revealed after submission |
| FR-5.4 | Assessment results stored per user session and reflected in progress tracking |

### FR-6: Progress Tracking

| Requirement | Detail |
|---|---|
| FR-6.1 | Module completion status persisted in browser local storage |
| FR-6.2 | Hub page shows completion state across all 19 modules |
| FR-6.3 | Per-module progress includes: visited, model trained, assessment attempted, assessment passed |

### FR-7: Async Model Training

Long-running models require background training with user-visible progress.

| Requirement | Detail |
|---|---|
| FR-7.1 | Models with >5s expected training time run asynchronously and return a `run_id` |
| FR-7.2 | Frontend polls training status at regular intervals and displays a progress bar |
| FR-7.3 | Training results are retrievable by `run_id` after completion |
| FR-7.4 | Applies to: ARIMA (auto-grid), Prophet, LSTM, TFT, DQN, Q-Learning |

### FR-8: LLM Integration

Modules M16 and M17 require live LLM interaction for prompt engineering and RAG demonstrations.

| Requirement | Detail |
|---|---|
| FR-8.1 | Tokenization visualizer: user types a prompt, sees it tokenized live with token count |
| FR-8.2 | Temperature playground: same prompt at different temperatures, showing response distribution |
| FR-8.3 | Prompt engineering playground: editable prompt with RCTC pattern scaffolding |
| FR-8.4 | RAG pipeline demonstration: document chunking, embedding, similarity search visualization, and LLM synthesis |
| FR-8.5 | LLM provider abstracted behind a client interface for provider independence |

### FR-9: Station Map Visualization

| Requirement | Detail |
|---|---|
| FR-9.1 | Interactive geographic map showing station locations with type/cluster color coding |
| FR-9.2 | Hover tooltips with station profile (type, volume, competitor count, elasticity) |
| FR-9.3 | Configurable base map (Mapbox with free-tier token; OpenStreetMap fallback) |
| FR-9.4 | Map reusable across modules (clustering, anomaly detection, optimization, synthesis) |

---

## 5. Non-Functional Requirements

### NFR-1: Performance

| Metric | Target |
|---|---|
| Simple model response (linear regression, K-Means) | < 2 seconds |
| Medium model response (XGBoost, ARIMA auto-grid) | < 30 seconds |
| Complex model training (LSTM, TFT, DQN) | Async with progress; 30s to 20 min depending on model |
| Chart render after data receipt | < 500 ms |
| Synthetic data generation (full dataset) | < 10 seconds |

### NFR-2: Scalability

| Requirement | Detail |
|---|---|
| NFR-2.1 | Stateless API design: all model state stored server-side by `run_id`; clients hold only run IDs |
| NFR-2.2 | Horizontal scaling: additional backend instances behind a load balancer |
| NFR-2.3 | Dataset caching: generated synthetic data cached in memory across requests |

### NFR-3: Accessibility & Deployment

| Requirement | Detail |
|---|---|
| NFR-3.1 | Web-based: no client-side installation required |
| NFR-3.2 | Docker Compose packaging for self-hosted deployment |
| NFR-3.3 | Cloud-ready via containerization (compatible with any container orchestration platform) |
| NFR-3.4 | Responsive layout: functional on desktop and tablet (primary target: desktop) |

### NFR-4: Portability

| Requirement | Detail |
|---|---|
| NFR-4.1 | Dash-first implementation with thin callbacks (no business logic in frontend) |
| NFR-4.2 | All chart data as Plotly JSON, consumable by both Dash and React (`react-plotly.js`) |
| NFR-4.3 | All API contracts defined as Pydantic schemas, serving as the contract for any frontend |
| NFR-4.4 | Clean migration path to React/Next.js without backend changes |

### NFR-5: Data Isolation

| Requirement | Detail |
|---|---|
| NFR-5.1 | All data is synthetically generated — no PII, no proprietary data, no licensed datasets |
| NFR-5.2 | No external API calls required for data generation |
| NFR-5.3 | LLM API calls (M16-M17) are the only external dependency and are optional |

---

## 6. Success Metrics

| Metric | Measurement Method | Target |
|---|---|---|
| **Module Completion Rate** | % of users who complete all 5 beats of a module | > 70% for M0-M3; > 50% for M11-M18 |
| **Assessment Pass Rate** | % of users scoring 2/3 or better on module assessments | > 60% on first attempt |
| **Time-to-Insight** | Median time from opening a module to producing a trained model with diagnostic output | < 10 minutes for M0-M5; < 20 minutes for M6-M18 |
| **Confidence Uplift** | Pre/post self-assessment: "I can evaluate an ML model's output for a pricing decision" (1-5 scale) | +1.5 point improvement |
| **Return Usage** | % of users who return to the platform within 30 days of first use | > 40% |
| **"Break It" Engagement** | % of users who trigger at least one deliberate failure mode | > 80% |

---

## 7. Constraints & Assumptions

### 7.1 User Assumptions

- Users are pricing-literate professionals — they understand margins, elasticity as a concept, competitive dynamics, and station-level economics
- Users are ML-new — they have not trained models, read SHAP plots, or evaluated model diagnostics before
- Python code shown on-screen is reference material; the primary interaction is through controls and charts, not code editing
- Users access the platform via desktop browser (Chrome, Edge, Firefox) on corporate networks

### 7.2 Technical Constraints

- All training data is synthetic and self-contained — no external data feeds required
- The platform is educational — model training runs are ephemeral (no long-term model persistence required)
- Concurrent user load is modest (< 50 simultaneous users for initial deployment)
- GPU is not required for any module (all models train on CPU within acceptable time bounds; TFT and DQN are the slowest at ~5-20 minutes)

### 7.3 Compliance & Governance

- No PII is processed or stored
- LLM interactions (M16-M17) use API calls to external providers; prompts contain only synthetic data
- The platform itself is not a high-risk AI system under regulatory frameworks — it is an educational tool
- All model outputs include appropriate caveats that this is a learning environment, not production guidance

---

## 8. Design Decisions

### 8.1 Resolved Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **LLM Backend** | Anthropic Claude API, abstracted behind `llm_client.py` interface | Best-in-class for structured reasoning tasks; interface abstraction allows provider swapping without code changes |
| **Async Task Queue** | FastAPI BackgroundTasks for lightweight models; Celery + Redis for heavy models (LSTM, TFT, DQN) | BackgroundTasks avoids infrastructure overhead for simple cases; Celery provides reliable distributed task execution for long-running training |
| **Hosting** | Docker Compose (FastAPI + Redis) for self-hosted deployment; cloud-ready via containerized architecture | Docker Compose is the simplest self-hosted option; same containers deploy to any cloud container service |

### 8.2 Recommendations (Flexible)

| Decision | Recommendation | Alternative | Trade-off |
|---|---|---|---|
| **Vector Store (RAG)** | FAISS (in-memory) | Chroma (persistent), Weaviate (managed) | FAISS is fastest for single-server; Chroma adds persistence if RAG corpus evolves; Weaviate for enterprise multi-tenant |
| **Station Map Tiles** | Mapbox (free tier) | OpenStreetMap / Plotly `scatter_geo` | Mapbox is highest quality; `scatter_geo` requires no API token but less visual polish |
| **TFT Implementation** | PyTorch Forecasting | Darts, custom implementation | PyTorch Forecasting is most complete with built-in TFT; Darts is lighter but less configurable |
| **Authentication** | None (open access) for learning platform | Session-based, OAuth | Authentication adds friction for an educational tool; add if deploying in restricted enterprise environments |

---

*This PRD defines what the Fuel Pricing Intelligence Lab is and why it exists. The companion Technical Specification (spec.md) defines how it is built.*
