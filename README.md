# Pricing Intelligence Lab

An interactive learning platform that teaches pricing analytics through 19 progressive modules — from basic regression to reinforcement learning and LLMs. Built with a real + synthetic hybrid data engine using German public retail location data.

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Next.js Frontend (port 3000)          │
│  19 Module Pages │ Plotly.js Charts │ Zustand   │
└────────────────────────┬────────────────────────┘
                         │ fetch /api/*
┌────────────────────────▼────────────────────────┐
│           FastAPI Backend (port 8000)            │
│  Module Routers │ ML Models │ Data Engine        │
├──────────┬──────────┬───────────┬───────────────┤
│  Redis   │  Celery  │   FAISS   │  NIM (8001)   │
│  Queue   │  Workers │  Vectors  │  Nemotron-9B  │
└──────────┴──────────┴───────────┴───────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 20+
- Docker (optional, for NIM + Redis)

### 1. Clone and setup

```bash
git clone git@github.com:SharathSPhD/pidem.git
cd pidem

# Backend
cd backend
uv venv && source .venv/bin/activate
uv pip install -r pyproject.toml
cd ..

# Frontend
cd frontend
npm install
cd ..
```

### 2. Environment variables

Create `.env` in the project root:

```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
NGC_API_KEY=your_ngc_key
GITHUB_TOKEN=your_github_token
```

### 3. Data (included for portability)

The repo includes `backend/data/pidem.db` and `backend/data/raw/` so you can run without generating data. To regenerate or refresh:

```bash
./scripts/download_data.sh
```

### 4. Start the backend

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

### 5. Start the frontend

```bash
cd frontend
npm run dev
```

Open http://localhost:3000 to access the Module Hub.

### Port to Lovable.dev

Data and DB are committed so the repo is self-contained. See **[LOVABLE.md](LOVABLE.md)** for stack summary, build/run steps, and a port checklist so Lovable (or any builder) can rebuild the full project from this repo.

### 6. (Optional) Start NIM for LLM features

```bash
docker compose up nemotron-nim redis
```

### 7. (Optional) Fine-tune Nemotron

```bash
./scripts/fine_tune.sh 3 2e-4 4
```

## Module Curriculum

| #   | Module                     | Category       | ML Technique                         |
| --- | -------------------------- | -------------- | ------------------------------------ |
| M00 | Foundations                | Foundation     | Polynomial fit, bias-variance        |
| M01 | Price Elasticity           | Supervised     | Ridge regression, confidence intervals|
| M02 | Threat Detection           | Supervised     | Logistic/Tree/XGBoost classification |
| M03 | Ensembles + Interpretability| Supervised    | XGBoost, SHAP values                 |
| M04 | Station Segmentation       | Unsupervised   | K-Means, PCA, silhouette analysis    |
| M05 | Anomaly Detection          | Unsupervised   | Isolation Forest, control charts     |
| M06 | Classical Time Series      | Time Series    | STL, ARIMA, Prophet                  |
| M07 | Sequence Models            | Time Series    | LightGBM-lag, LSTM                   |
| M08 | Temporal Fusion Transformer| Time Series    | TFT, multi-horizon forecasting       |
| M09 | Price Optimization         | Optimization   | LP/NLP, Pareto frontier              |
| M10 | Multi-Armed Bandits        | RL             | ε-greedy, UCB1, Thompson Sampling    |
| M11 | Q-Learning                 | RL             | Tabular MDP, policy heatmaps         |
| M12 | Deep Q-Networks            | RL             | DQN, experience replay               |
| M13 | MLP Explorer               | Neural         | PyTorch MLP, gradient flow           |
| M14 | FT-Transformer             | Neural         | Self-attention on tabular data       |
| M15 | Transformer Zoo            | LLM            | BERT/GPT/T5 architecture comparison  |
| M16 | LLM Capabilities           | LLM            | Tokenization, generation             |
| M17 | RAG + Prompt Engineering   | LLM            | FAISS retrieval, NIM synthesis       |
| M18 | System Design              | Synthesis      | Architecture, governance             |

## Data Strategy

**Real public data:**
- Station metadata + competitor prices (Tankerkoenig / Mendeley 2022)
- Weather (Open-Meteo Historical API)
- Crude oil (Brent via yfinance)
- German holidays (python-holidays)

**Synthetic overlay:**
- Own-brand volumes (log-linear demand model with hidden elasticities)
- Margins / COGS (crude × conversion + noise)
- Competitive response behavior

## LLM Strategy

- **Base inference:** Nemotron-Nano-9B via NVIDIA NIM container (OpenAI-compatible API)
- **RAG:** FAISS + sentence-transformers on curated pricing corpus
- **Fine-tuning:** LoRA (rank=16) on curriculum-specific Q&A, tracked with W&B

## Tech Stack

| Layer    | Technology                                          |
| -------- | --------------------------------------------------- |
| Frontend | Next.js 16, React 19, Tailwind CSS, Plotly.js, Zustand |
| Backend  | FastAPI, Pydantic, scikit-learn, XGBoost, LightGBM  |
| ML/DL    | PyTorch, statsmodels, Prophet, SHAP, PuLP           |
| LLM      | NVIDIA NIM, sentence-transformers, FAISS, peft      |
| Infra    | Docker Compose, Redis, Celery, uv                   |
| Tracking | Weights & Biases, HuggingFace Hub                   |

## Project Structure

```
pidem/
├── backend/
│   ├── main.py              # FastAPI app entry
│   ├── config.py             # Settings (env vars)
│   ├── data/                 # Data engine
│   │   ├── ingest.py         # Real data loaders
│   │   ├── generator.py      # Hybrid data builder
│   │   ├── schemas.py        # Pydantic models
│   │   └── cache.py          # In-memory DataFrame cache
│   ├── models/               # ML model implementations
│   ├── routers/              # API endpoints
│   │   └── modules.py        # Unified /api/mXX/* routes
│   ├── services/             # LLM, RAG, run manager
│   ├── training_data/        # Fine-tuning dataset
│   └── utils/                # Metrics, SHAP, chart helpers
├── frontend/
│   └── src/
│       ├── app/              # Next.js pages
│       │   └── modules/      # 19 module pages
│       ├── components/       # Shared UI components
│       ├── hooks/            # React hooks
│       └── lib/              # API client, store, utils
├── scripts/                  # Shell scripts
├── nim/                      # NIM container docs
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── PRD.md
└── spec.md
```

## License

For educational and research purposes.
