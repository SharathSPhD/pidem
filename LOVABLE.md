# Port & Rebuild on Lovable.dev

This document describes how to port or rebuild the **Pricing Intelligence Lab** from this GitHub repo so Lovable (or any builder) can understand the stack and run the full project.

## Repo layout (what Lovable needs)

- **Frontend**: `frontend/` — Next.js 16, React 19, TypeScript, Tailwind, Plotly.js, Zustand.
- **Backend**: `backend/` — FastAPI, Pydantic, Python 3.12+. Serves all `/api/*` and ML endpoints.
- **Data (included)**: `backend/data/raw/` (CSV inputs) and `backend/data/pidem.db` (SQLite). **No data generation step required** — the app loads from the DB on first run.
- **Config**: Root `.env` is gitignored; create it for optional tokens (see below). Backend uses `backend/config.py` (env vars / defaults).

## Stack summary

| Layer    | Tech |
| -------- | ----- |
| Frontend | Next.js 16, React 19, Tailwind CSS, Plotly.js, Zustand |
| Backend  | FastAPI, Pydantic, uvicorn |
| Data     | SQLite (`backend/data/pidem.db`) + CSV in `backend/data/raw/` |
| ML/DS    | scikit-learn, XGBoost, LightGBM, PyTorch, statsmodels, SHAP, PuLP (see `backend/pyproject.toml`) |

## Build & run (full project)

### 1. Backend (required for app to work)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"     # or: pip install -r pyproject.toml deps
uvicorn main:app --host 0.0.0.0 --port 8000
```

- Backend reads from `backend/data/pidem.db` (and `backend/data/raw/` if DB is missing; then it can regenerate and persist).
- API base: `http://localhost:8000` (e.g. `GET /health`, `POST /api/m00/train`, etc.).

### 2. Frontend (consumes backend API)

```bash
cd frontend
npm install
npm run dev
```

- Runs on port 3000. Set `NEXT_PUBLIC_API_URL=http://localhost:8000` if the API is not on same host.
- All module pages call `fetch('/api/...')`; in production, proxy or set API base URL to the backend.

### 3. Environment variables (optional)

Create `.env` in project root if you need:

- `HF_TOKEN`, `WANDB_API_KEY`, `NGC_API_KEY` — for optional LLM/fine-tuning (app runs without them; LLM features show graceful “offline” messages).
- `github_token` — only for scripts that push to GitHub (e.g. `scripts/push_with_token.sh`).

No env vars are required for a basic run; SQLite and included data are enough.

## Data and DB (included for Lovable)

- **`backend/data/pidem.db`** — SQLite database with all app datasets (pre-built). Commit this so the repo is self-contained.
- **`backend/data/raw/`** — CSV files (e.g. Mendeley, Brent, weather, stations). Also committed so the project can be rebuilt without external downloads.

Do **not** gitignore `backend/data/*.db` or `backend/data/raw/` if you want a one-click port to Lovable or other hosts.

## Why "Backend offline" when opening the app from Lovable (HTTPS)

The app is served from **HTTPS** (e.g. `https://lovable.dev/...`). If the frontend calls **`http://localhost:8000`**, the browser can **block** the request as **mixed content** (HTTPS page loading an HTTP resource). So even with the backend running and `http://localhost:8000/` working in another tab, the Lovable-hosted app may still show "Backend offline".

**Fix: expose your local backend over HTTPS and point the app at it**

1. **Expose port 8000 with a tunnel** (pick one):
   - **ngrok:** `ngrok http 8000` → use the `https://xxxx.ngrok-free.app` URL.
   - **Cloudflare Tunnel:** `cloudflared tunnel --url http://localhost:8000` → use the provided `https://` URL.
2. **Set the frontend API base** to that HTTPS URL:
   - **Next.js:** in Lovable project settings or `.env`, set `NEXT_PUBLIC_API_URL=https://xxxx.ngrok-free.app` (or your tunnel URL).
   - **Vite:** set `VITE_API_URL=https://xxxx.ngrok-free.app`.
3. Restart/rebuild the frontend so it uses the new URL. The app will then call your backend via HTTPS and the browser will allow it.

**Alternative:** Run the frontend **locally** (`npm run dev` in `frontend/`) and open `http://localhost:3000`. Then both app and API are on HTTP localhost and mixed content is not an issue.

## Port / Rebuild checklist for Lovable

1. **Connect this repo** to a Lovable project (GitHub integration: connect project to this repository).
2. **Backend**: Lovable is frontend-focused; for full functionality you need the FastAPI backend running elsewhere (e.g. Railway, Render, Fly.io) and point the frontend’s API base to that URL, or run backend locally and use Lovable for frontend edits.
3. **Frontend**: Root of the app is `frontend/`; entry is `frontend/src/app/page.tsx` (hub). All 19 modules live under `frontend/src/app/modules/`.
4. **Data**: No extra data step — use the committed `backend/data/pidem.db` and `backend/data/raw/` so the backend works out of the box.
5. **API contract**: See `backend/routers/modules.py` for all `/api/m00/train` … `/api/m18/...` endpoints; frontend uses `POST` with JSON bodies and expects `{ figures, metrics, data }` where applicable.

## One-sentence summary for Lovable

**Full-stack pricing learning platform: Next.js frontend in `frontend/`, FastAPI backend in `backend/`, SQLite + CSV in `backend/data/` (included). Run backend on port 8000, frontend on 3000; no data generation needed.**
