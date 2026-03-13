from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers.modules import ALL_MODULE_ROUTERS

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://lovable.dev",
        "https://www.lovable.dev",
    ],
    allow_origin_regex=r"https://(.*\.)?(lovable|lovableproject|entri|vercel|railway)\.(app|dev|com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in ALL_MODULE_ROUTERS:
    app.include_router(router)

from routers.data import router as data_router
from routers.eda import router as eda_router
app.include_router(data_router)
app.include_router(eda_router)

from routers.regression import router as reg_router
from routers.classification import router as cls_router
from routers.clustering import router as clust_router
from routers.timeseries import router as ts_router
from routers.optimization import router as opt_router
from routers.bandit import router as bandit_router
from routers.rl import router as rl_router
from routers.neural import router as neural_router
app.include_router(reg_router)
app.include_router(cls_router)
app.include_router(clust_router)
app.include_router(ts_router)
app.include_router(opt_router)
app.include_router(bandit_router)
app.include_router(rl_router)
app.include_router(neural_router)


@app.get("/")
def root():
    """Root path for health checks and discovery. Use /health for full status."""
    return {
        "app": settings.app_name,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    dependencies = {}

    # Database
    try:
        from data.cache import _get_engine
        engine = _get_engine()
        drivername = engine.url.drivername
        if "postgresql" in drivername:
            dependencies["database"] = "postgresql"
        elif "sqlite" in drivername:
            dependencies["database"] = "sqlite"
        else:
            dependencies["database"] = drivername
    except Exception:
        dependencies["database"] = "unavailable"

    # Redis
    try:
        from data.cache import _get_redis
        redis_client = _get_redis()
        dependencies["redis"] = "connected" if redis_client else "unavailable"
    except Exception:
        dependencies["redis"] = "unavailable"

    # NIM
    try:
        nim_url = getattr(settings, "nim_url", None) or ""
        dependencies["nim"] = "configured (not tested)" if nim_url else "not configured"
    except Exception:
        dependencies["nim"] = "not configured"

    return {
        "status": "ok",
        "app": settings.app_name,
        "dependencies": dependencies,
    }
