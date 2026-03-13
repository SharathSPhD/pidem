"""Time series router: STL, ARIMA, Prophet, LightGBM, walk-forward validation."""

from fastapi import APIRouter, BackgroundTasks, HTTPException

from models import timeseries as ts_models
from services.run_manager import complete_run, create_run, fail_run, get_run

router = APIRouter(prefix="/api/v1/models/timeseries", tags=["timeseries"])


def _run_arima(run_id: str, station_id: str, p: int, d: int, q: int, forecast_horizon: int):
    try:
        result = ts_models.train_arima(station_id, p=p, d=d, q=q, forecast_horizon=forecast_horizon)
        if "error" in result:
            fail_run(run_id, result["error"])
        else:
            wf = ts_models.walk_forward_validation(station_id, model_type="arima", n_splits=3, horizon=7)
            result["walkforward"] = wf
            complete_run(run_id, metrics={}, results=result)
    except Exception as e:
        fail_run(run_id, str(e))


def _run_prophet(run_id: str, station_id: str, holidays: bool, changepoint_scale: float, horizon: int):
    try:
        result = ts_models.train_prophet(
            station_id, holidays=holidays, changepoint_scale=changepoint_scale, horizon=horizon
        )
        if "error" in result:
            fail_run(run_id, result["error"])
        else:
            wf = ts_models.walk_forward_validation(station_id, model_type="arima", n_splits=3, horizon=7)
            result["walkforward"] = wf
            complete_run(run_id, metrics={}, results=result)
    except Exception as e:
        fail_run(run_id, str(e))


@router.post("/decompose")
def decompose_stl(station_id: str = "STN_001", period: int = 7):
    """STL decomposition (sync). Returns figure + components."""
    result = ts_models.decompose_stl(station_id, period=period)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/arima")
def train_arima(
    background_tasks: BackgroundTasks,
    station_id: str = "STN_001",
    p: int = 1,
    d: int = 0,
    q: int = 1,
    forecast_horizon: int = 14,
):
    """ARIMA training (async). Returns run_id for polling."""
    run = create_run(est_seconds=30)
    background_tasks.add_task(
        _run_arima, run.run_id, station_id, p, d, q, forecast_horizon
    )
    return {"run_id": run.run_id, "status": "training"}


@router.post("/prophet")
def train_prophet(
    background_tasks: BackgroundTasks,
    station_id: str = "STN_001",
    holidays: bool = True,
    changepoint_scale: float = 0.05,
    horizon: int = 14,
):
    """Prophet training (async). Returns run_id for polling."""
    run = create_run(est_seconds=45)
    background_tasks.add_task(
        _run_prophet, run.run_id, station_id, holidays, changepoint_scale, horizon
    )
    return {"run_id": run.run_id, "status": "training"}


@router.post("/lightgbm_lag")
def train_lightgbm(
    station_id: str = "STN_001",
    lag_features: list[int] | None = None,
    horizon: int = 14,
):
    """LightGBM with lag features (sync). Returns forecast + SHAP importance."""
    result = ts_models.train_lightgbm_lag(
        station_id, lag_features=lag_features, horizon=horizon
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/{run_id}/forecast")
def get_forecast(run_id: str):
    """Get forecast results for ARIMA/Prophet/LightGBM run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status.value == "failed":
        raise HTTPException(status_code=400, detail=run.error or "Run failed")
    if run.status.value != "complete":
        return {"run_id": run_id, "status": run.status.value, "progress": run.progress}
    return {"run_id": run_id, "status": "complete", **run.results}


@router.get("/{run_id}/walkforward")
def get_walkforward(run_id: str):
    """Get walk-forward validation results for a completed forecast run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status.value != "complete":
        raise HTTPException(status_code=400, detail=f"Run not complete: {run.status.value}")
    wf = run.results.get("walkforward", {})
    return {"run_id": run_id, **wf}
