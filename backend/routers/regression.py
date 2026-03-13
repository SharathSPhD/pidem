"""Regression API: elasticity model training and diagnostics."""

from fastapi import APIRouter, HTTPException

from pydantic import BaseModel, Field

from data.generator import build_all_datasets
from models.regression import train_elasticity_model
from services.run_manager import create_run, complete_run, get_run, fail_run
from utils.chart_helpers import fig_to_response

router = APIRouter(prefix="/api/v1/models/regression", tags=["regression"])


class TrainRequest(BaseModel):
    station_type: str | None = Field(None, description="Filter by station type: motorway, urban, rural")
    regularization: float = Field(1.0, ge=0.01, description="Ridge alpha")
    date_range: list[str] | None = Field(None, description="[start_date, end_date] as YYYY-MM-DD")


@router.post("/train")
def train_elasticity(req: TrainRequest | None = None):
    """Train elasticity model; returns metrics and diagnostic figures as Plotly JSON."""
    req = req or TrainRequest()
    run = create_run(est_seconds=15)

    try:
        ds = build_all_datasets()
        df = ds["df_daily"]
        if df.empty:
            raise ValueError("df_daily is empty")

        date_range = tuple(req.date_range) if req.date_range and len(req.date_range) == 2 else None
        result = train_elasticity_model(
            features=df,
            station_type=req.station_type,
            regularization=req.regularization,
            date_range=date_range,
        )

        # Serialize figures to Plotly JSON for response and storage
        figures_json = {}
        for name, fig in result["figures"].items():
            figures_json[name] = fig_to_response(fig)

        metrics = result["metrics"]
        results = {
            "coefficients": result["coefficients"],
            "elasticity_price_gap": result["elasticity_price_gap"],
            "elasticity_own_price": result["elasticity_own_price"],
            "vif": result["vif"],
            "figures": figures_json,
        }

        complete_run(run.run_id, metrics=metrics, results=results)

        return {
            "run_id": run.run_id,
            "status": "complete",
            "metrics": metrics,
            "coefficients": result["coefficients"],
            "elasticity_price_gap": result["elasticity_price_gap"],
            "elasticity_own_price": result["elasticity_own_price"],
            "figures": figures_json,
        }
    except ValueError as e:
        fail_run(run.run_id, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        fail_run(run.run_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/diagnostics")
def get_diagnostics(run_id: str):
    """Return residual diagnostics (residual plot, QQ, VIF, Cook's) for a completed run."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run.status.value != "complete":
        raise HTTPException(status_code=400, detail=f"Run not complete: {run.status.value}")

    figures = run.results.get("figures", {})
    if not figures:
        raise HTTPException(status_code=404, detail="No diagnostic figures stored for this run")

    return {"run_id": run_id, "figures": figures}
