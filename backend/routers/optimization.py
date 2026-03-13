"""Optimization API router."""

from typing import Literal

import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.optimization import solve_pricing_lp, OptimizationResult
from services.run_manager import create_run, complete_run, get_run
from utils.chart_helpers import fig_to_response

router = APIRouter(prefix="/api/v1/optimization", tags=["optimization"])


class SolveRequest(BaseModel):
    price_band: float = 0.04
    volume_floor: float = 0.9
    agg_target: float | None = None
    solver_mode: Literal["lp", "nlp"] = "lp"


@router.post("/solve")
def solve(request: SolveRequest):
    """Solve LP/NLP pricing optimization."""
    run = create_run(est_seconds=60)
    result = solve_pricing_lp(
        price_band=request.price_band,
        volume_floor=request.volume_floor,
        agg_target=request.agg_target,
        solver_mode=request.solver_mode,
    )
    complete_run(
        run.run_id,
        metrics={
            "total_margin": result.total_margin,
            "total_volume": result.total_volume,
            "status": result.status,
        },
        results={
            "price_recommendations": result.price_recommendations,
            "pareto_frontier": result.pareto_frontier,
            "shadow_prices": result.shadow_prices,
            "sensitivity_tornado": result.sensitivity_tornado,
        },
    )
    return {"run_id": run.run_id, "status": result.status}


@router.get("/{run_id}/shadow_prices")
def get_shadow_prices(run_id: str):
    """Get shadow prices from optimization run."""
    run = get_run(run_id)
    if not run or not run.results:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"shadow_prices": run.results.get("shadow_prices", {})}


@router.get("/{run_id}/frontier")
def get_frontier(run_id: str):
    """Get Pareto frontier as Plotly figure."""
    run = get_run(run_id)
    if not run or not run.results:
        raise HTTPException(status_code=404, detail="Run not found")
    frontier = run.results.get("pareto_frontier", [])
    if not frontier:
        return {"figure": None}
    margins = [p["margin"] for p in frontier]
    volumes = [p["volume"] for p in frontier]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=volumes, y=margins, mode="lines+markers", name="Pareto"))
    fig.update_layout(
        title="Pareto Frontier: Margin vs Volume",
        xaxis_title="Volume (L)",
        yaxis_title="Gross Margin (EUR)",
    )
    return fig_to_response(fig)


@router.get("/{run_id}/sensitivity")
def get_sensitivity(run_id: str):
    """Get sensitivity tornado as Plotly figure."""
    run = get_run(run_id)
    if not run or not run.results:
        raise HTTPException(status_code=404, detail="Run not found")
    sens = run.results.get("sensitivity_tornado", {})
    if not sens:
        return {"figure": None}
    params = list(sens.keys())
    lows = [sens[p][0] for p in params]
    highs = [sens[p][1] for p in params]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Low", x=params, y=lows, marker_color="#ea4335"))
    fig.add_trace(go.Bar(name="High", x=params, y=highs, marker_color="#34a853"))
    fig.update_layout(
        title="Sensitivity Tornado",
        barmode="group",
        xaxis_title="Parameter",
        yaxis_title="Total Margin (EUR)",
    )
    return fig_to_response(fig)
