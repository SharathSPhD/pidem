"""Bandit simulation API router."""

from typing import Literal

import plotly.graph_objects as go
from fastapi import APIRouter
from pydantic import BaseModel

from models.bandit import simulate_bandit
from utils.chart_helpers import fig_to_response

router = APIRouter(prefix="/api/v1/bandit", tags=["bandit"])


class SimulateRequest(BaseModel):
    algorithm: Literal["epsilon_greedy", "ucb1", "thompson_sampling", "fdsw_thompson"] = "thompson_sampling"
    n_hours: int = 168
    true_elasticity: float = -1.2
    non_stationary: bool = False
    prior_strength: float = 1.0


@router.post("/simulate")
def simulate(request: SimulateRequest):
    """Run bandit simulation."""
    result = simulate_bandit(
        algorithm=request.algorithm,
        n_hours=request.n_hours,
        true_elasticity=request.true_elasticity,
        non_stationary=request.n_stationary,
        prior_strength=request.prior_strength,
    )
    # Build Plotly figures
    fig_regret = go.Figure()
    fig_regret.add_trace(go.Scatter(y=result.cumulative_regret, mode="lines", name="Cumulative Regret"))
    fig_regret.update_layout(
        title="Cumulative Regret",
        xaxis_title="Hour",
        yaxis_title="Regret (EUR)",
    )

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=result.arm_selection_heatmap["z"],
        x=result.arm_selection_heatmap["x"],
        y=result.arm_selection_heatmap["y"],
        colorscale="Blues",
    ))
    fig_heatmap.update_layout(
        title="Arm Selection Heatmap",
        xaxis_title="Time Bucket",
        yaxis_title="Price Level (ct)",
    )

    return {
        "cumulative_regret": result.cumulative_regret,
        "revenue_comparison": result.revenue_comparison,
        "posterior_evolution": result.posterior_evolution,
        "figures": {
            "regret": fig_to_response(fig_regret),
            "heatmap": fig_to_response(fig_heatmap),
        },
    }
