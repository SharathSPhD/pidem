"""Plotly figure factory helpers for consistent chart styling."""

import json
from typing import Any

import plotly.graph_objects as go

COLORS = {
    "primary": "#1a73e8",
    "secondary": "#ea4335",
    "success": "#34a853",
    "warning": "#fbbc04",
    "motorway": "#1a73e8",
    "urban": "#ea4335",
    "rural": "#34a853",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
)


def fig_to_response(fig: go.Figure) -> dict[str, Any]:
    fig.update_layout(**LAYOUT_DEFAULTS)
    return {"figure": json.loads(fig.to_json())}


def quality_gate_warning(metric_name: str, value: float, threshold: float) -> dict:
    passed = value >= threshold
    return {
        "metric": metric_name,
        "value": round(value, 4),
        "threshold": threshold,
        "passed": passed,
        "message": (
            f"{metric_name} = {value:.4f} (threshold: {threshold}). "
            + ("Quality gate passed." if passed else
               "This model configuration may not generalize well. "
               "Consider adjusting hyperparameters.")
        ),
    }
