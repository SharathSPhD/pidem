"""SHAP explainability helpers for model interpretability."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import BaseEstimator

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False

from utils.chart_helpers import LAYOUT_DEFAULTS


def compute_shap_values(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
) -> tuple[Any, Any]:
    """
    Compute SHAP values using TreeExplainer for tree models or LinearExplainer for linear models.

    Returns:
        Tuple of (shap_values as Explanation object, explainer)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is not installed")
    if X is None or (hasattr(X, "__len__") and len(X) == 0):
        raise ValueError("X cannot be empty")

    if hasattr(X, "columns"):
        names = list(X.columns)
    elif feature_names is not None:
        names = feature_names
    else:
        names = [f"x{i}" for i in range(X.shape[1])]

    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X

    # Tree-based models: XGBoost, DecisionTree, RandomForest, etc.
    tree_models = (
        "XGBClassifier",
        "XGBRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
    )
    model_type = type(model).__name__

    if model_type in tree_models:
        try:
            explainer = shap.TreeExplainer(model, X_arr, feature_names=names)
            shap_values = explainer(X_arr)
        except Exception:
            # Fallback: use model.predict to get background
            explainer = shap.TreeExplainer(model, feature_names=names)
            shap_values = explainer(X_arr)
    else:
        # Linear models: LogisticRegression, Ridge, etc.
        try:
            masker = shap.maskers.Independent(X_arr, feature_names=names)
            explainer = shap.LinearExplainer(model, masker)
            shap_values = explainer(X_arr)
        except Exception:
            # Fallback: use KernelExplainer with a sample
            sample_size = min(100, len(X_arr))
            background = X_arr[np.random.RandomState(42).choice(len(X_arr), sample_size, replace=False)]
            explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict, background)
            shap_values = explainer(X_arr)

    return shap_values, explainer


def shap_waterfall_figure(
    shap_values: Any,
    feature_names: list[str],
    idx: int,
    base_value: float | None = None,
    title: str = "SHAP Waterfall",
) -> go.Figure:
    """
    Create a Plotly waterfall chart for a single observation's SHAP values.

    Args:
        shap_values: SHAP Explanation or array of shape (n_samples, n_features)
        feature_names: List of feature names
        idx: Index of observation to explain
        base_value: E[f(x)] - if None, inferred from shap_values
        title: Chart title

    Returns:
        Plotly Figure
    """
    if SHAP_AVAILABLE and isinstance(shap_values, shap.Explanation):
        vals = shap_values.values[idx]
        if base_value is None:
            base_value = float(shap_values.base_values[idx]) if hasattr(shap_values.base_values, "__getitem__") else float(shap_values.base_values)
    else:
        vals = np.asarray(shap_values)[idx]
        base_value = base_value or 0.0

    vals = np.asarray(vals).flatten()
    if len(vals) != len(feature_names):
        feature_names = feature_names[: len(vals)] or [f"x{i}" for i in range(len(vals))]

    # Sort by absolute SHAP value for readability
    order = np.argsort(np.abs(vals))[::-1]
    vals_ord = vals[order]
    names_ord = [feature_names[i] for i in order]

    # Build waterfall: base (absolute) + contributions (relative) + total
    pred = float(base_value) + float(np.sum(vals_ord))
    measures = ["absolute"] + ["relative"] * len(vals_ord) + ["total"]
    y_vals = [float(base_value)] + [float(v) for v in vals_ord] + [0]  # total y=0 auto-calculates
    text_vals = [f"{base_value:.3f}"] + [f"{v:+.3f}" for v in vals_ord] + [f"{pred:.3f}"]
    labels = ["Base"] + [str(n) for n in names_ord] + ["Prediction"]

    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measures,
            x=labels,
            y=y_vals,
            text=text_vals,
            textposition="outside",
            connector={"line": {"color": "rgb(180,180,180)"}},
            increasing={"marker": {"color": "#34a853"}},
            decreasing={"marker": {"color": "#ea4335"}},
            totals={"marker": {"color": "#1a73e8"}},
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="SHAP value → prediction",
        xaxis_tickangle=-45,
        showlegend=False,
        **LAYOUT_DEFAULTS,
    )
    return fig


def shap_beeswarm_figure(
    shap_values: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    max_display: int = 15,
    title: str = "SHAP Summary (Beeswarm)",
) -> go.Figure:
    """
    Create a Plotly beeswarm-style summary of SHAP values across all observations.

    Returns:
        Plotly Figure
    """
    if SHAP_AVAILABLE and isinstance(shap_values, shap.Explanation):
        vals = np.asarray(shap_values.values)
    else:
        vals = np.asarray(shap_values)

    if vals.ndim == 3:
        vals = vals[:, :, 1]  # For binary classification, use positive class

    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)
    elif feature_names is None:
        feature_names = [f"x{i}" for i in range(vals.shape[1])]

    # Mean absolute SHAP per feature for ordering
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    vals_ord = vals[:, order]
    names_ord = [feature_names[i] for i in order]
    X_ord = X_arr[:, order] if X_arr.ndim > 1 else X_arr

    fig = go.Figure()
    for i, name in enumerate(names_ord):
        shaps = vals_ord[:, i]
        feat_vals = X_ord[:, i] if X_ord.ndim > 1 else np.zeros(len(shaps))
        # Use numeric y for scatter, map to names via tickvals
        fig.add_trace(
            go.Scatter(
                x=shaps,
                y=np.full(len(shaps), i) + np.random.RandomState(42).uniform(-0.15, 0.15, len(shaps)),
                mode="markers",
                marker=dict(
                    size=5,
                    color=feat_vals,
                    colorscale="RdBu",
                    showscale=(i == 0),
                    colorbar=dict(title="Feature value"),
                    line=dict(width=0.5, color="gray"),
                ),
                name=name,
                text=[f"{name}={v:.2f}, SHAP={s:.3f}" for v, s in zip(feat_vals, shaps)],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="SHAP value (impact on model output)",
        yaxis_title="Feature",
        yaxis=dict(
            tickvals=list(range(len(names_ord))),
            ticktext=names_ord,
            autorange="reversed",
        ),
        **LAYOUT_DEFAULTS,
    )
    return fig
