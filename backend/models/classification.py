"""Threat classifier: binary classification for volume loss >15% vs rolling baseline."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from utils.metrics import classification_metrics
from utils.chart_helpers import LAYOUT_DEFAULTS
from utils.shap_helpers import compute_shap_values, shap_waterfall_figure, shap_beeswarm_figure

MODEL_TYPES = ("logistic", "tree", "xgboost")


def _build_target(df: pd.DataFrame) -> pd.Series:
    """Compute volume_loss_gt_15pct: 1 if volume dropped >15% vs 4-day rolling avg."""
    df = df.sort_values(["station_id", "date"])
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station_id", "date_dt"])

    def rolling_baseline(g):
        vol = g["volume_litres"].astype(float)
        roll = vol.rolling(4, min_periods=1).mean().shift(1)
        pct_change = (vol - roll) / roll.replace(0, np.nan)
        return (pct_change < -0.15).astype(int)

    target = df.groupby("station_id", group_keys=False).apply(rolling_baseline)
    return target.reindex(df.index).fillna(0).astype(int)


def _prepare_data(
    df: pd.DataFrame,
    features: list[str] | None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare X, y for classification."""
    req = ["volume_litres", "date", "station_id"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df = df.dropna(subset=req).copy()
    if df.empty:
        raise ValueError("No data after dropping missing values")

    y = _build_target(df)

    default_features = [
        "our_price",
        "min_comp_price",
        "gross_margin",
        "temperature",
        "highway_index",
        "is_holiday",
        "n_competitors",
    ]
    use_features = features or [c for c in default_features if c in df.columns]
    missing = [c for c in use_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[use_features].copy()
    X["is_holiday"] = X["is_holiday"].astype(int)
    X = X.fillna(X.median())

    # Drop rows where target is NaN (e.g. first rows per station)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    if len(np.unique(y)) < 2:
        raise ValueError("Target has only one class; need both 0 and 1")

    return X, y, use_features


def train_threat_classifier(
    features: pd.DataFrame | None = None,
    model_type: str = "logistic",
    threshold: float = 0.5,
    tree_depth: int = 5,
    feature_list: list[str] | None = None,
) -> dict[str, Any]:
    """
    Train binary classifier for volume_loss_gt_15pct.

    Args:
        features: DataFrame with required columns
        model_type: "logistic", "tree", or "xgboost"
        threshold: Classification threshold
        tree_depth: Max depth for tree/xgboost
        feature_list: Override default features

    Returns:
        Dict with model, metrics, confusion matrix with costs, ROC, PR, tree structure, SHAP
    """
    if features is None:
        raise ValueError("features DataFrame is required")

    model_type = model_type.lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")

    X, y, feature_names = _prepare_data(features, feature_list)
    X_arr = np.asarray(X)

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "tree":
        model = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
    else:
        model = XGBClassifier(max_depth=tree_depth, n_estimators=100, random_state=42, eval_metric="logloss")

    model.fit(X_arr, y)
    y_pred = (model.predict_proba(X_arr)[:, 1] >= threshold).astype(int)
    y_prob = model.predict_proba(X_arr)[:, 1]

    metrics = classification_metrics(y, y_pred, y_prob)

    # Confusion matrix with business costs (example: FN cost > FP cost for volume loss)
    cm = np.array(metrics["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    cost_fn = 500  # Missing a threat
    cost_fp = 100  # False alarm
    business_cost = fn * cost_fn + fp * cost_fp
    metrics["confusion_matrix_costs"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "total_cost": int(business_cost),
    }

    # ROC curve
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", fill="tozeroy"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", **LAYOUT_DEFAULTS)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y, y_prob)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR", fill="tozeroy"))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", **LAYOUT_DEFAULTS)

    # Decision tree structure (only for tree)
    tree_text = None
    if model_type == "tree":
        tree_text = export_text(model, feature_names=feature_names)

    # SHAP
    try:
        shap_values, _ = compute_shap_values(model, X, feature_names)
        fig_beeswarm = shap_beeswarm_figure(shap_values, X, feature_names)
    except Exception:
        shap_values = None
        fig_beeswarm = None

    return {
        "model": model,
        "metrics": metrics,
        "figures": {
            "roc": fig_roc,
            "precision_recall": fig_pr,
            "shap_beeswarm": fig_beeswarm,
        },
        "tree_structure": tree_text,
        "shap_values": shap_values,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "y_prob": y_prob,
    }


def get_shap_waterfall_for_observation(
    result: dict[str, Any],
    observation_idx: int,
) -> go.Figure | None:
    """Get SHAP waterfall for a single observation from train result."""
    shap_vals = result.get("shap_values")
    X = result.get("X")
    feature_names = result.get("feature_names", [])
    if shap_vals is None or X is None or observation_idx < 0 or observation_idx >= len(X):
        return None
    return shap_waterfall_figure(
        shap_vals,
        feature_names,
        observation_idx,
        title=f"SHAP Waterfall - Observation {observation_idx}",
    )
