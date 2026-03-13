"""Elasticity regression model: log-log demand with Ridge regularization."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.metrics import regression_metrics
from utils.chart_helpers import LAYOUT_DEFAULTS

REQUIRED_COLS = [
    "our_price",
    "min_comp_price",
    "volume_litres",
    "temperature",
    "highway_index",
    "is_holiday",
    "station_type",
]


def _prepare_features(
    df: pd.DataFrame,
    station_type: str | None,
    date_range: tuple[str, str] | None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare X, y and feature names for the log-log model."""
    df = df.dropna(subset=REQUIRED_COLS)
    if df.empty:
        raise ValueError("No rows after dropping missing required columns")

    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if date_range:
        df["date_dt"] = pd.to_datetime(df["date"])
        start, end = date_range
        df = df[(df["date_dt"] >= start) & (df["date_dt"] <= end)]
        df = df.drop(columns=["date_dt"], errors="ignore")

    if station_type:
        df = df[df["station_type"] == station_type]

    if df.empty:
        raise ValueError("No data after filtering by station_type/date_range")

    # Log-log: log(Volume) = alpha + beta1*(P_our - P_min_comp) + beta2*P_our + controls
    price_gap = df["our_price"] - df["min_comp_price"]
    log_volume = np.log(df["volume_litres"].clip(lower=1))

    X_parts = {
        "price_gap": price_gap,
        "our_price": df["our_price"],
        "temperature": df["temperature"],
        "highway_index": df["highway_index"],
        "is_holiday": df["is_holiday"].astype(int),
    }

    # Station type dummies if not filtered
    if not station_type and "station_type" in df.columns:
        dummies = pd.get_dummies(df["station_type"], prefix="st", drop_first=True)
        for c in dummies.columns:
            X_parts[c] = dummies[c]

    X = pd.DataFrame(X_parts)
    feature_names = list(X.columns)
    y = log_volume.values
    return X, pd.Series(y), feature_names


def train_elasticity_model(
    features: pd.DataFrame | None = None,
    station_type: str | None = None,
    regularization: float = 1.0,
    date_range: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """
    Train log-log demand elasticity model with Ridge regression.

    Model: log(Volume) = alpha + beta1*(P_our - P_min_comp) + beta2*P_our + controls

    Returns:
        Dict with coefficients, metrics, plotly figures (residual, qq, vif, cooks)
    """
    if features is None:
        raise ValueError("features DataFrame is required")

    X, y, feature_names = _prepare_features(features, station_type, date_range)
    X_arr = np.asarray(X, dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    X_with_const = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    model = Ridge(alpha=regularization, random_state=42)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    # Intercept and coefficients (unscale for interpretability)
    coef_raw = np.concatenate([[model.intercept_], model.coef_])
    # Approximate standard errors via OLS-style formula
    n, p = X_with_const.shape
    resid = y - y_pred
    dof = max(n - p, 1)
    mse = np.sum(resid**2) / dof
    try:
        var_b = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se = np.sqrt(np.maximum(np.diag(var_b), 0))
    except np.linalg.LinAlgError:
        se = np.full(p + 1, np.nan)

    t_crit = stats.t.ppf(0.975, max(n - p - 1, 1))
    ci_low = coef_raw - t_crit * se
    ci_high = coef_raw + t_crit * se

    coef_names = ["intercept"] + feature_names
    coefficients = [
        {
            "name": name,
            "value": round(float(v), 6),
            "se": round(float(s), 6),
            "ci_95_low": round(float(l), 6),
            "ci_95_high": round(float(h), 6),
        }
        for name, v, s, l, h in zip(coef_names, coef_raw, se, ci_low, ci_high)
    ]

    # Elasticity: beta1 on price_gap, beta2 on our_price
    elasticity_gap = next((c["value"] for c in coefficients if c["name"] == "price_gap"), None)
    elasticity_own = next((c["value"] for c in coefficients if c["name"] == "our_price"), None)

    metrics = regression_metrics(y, y_pred)

    # VIF -- add intercept column (statsmodels requires it for correct VIF)
    X_vif = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
    vif_data = []
    for i, name in enumerate(feature_names):
        try:
            vif = variance_inflation_factor(X_vif, i + 1)
            if not np.isfinite(vif):
                vif_data.append({"feature": name, "vif": None})
            else:
                vif_data.append({"feature": name, "vif": round(float(vif), 2)})
        except Exception:
            vif_data.append({"feature": name, "vif": None})

    # Cook's distance (manual computation)
    from numpy.linalg import lstsq
    beta_ols, _, _, _ = lstsq(X_with_const, y, rcond=None)
    fitted = X_with_const @ beta_ols
    resid_ols = y - fitted
    mse_ols = np.sum(resid_ols**2) / dof
    hat = np.diag(X_with_const @ np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T)
    hat = np.clip(hat, 0, 1 - 1e-10)
    denom = p * max(mse_ols, 1e-12)
    cooks_d = (resid_ols**2 / denom) * (hat / (1 - hat) ** 2)
    cooks_d = np.where(np.isfinite(cooks_d), cooks_d, 0)

    # Figures
    fig_residual = _residual_plot(y, y_pred, resid)
    fig_qq = _qq_plot(resid)
    fig_vif = _vif_bar_figure(vif_data)
    fig_cooks = _cooks_figure(cooks_d)

    return {
        "coefficients": coefficients,
        "elasticity_price_gap": elasticity_gap,
        "elasticity_own_price": elasticity_own,
        "metrics": metrics,
        "vif": vif_data,
        "figures": {
            "residual": fig_residual,
            "qq": fig_qq,
            "vif": fig_vif,
            "cooks": fig_cooks,
        },
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
    }


def _residual_plot(y_true: np.ndarray, y_pred: np.ndarray, resid: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=resid,
            mode="markers",
            marker=dict(size=6, opacity=0.6),
            name="Residuals",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Fitted (log Volume)",
        yaxis_title="Residual",
        **LAYOUT_DEFAULTS,
    )
    return fig


def _qq_plot(resid: np.ndarray) -> go.Figure:
    resid_clean = resid[np.isfinite(resid)]
    fig = go.Figure()

    if len(resid_clean) < 2:
        fig.add_annotation(
            text="Not enough residuals for Q-Q plot",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(title="Q-Q Plot (Residuals vs Normal)", **LAYOUT_DEFAULTS)
        return fig

    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(resid_clean)))
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=np.sort(resid_clean),
            mode="markers",
            marker=dict(size=6),
            name="Sample",
        )
    )
    q_min, q_max = theoretical.min(), theoretical.max()
    fig.add_trace(
        go.Scatter(
            x=[q_min, q_max],
            y=[q_min, q_max],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Normal",
        )
    )
    fig.update_layout(
        title="Q-Q Plot (Residuals vs Normal)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        **LAYOUT_DEFAULTS,
    )
    return fig


def _vif_bar_figure(vif_data: list[dict]) -> go.Figure:
    VIF_CAP = 100
    names = [d["feature"] for d in vif_data]
    vals = [min(d["vif"], VIF_CAP) if d["vif"] is not None else 0 for d in vif_data]
    raw_vals = [d["vif"] for d in vif_data]
    colors = ["#ea4335" if v > 10 else "#34a853" for v in vals]
    labels = []
    for v, raw in zip(vals, raw_vals):
        if raw is None:
            labels.append("N/A")
        elif raw > VIF_CAP:
            labels.append(f">{VIF_CAP}")
        else:
            labels.append(f"{v:.1f}")
    fig = go.Figure(
        go.Bar(x=names, y=vals, marker_color=colors, text=labels, textposition="outside")
    )
    fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="VIF=10")
    fig.update_layout(
        title="Variance Inflation Factor",
        xaxis_title="Feature",
        yaxis_title="VIF",
        xaxis_tickangle=-45,
        **LAYOUT_DEFAULTS,
    )
    return fig


def _cooks_figure(cooks_d: np.ndarray) -> go.Figure:
    fig = go.Figure()
    n = len(cooks_d)
    if n == 0:
        fig.add_annotation(
            text="No observations for Cook's distance",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(title="Cook's Distance (Influential Points)", **LAYOUT_DEFAULTS)
        return fig

    threshold = max(4 / n, 0.01)
    # Only plot top 500 observations sorted by Cook's D to keep JSON small
    if n > 500:
        top_idx = np.argsort(cooks_d)[-500:]
        top_idx = np.sort(top_idx)
        plot_x = top_idx.tolist()
        plot_y = cooks_d[top_idx].tolist()
        title_suffix = f" (top 500 of {n})"
    else:
        plot_x = list(range(n))
        plot_y = cooks_d.tolist()
        title_suffix = ""

    fig.add_trace(
        go.Bar(x=plot_x, y=plot_y, marker_color="#1a73e8", name="Cook's D")
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"4/n={threshold:.3f}")
    fig.update_layout(
        title=f"Cook's Distance (Influential Points){title_suffix}",
        xaxis_title="Observation",
        yaxis_title="Cook's D",
        **LAYOUT_DEFAULTS,
    )
    return fig
