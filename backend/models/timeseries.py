"""Time series models (M6-M8): STL, ARIMA, Prophet, LightGBM with lag features."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import shap
except Exception:
    shap = None

from data.generator import build_all_datasets
from utils.chart_helpers import fig_to_response
from utils.metrics import forecast_metrics


def _get_station_daily(station_id: str, target: str = "volume_litres") -> pd.Series:
    """Get daily time series for a station."""
    ds = build_all_datasets()
    df = ds["df_daily"]
    df = df[df["station_id"] == station_id].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=[target])
    return df.set_index("date")[target]


def decompose_stl(station_id: str, period: int = 7) -> dict[str, Any]:
    """STL decomposition: trend, seasonal, residual. Returns Plotly figure as JSON."""
    y = _get_station_daily(station_id)
    if len(y) < 2 * period + 1:
        return {"error": f"Need at least {2 * period + 1} points for STL", "figure": None}

    try:
        stl = STL(y, period=period, seasonal=min(period * 2 - 1, 13))
        res = stl.fit()
    except Exception as e:
        return {"error": f"STL decomposition failed: {e}", "figure": None}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=res.observed, name="Observed", mode="lines"))
    fig.add_trace(go.Scatter(x=y.index, y=res.trend, name="Trend", mode="lines"))
    fig.add_trace(go.Scatter(x=y.index, y=res.seasonal, name="Seasonal", mode="lines"))
    fig.add_trace(go.Scatter(x=y.index, y=res.resid, name="Residual", mode="lines"))
    fig.update_layout(
        title=f"STL Decomposition (period={period}) - {station_id}",
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h"),
    )
    return {
        **fig_to_response(fig),
        "components": {
            "trend": res.trend.tolist(),
            "seasonal": res.seasonal.tolist(),
            "residual": res.resid.tolist(),
        },
        "dates": [str(d) for d in y.index],
    }


def train_arima(
    station_id: str,
    p: int = 1,
    d: int = 0,
    q: int = 1,
    forecast_horizon: int = 14,
) -> dict[str, Any]:
    """ARIMA forecast with prediction intervals. Returns Plotly figure as JSON."""
    y = _get_station_daily(station_id)
    y = y.ffill().bfill()
    if len(y) < 30:
        return {"error": "Need at least 30 points for ARIMA", "figure": None}

    try:
        model = ARIMA(y, order=(p, d, q))
        fitted = model.fit()
        fcast = fitted.get_forecast(steps=forecast_horizon)
        pred = fcast.predicted_mean
        ci = fcast.conf_int(alpha=0.1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y.index, y=y.values, name="Historical", mode="lines"))
        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Forecast", mode="lines"))
        fig.add_trace(
            go.Scatter(
                x=ci.index.tolist() + ci.index.tolist()[::-1],
                y=ci.iloc[:, 0].tolist() + ci.iloc[:, 1].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(26,115,232,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="80% CI",
            )
        )
        fig.update_layout(
            title=f"ARIMA({p},{d},{q}) Forecast - {station_id}",
            xaxis_title="Date",
            yaxis_title="Volume (L)",
        )
        return {
            **fig_to_response(fig),
            "forecast": [{"date": str(d), "value": float(v)} for d, v in zip(pred.index, pred.values)],
            "intervals": [
                {"date": str(d), "lower": float(ci.loc[d].iloc[0]), "upper": float(ci.loc[d].iloc[1])}
                for d in pred.index
            ],
        }
    except Exception as e:
        return {"error": str(e), "figure": None}


def train_prophet(
    station_id: str,
    holidays: bool = True,
    changepoint_scale: float = 0.05,
    horizon: int = 14,
) -> dict[str, Any]:
    """Prophet forecast with trend/seasonality components. Returns Plotly figure as JSON."""
    if Prophet is None:
        return {
            "error": (
                "Prophet is unavailable in this environment. "
                "Install prophet or run ARIMA/LightGBM alternatives."
            ),
            "figure": None,
        }
    y = _get_station_daily(station_id)
    y = y.ffill().bfill()
    if len(y) < 30:
        return {"error": "Need at least 30 points for Prophet", "figure": None}

    df_prophet = pd.DataFrame({"ds": y.index, "y": y.values})

    try:
        m = Prophet(
            changepoint_prior_scale=changepoint_scale,
            yearly_seasonality=True,
            weekly_seasonality=True,
        )
        if holidays:
            m.add_country_holidays(country_name="DE")
        m.fit(df_prophet)

        future = m.make_future_dataframe(periods=horizon)
        fcast = m.predict(future)
    except Exception as e:
        return {"error": f"Prophet fitting failed: {e}", "figure": None}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], name="Historical", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=fcast["ds"],
            y=fcast["yhat"],
            name="Forecast",
            mode="lines",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fcast["ds"].tolist() + fcast["ds"].tolist()[::-1],
            y=fcast["yhat_lower"].tolist() + fcast["yhat_upper"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(26,115,232,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Uncertainty",
        )
    )
    fig.update_layout(
        title=f"Prophet Forecast - {station_id}",
        xaxis_title="Date",
        yaxis_title="Volume (L)",
    )

    components = {}
    if "trend" in fcast.columns:
        components["trend"] = fcast["trend"].tolist()
    if "weekly" in fcast.columns:
        components["weekly"] = fcast["weekly"].tolist()
    if "yearly" in fcast.columns:
        components["yearly"] = fcast["yearly"].tolist()

    return {
        **fig_to_response(fig),
        "forecast": [{"date": str(d), "value": float(v)} for d, v in zip(fcast["ds"].tail(horizon), fcast["yhat"].tail(horizon))],
        "components": components,
    }


def train_lightgbm_lag(
    station_id: str,
    lag_features: list[int] | None = None,
    horizon: int = 14,
) -> dict[str, Any]:
    """LightGBM with lag features. Returns forecast + SHAP importance."""
    if lgb is None:
        return {"error": "LightGBM is unavailable in this environment", "figure": None}
    lag_features = lag_features or [1, 7, 14]
    y = _get_station_daily(station_id)
    y = y.ffill().bfill()
    if len(y) < max(lag_features) + horizon + 10:
        return {"error": "Insufficient data for lag features", "figure": None}

    # Build lag matrix
    df = pd.DataFrame({"y": y})
    for lag in lag_features:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_mean_7"] = df["y"].rolling(7, min_periods=1).mean().shift(1)
    df["rolling_std_7"] = df["y"].rolling(7, min_periods=1).std().shift(1).fillna(0)
    df = df.dropna()

    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols]
    y_target = df["y"]

    try:
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
        model.fit(X, y_target)
    except Exception as e:
        return {"error": f"LightGBM training failed: {e}", "figure": None}

    # SHAP
    if shap is not None:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        importance = dict(zip(feature_cols, np.abs(shap_vals).mean(axis=0).tolist()))
    else:
        # Graceful fallback when SHAP isn't installed.
        raw_importance = model.feature_importances_
        denom = float(np.sum(raw_importance)) or 1.0
        importance = dict(
            zip(feature_cols, [float(v) / denom for v in raw_importance])
        )

    # Forecast (recursive)
    hist = df["y"].tolist()
    preds = []
    for _ in range(horizon):
        row = {}
        for lag in lag_features:
            if len(hist) >= lag:
                row[f"lag_{lag}"] = hist[-lag]
            else:
                row[f"lag_{lag}"] = hist[-1] if hist else 0
        recent = hist[-7:] if len(hist) >= 7 else hist
        row["rolling_mean_7"] = float(np.mean(recent)) if recent else 0.0
        row["rolling_std_7"] = float(np.std(recent)) if len(recent) > 1 else 0.0
        X_last = pd.DataFrame([row])[feature_cols]
        pred = model.predict(X_last)[0]
        if not np.isfinite(pred):
            pred = hist[-1] if hist else 0.0
        preds.append(pred)
        hist.append(pred)

    future_dates = pd.date_range(start=y.index[-1], periods=horizon + 1, freq="D")[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y.values, name="Historical", mode="lines"))
    fig.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast", mode="lines", line=dict(dash="dash")))
    fig.update_layout(
        title=f"LightGBM Lag Forecast - {station_id}",
        xaxis_title="Date",
        yaxis_title="Volume (L)",
    )

    return {
        **fig_to_response(fig),
        "forecast": [{"date": str(d), "value": float(v)} for d, v in zip(future_dates, preds)],
        "shap_importance": importance,
    }


def walk_forward_validation(
    station_id: str,
    model_type: str = "arima",
    n_splits: int = 3,
    horizon: int = 7,
) -> dict[str, Any]:
    """Walk-forward validation for honest evaluation."""
    y = _get_station_daily(station_id)
    y = y.ffill().bfill()
    if len(y) < (n_splits + 1) * horizon:
        return {"error": "Insufficient data for walk-forward", "metrics": []}

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)
    all_metrics = []

    for split, (train_idx, test_idx) in enumerate(tscv.split(y)):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if model_type == "arima":
            try:
                model = ARIMA(y_train, order=(1, 0, 1))
                fitted = model.fit()
                pred = fitted.forecast(steps=len(y_test))
            except Exception:
                pred = np.full(len(y_test), y_train.mean())
        elif model_type == "lightgbm":
            if lgb is None:
                pred = np.full(len(y_test), y_train.mean())
                m = forecast_metrics(y_test.values, pred)
                m["split"] = split
                m["warning"] = "LightGBM unavailable; used mean baseline."
                all_metrics.append(m)
                continue
            df = pd.DataFrame({"y": y_train})
            for lag in [1, 7]:
                df[f"lag_{lag}"] = df["y"].shift(lag)
            df = df.dropna()
            X = df[["lag_1", "lag_7"]]
            y_tr = df["y"]
            model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbosity=-1)
            model.fit(X, y_tr)
            pred_list = []
            hist = y_train.tolist()
            for _ in range(len(y_test)):
                lag1 = hist[-1] if hist else 0
                lag7 = hist[-7] if len(hist) >= 7 else hist[-1] if hist else 0
                X_pred = pd.DataFrame([{"lag_1": lag1, "lag_7": lag7}])
                p = model.predict(X_pred)[0]
                pred_list.append(p)
                hist.append(p)
            pred = np.array(pred_list)
        else:
            pred = np.full(len(y_test), y_train.mean())

        m = forecast_metrics(y_test.values, pred)
        m["split"] = split
        all_metrics.append(m)

    return {"metrics": all_metrics, "model_type": model_type}
