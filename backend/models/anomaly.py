"""Anomaly detection for volume/price (M5): Isolation Forest + Shewhart control chart."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from data.generator import build_all_datasets


def _get_station_series(station_id: str | None, target: str = "volume_litres") -> pd.DataFrame:
    """Get daily time series for a station (or all stations aggregated)."""
    ds = build_all_datasets()
    df = ds["df_daily"].copy()
    df["date"] = pd.to_datetime(df["date"])

    if station_id:
        df = df[df["station_id"] == station_id].sort_values("date")
    else:
        df = df.groupby("date").agg({target: "mean"}).reset_index()

    df = df.dropna(subset=[target])
    return df


def _inject_synthetic_anomaly(
    df: pd.DataFrame, target: str, n_anomalies: int = 3, seed: int = 42
) -> pd.DataFrame:
    """Inject synthetic anomalies for testing."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    n = len(df)
    if n < 10:
        return df

    idx = rng.choice(n, size=min(n_anomalies, n // 5), replace=False)
    vals = df[target].values.copy()
    mean_val = np.nanmean(vals)
    std_val = np.nanstd(vals) or 1e-8

    for i in idx:
        # Random spike or dip
        if rng.random() > 0.5:
            vals[i] = mean_val + rng.uniform(2.5, 4.0) * std_val
        else:
            vals[i] = mean_val - rng.uniform(2.5, 4.0) * std_val

    df[target] = vals
    return df


def train_isolation_forest(
    contamination: float = 0.05,
    station_id: str | None = None,
    target: str = "volume_litres",
    inject_test_anomaly: bool = False,
) -> dict[str, Any]:
    """
    Train Isolation Forest on station volume (or price) series.
    Returns anomaly scores timeseries, flagged events list, control chart data.
    """
    df = _get_station_series(station_id, target)
    if len(df) < 10:
        return {
            "anomaly_scores": [],
            "flagged_events": [],
            "control_chart": {},
            "error": "Insufficient data (need at least 10 points)",
        }

    if inject_test_anomaly:
        df = _inject_synthetic_anomaly(df, target)

    # Features: target + lag + rolling stats
    df = df.sort_values("date").reset_index(drop=True)
    df["lag1"] = df[target].shift(1)
    df["rolling_mean"] = df[target].rolling(7, min_periods=1).mean()
    df["rolling_std"] = df[target].rolling(7, min_periods=1).std().fillna(0)

    feature_cols = [target, "lag1", "rolling_mean", "rolling_std"]
    X = df[feature_cols].fillna(0).values

    iso = IsolationForest(contamination=contamination, random_state=42)
    scores = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
    anomaly_scores = iso.decision_function(X)  # lower = more anomalous

    anomaly_mask = scores == -1
    flagged = df.loc[anomaly_mask, ["date", target]].copy()
    flagged["date"] = flagged["date"].astype(str)
    flagged["anomaly_score"] = anomaly_scores[anomaly_mask].tolist()
    flagged_events = flagged.to_dict("records")

    # Shewhart control chart data
    rolling_mean = df[target].rolling(7, min_periods=1).mean()
    rolling_std = df[target].rolling(7, min_periods=1).std().fillna(0)
    ucl = rolling_mean + 3 * rolling_std
    lcl = rolling_mean - 3 * rolling_std

    control_chart = {
        "dates": df["date"].astype(str).tolist(),
        "values": df[target].tolist(),
        "center": rolling_mean.tolist(),
        "ucl": ucl.tolist(),
        "lcl": lcl.tolist(),
    }

    return {
        "anomaly_scores": [
            {"date": str(d), "score": float(s)}
            for d, s in zip(df["date"], anomaly_scores)
        ],
        "flagged_events": flagged_events,
        "control_chart": control_chart,
        "n_flagged": int(anomaly_mask.sum()),
    }
