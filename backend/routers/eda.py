import json

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fastapi import APIRouter, Query

from data.generator import build_all_datasets

router = APIRouter(prefix="/api/v1/eda", tags=["eda"])


def _fig_json(fig) -> dict:
    return json.loads(fig.to_json())


@router.get("/price_volume_scatter")
def price_volume_scatter(station_id: str | None = None):
    ds = build_all_datasets()
    df = ds["df_daily"].dropna(subset=["our_price", "volume_litres"])
    if station_id:
        df = df[df["station_id"] == station_id]
    else:
        df = df.sample(min(2000, len(df)), random_state=42)

    fig = px.scatter(df, x="our_price", y="volume_litres",
                     color="station_type" if "station_type" in df.columns else None,
                     title="Price vs Volume", opacity=0.5,
                     labels={"our_price": "Diesel Price (EUR/L)", "volume_litres": "Volume (L)"})
    return {"figure": _fig_json(fig)}


@router.get("/price_gap_distribution")
def price_gap_distribution():
    ds = build_all_datasets()
    df = ds["df_prices"].copy()
    df["price_gap"] = df["our_price"] - df["min_comp_price"]
    fig = px.histogram(df.dropna(subset=["price_gap"]), x="price_gap", nbins=60,
                       title="Price Gap Distribution (Our Price - Min Competitor)",
                       labels={"price_gap": "Price Gap (EUR)"})
    return {"figure": _fig_json(fig)}


@router.get("/volume_by_type")
def volume_by_type():
    ds = build_all_datasets()
    df = ds["df_daily"].dropna(subset=["volume_litres", "station_type"])
    fig = px.box(df, x="station_type", y="volume_litres",
                 title="Volume Distribution by Station Type",
                 labels={"station_type": "Station Type", "volume_litres": "Daily Volume (L)"})
    return {"figure": _fig_json(fig)}


@router.get("/price_timeseries")
def price_timeseries(station_id: str = "STN_001"):
    ds = build_all_datasets()
    df = ds["df_prices"][ds["df_prices"]["station_id"] == station_id].copy()
    if df.empty:
        return {"error": "Station not found", "figure": None}
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["our_price"], name="Our Price", mode="lines"))
    if "min_comp_price" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["min_comp_price"],
                                 name="Min Competitor", mode="lines", line=dict(dash="dash")))
    fig.update_layout(title=f"Price Time Series - {station_id}",
                      xaxis_title="Date", yaxis_title="Price (EUR/L)")
    return {"figure": _fig_json(fig)}


@router.get("/station_map")
def station_map():
    ds = build_all_datasets()
    df = ds["df_stations"]
    fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="station_type",
                            hover_name="station_id",
                            hover_data=["brand", "region", "n_competitors"],
                            title="Station Network Map", zoom=5,
                            mapbox_style="carto-positron")
    return {"figure": _fig_json(fig)}


@router.get("/crude_vs_price")
def crude_vs_price():
    ds = build_all_datasets()
    df = ds["df_market"]
    df_p = ds["df_prices"]
    avg_price = df_p.groupby("date")["our_price"].mean().reset_index()
    avg_price["date"] = pd.to_datetime(avg_price["date"])
    market = df.copy()
    market["date"] = pd.to_datetime(market["date"])
    merged = market.merge(avg_price, on="date", how="inner")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["crude_eur"], name="Brent Crude (ct/L)", yaxis="y"))
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["our_price"] * 100, name="Avg Diesel (ct/L)", yaxis="y2"))
    fig.update_layout(
        title="Crude Oil vs Retail Diesel Price",
        yaxis=dict(title="Brent Crude (ct/L)"),
        yaxis2=dict(title="Diesel Price (ct/L)", overlaying="y", side="right"),
    )
    return {"figure": _fig_json(fig)}


import pandas as pd
