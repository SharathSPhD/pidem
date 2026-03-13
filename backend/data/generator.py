"""
Hybrid data generator: merges real public data with synthetic overlays.

Real: station metadata, competitor prices, weather, crude oil, holidays
Synthetic: own-brand volumes, margins/COGS, competitive response behavior
"""

import logging

import numpy as np
import pandas as pd

from data import cache
from data.ingest import (
    load_brent_crude,
    load_holidays,
    load_mendeley_prices,
    load_station_metadata,
    load_weather,
)

logger = logging.getLogger(__name__)

ELASTICITY = {"motorway": -1.8, "urban": -1.3, "rural": -0.9}
OWN_PRICE_ELASTICITY = -0.4
SEASONALITY_SUMMER_MOTORWAY = 0.35
SEASONALITY_JANUARY = -0.15
SEASONALITY_FRIDAY_PM = 0.20


def build_all_datasets(force: bool = False) -> dict[str, pd.DataFrame]:
    """Build, persist, and retrieve all 6 core DataFrames.

    Priority:
    1) In-memory/Redis cache
    2) Persistent database tables
    3) Regenerate from ingest layer (one-time) and persist
    """
    if not force and cache.has("df_stations"):
        return {k: cache.get(k) for k in
                ["df_stations", "df_prices", "df_volume", "df_market", "df_daily", "df_hourly"]}

    if not force:
        db_datasets = cache.load_all_from_db()
        if db_datasets:
            logger.info("Loaded datasets from persistent database tables")
            for k, v in db_datasets.items():
                cache.put(k, v)
            return db_datasets

    logger.info("Building datasets...")
    df_stations = load_station_metadata()
    df_raw_prices = load_mendeley_prices()
    df_crude = load_brent_crude()
    df_weather = load_weather()
    df_holidays = load_holidays()

    df_prices = _build_prices(df_stations, df_raw_prices, df_crude)
    df_market = _build_market(df_crude, df_weather, df_holidays)
    df_volume = _build_volumes(df_stations, df_prices, df_market)
    df_daily = _build_daily(df_stations, df_prices, df_volume, df_market)
    df_hourly = _build_hourly_sample(df_daily)

    for k, v in [("df_stations", df_stations), ("df_prices", df_prices),
                 ("df_volume", df_volume), ("df_market", df_market),
                 ("df_daily", df_daily), ("df_hourly", df_hourly)]:
        cache.put(k, v)

    # One-time persistence; subsequent app restarts reuse DB tables.
    try:
        cache.persist_all({
            "df_stations": df_stations,
            "df_prices": df_prices,
            "df_volume": df_volume,
            "df_market": df_market,
            "df_daily": df_daily,
            "df_hourly": df_hourly,
        })
        logger.info("Persisted datasets to database")
    except Exception:
        logger.exception("Failed to persist datasets to database")

    logger.info(f"Datasets built: {len(df_stations)} stations, "
                f"{len(df_prices)} price records, {len(df_volume)} volume records")
    return {"df_stations": df_stations, "df_prices": df_prices,
            "df_volume": df_volume, "df_market": df_market,
            "df_daily": df_daily, "df_hourly": df_hourly}


def _build_prices(df_stations: pd.DataFrame, df_raw: pd.DataFrame,
                  df_crude: pd.DataFrame) -> pd.DataFrame:
    """Build price DataFrame with real competitor prices + synthetic margins."""
    rng = np.random.default_rng(42)
    station_ids = df_stations["station_id"].values
    dates = sorted(df_raw["date"].unique()) if "date" in df_raw.columns else pd.date_range("2022-01-01", "2022-12-31", freq="D")

    records = []
    crude_lookup = {}
    if not df_crude.empty:
        for _, row in df_crude.iterrows():
            d = str(row["Date"])[:10] if "Date" in df_crude.columns else str(row.iloc[0])[:10]
            crude_lookup[d] = row.get("brent_eur_ct_l", 45.0)

    for sid in station_ids:
        stn_rows = df_raw[df_raw["station_id"] == sid] if "station_id" in df_raw.columns else pd.DataFrame()
        stype = df_stations.loc[df_stations["station_id"] == sid, "station_type"].iloc[0]

        for d in dates:
            d_str = str(d)[:10]
            crude_ct = crude_lookup.get(d_str, 45.0)
            cogs = crude_ct * 0.85 + rng.normal(8, 1.5)

            if not stn_rows.empty and d_str in stn_rows["date"].astype(str).values:
                row = stn_rows[stn_rows["date"].astype(str) == d_str].iloc[0]
                our_price = row.get("diesel", 1.65 + rng.normal(0, 0.02))
            else:
                base = 1.55 + crude_ct / 100 * 0.3
                type_offset = {"motorway": 0.06, "urban": 0.0, "rural": -0.02}[stype]
                our_price = base + type_offset + rng.normal(0, 0.012)

            comp_cols = [c for c in df_raw.columns if c.startswith("comp_")] if not stn_rows.empty else []
            comp_prices = {}
            for c in comp_cols[:4]:
                if not stn_rows.empty and d_str in stn_rows["date"].astype(str).values:
                    cp = stn_rows[stn_rows["date"].astype(str) == d_str].iloc[0].get(c)
                    if pd.notna(cp):
                        comp_prices[c] = float(cp)
                        continue
                comp_prices[c] = our_price + rng.normal([-0.02, 0.01, 0.005, -0.03][len(comp_prices) % 4], 0.015)

            min_comp = min(comp_prices.values()) if comp_prices else our_price
            gross_margin = our_price * 100 - cogs

            records.append({
                "station_id": sid, "date": d_str,
                "our_price": round(our_price, 3),
                "cogs": round(cogs, 2),
                "gross_margin": round(gross_margin, 2),
                "min_comp_price": round(min_comp, 3),
                **{k: round(v, 3) for k, v in comp_prices.items()},
            })

    return pd.DataFrame(records)


def _build_market(df_crude: pd.DataFrame, df_weather: pd.DataFrame,
                  df_holidays: pd.DataFrame) -> pd.DataFrame:
    """Build market DataFrame merging crude, weather, and holidays."""
    dates = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    df = pd.DataFrame({"date": dates})

    if not df_crude.empty:
        crude = df_crude.copy()
        crude["date"] = pd.to_datetime(crude["Date"]).dt.date
        df["date_key"] = df["date"].dt.date
        df = df.merge(crude[["date", "brent_eur_ct_l"]].rename(columns={"date": "date_key"}),
                       on="date_key", how="left")
        df["crude_eur"] = df["brent_eur_ct_l"].ffill().bfill()
        df.drop(columns=["date_key", "brent_eur_ct_l"], inplace=True, errors="ignore")
    else:
        df["crude_eur"] = 45.0

    if not df_weather.empty:
        weather = df_weather.copy()
        weather["date"] = pd.to_datetime(weather["date"])
        df = df.merge(weather[["date", "temperature"]], on="date", how="left")
        df["temperature"] = df["temperature"].ffill().bfill()
    else:
        df["temperature"] = 10.0

    rng = np.random.default_rng(99)
    day_of_year = df["date"].dt.dayofyear.values
    df["highway_index"] = (
        80 + 20 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
        + 10 * (df["date"].dt.dayofweek.values < 5).astype(float)
        + rng.normal(0, 5, len(df))
    ).round(1)

    holiday_dates = set(df_holidays["date"].astype(str).values) if not df_holidays.empty else set()
    df["is_holiday"] = df["date"].dt.strftime("%Y-%m-%d").isin(holiday_dates)
    df["cpi_index"] = 100 + np.linspace(0, 8, len(df)).round(1)

    return df


def _build_volumes(df_stations: pd.DataFrame, df_prices: pd.DataFrame,
                   df_market: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic volumes using the hidden log-linear demand model."""
    rng = np.random.default_rng(314)
    records = []

    market_lookup = {}
    for _, row in df_market.iterrows():
        market_lookup[str(row["date"])[:10]] = row

    prev_vol = {}

    for _, stn in df_stations.iterrows():
        sid = stn["station_id"]
        stype = stn["station_type"]
        alpha = np.log(stn["capacity_l_day"] * 0.6) + rng.normal(0, 0.3)
        beta1 = ELASTICITY[stype]
        beta2 = OWN_PRICE_ELASTICITY
        sigma = 0.08 if stype != "rural" else 0.12

        stn_prices = df_prices[df_prices["station_id"] == sid].sort_values("date")

        for _, pr in stn_prices.iterrows():
            d_str = str(pr["date"])[:10]
            mkt = market_lookup.get(d_str, {})
            price_gap = pr["our_price"] - pr.get("min_comp_price", pr["our_price"])
            dt = pd.Timestamp(d_str)

            month_effect = -0.05 * np.cos(2 * np.pi * (dt.month - 7) / 12)
            dow_effect = 0.05 if dt.dayofweek == 4 else (-0.03 if dt.dayofweek == 6 else 0.0)
            holiday_effect = -0.08 if mkt.get("is_holiday", False) else 0.0

            temp = mkt.get("temperature", 10.0) if isinstance(mkt, dict) else getattr(mkt, "temperature", 10.0)
            temp_val = float(temp) if not isinstance(temp, (pd.Series,)) else 10.0
            temp_effect = 0.003 * (temp_val - 10)

            traffic = mkt.get("highway_index", 90.0) if isinstance(mkt, dict) else getattr(mkt, "highway_index", 90.0)
            traffic_val = float(traffic) if not isinstance(traffic, (pd.Series,)) else 90.0
            traffic_effect = 0.002 * (traffic_val - 90) if stype == "motorway" else 0.0

            prev = prev_vol.get(sid, np.exp(alpha))
            ar_term = 0.3 * np.log(max(prev, 100))

            log_v = (alpha + beta1 * price_gap + beta2 * pr["our_price"]
                     + ar_term + month_effect + dow_effect + holiday_effect
                     + temp_effect + traffic_effect + rng.normal(0, sigma))

            if rng.random() < 0.003:
                log_v += rng.choice([-0.5, 0.4, -0.8, 0.6])

            volume = max(500, np.exp(log_v))
            gross_profit = volume * pr["gross_margin"] / 100

            prev_vol[sid] = volume
            records.append({
                "station_id": sid, "date": d_str,
                "volume_litres": round(volume, 0),
                "gross_profit_eur": round(gross_profit, 2),
            })

    return pd.DataFrame(records)


def _build_daily(df_stations: pd.DataFrame, df_prices: pd.DataFrame,
                 df_volume: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
    """Merge prices + volumes + market into a daily panel."""
    df = df_prices.merge(df_volume, on=["station_id", "date"], how="left")
    df["date_dt"] = pd.to_datetime(df["date"])
    market = df_market.copy()
    market["date_dt"] = pd.to_datetime(market["date"])
    df = df.merge(market[["date_dt", "crude_eur", "temperature", "highway_index", "is_holiday"]],
                  on="date_dt", how="left")
    df = df.merge(df_stations[["station_id", "station_type", "region", "n_competitors"]],
                  on="station_id", how="left")
    df.drop(columns=["date_dt"], inplace=True)
    return df


def _build_hourly_sample(df_daily: pd.DataFrame, hours_per_day: int = 24) -> pd.DataFrame:
    """Expand a sample of daily data into hourly for time-series modules.
    
    Only expands a subset (first 5 stations, last 30 days) to keep memory reasonable.
    """
    rng = np.random.default_rng(555)
    sample_stations = df_daily["station_id"].unique()[:5]
    sample = df_daily[df_daily["station_id"].isin(sample_stations)].tail(5 * 30)

    records = []
    hourly_pattern = np.array([
        0.2, 0.15, 0.1, 0.1, 0.15, 0.4, 0.8, 1.2, 1.1, 0.9,
        0.85, 0.95, 1.0, 0.9, 0.85, 1.0, 1.3, 1.5, 1.3, 1.0,
        0.8, 0.6, 0.4, 0.3,
    ])
    hourly_pattern /= hourly_pattern.sum()

    for _, row in sample.iterrows():
        daily_vol = row.get("volume_litres", 30000)
        if pd.isna(daily_vol):
            daily_vol = 30000
        for h in range(hours_per_day):
            hourly_vol = daily_vol * hourly_pattern[h] * (1 + rng.normal(0, 0.05))
            records.append({
                "station_id": row["station_id"],
                "datetime": f"{row['date']} {h:02d}:00:00",
                "our_price": row.get("our_price", 1.65),
                "volume_litres": round(max(0, hourly_vol), 0),
            })

    return pd.DataFrame(records)
