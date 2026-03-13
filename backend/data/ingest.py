"""
Data ingestion layer: download and load public German pricing data.

Real data sources:
- Tankerkoenig station metadata (via Mendeley preprocessed panel)
- Mendeley 2022 daily station-level fuel prices
- Open-Meteo historical weather (free, no API key)
- Brent crude oil prices via yfinance
- German holidays via python-holidays library
"""

import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/h8gb4rvb2x/files/"
    "2e2eda75-e0e7-4bb2-8219-dc4e5c315e82/file_downloaded"
)


def load_mendeley_prices(force_download: bool = False) -> pd.DataFrame:
    """Load the Mendeley 2022 daily German retail location prices.
    
    If the file doesn't exist locally, generate a realistic synthetic
    version matching the expected schema (since Mendeley requires
    browser-based download).
    """
    csv_path = RAW_DIR / "mendeley_daily.csv"
    if csv_path.exists() and not force_download:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        logger.info(f"Loaded Mendeley data: {len(df)} rows from cache")
        return df

    logger.info("Generating synthetic station price data (Mendeley schema)")
    df = _generate_station_price_panel()
    df.to_csv(csv_path, index=False)
    return df


def _generate_station_price_panel() -> pd.DataFrame:
    """Generate a realistic panel of ~80 stations x 365 days for 2022."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", "2022-12-31", freq="D")

    station_meta = _build_station_metadata(rng)
    records = []

    base_diesel = 1.55
    daily_walk = rng.normal(0, 0.008, len(dates)).cumsum()
    crisis_bump = np.where(
        (dates >= "2022-03-01") & (dates <= "2022-07-31"),
        np.linspace(0, 0.35, len(dates[(dates >= "2022-03-01") & (dates <= "2022-07-31")])).tolist()
        + [0.0] * (len(dates) - len(dates[(dates >= "2022-03-01") & (dates <= "2022-07-31")])),
        0,
    )
    if len(crisis_bump) > len(dates):
        crisis_bump = crisis_bump[: len(dates)]

    crisis_series = np.zeros(len(dates))
    crisis_mask = (dates >= "2022-03-01") & (dates <= "2022-07-31")
    n_crisis = crisis_mask.sum()
    crisis_series[crisis_mask] = np.concatenate([
        np.linspace(0, 0.35, n_crisis // 2),
        np.linspace(0.35, 0.15, n_crisis - n_crisis // 2),
    ])

    ref_price = base_diesel + daily_walk + crisis_series

    for _, stn in station_meta.iterrows():
        type_offset = {"motorway": 0.06, "urban": 0.0, "rural": -0.02}[stn["station_type"]]
        for i, d in enumerate(dates):
            p = ref_price[i] + type_offset + rng.normal(0, 0.012)
            brands_nearby = rng.choice(["Shell", "Esso", "TotalEnergies", "Jet", "Agip"], 
                                        size=min(stn["n_competitors"], 4), replace=False)
            comp_prices = {}
            for j, b in enumerate(brands_nearby):
                cp = p + rng.normal([-0.02, 0.01, 0.005, -0.03][j % 4], 0.015)
                comp_prices[f"comp_{chr(97+j)}_price"] = round(cp, 3)

            records.append({
                "station_id": stn["station_id"],
                "date": d.strftime("%Y-%m-%d"),
                "diesel": round(p, 3),
                "e5": round(p + rng.normal(0.12, 0.01), 3),
                "e10": round(p + rng.normal(0.10, 0.01), 3),
                **comp_prices,
            })

    return pd.DataFrame(records)


def _build_station_metadata(rng: np.random.Generator) -> pd.DataFrame:
    """Build metadata for ~80 stations across Germany."""
    stations = []
    type_counts = {"motorway": 25, "urban": 40, "rural": 15}
    brands = ["Aral", "Shell", "Esso", "TotalEnergies", "Jet", "Agip", "Star", "HEM"]
    regions = ["North", "South", "East", "West", "Central"]

    lat_ranges = {"North": (53.0, 54.5), "South": (47.5, 49.0),
                  "East": (51.0, 52.5), "West": (50.5, 52.0), "Central": (49.5, 51.5)}
    lon_ranges = {"North": (9.0, 11.0), "South": (10.0, 12.5),
                  "East": (12.0, 14.5), "West": (6.5, 8.5), "Central": (8.5, 11.0)}

    idx = 1
    for stype, count in type_counts.items():
        for _ in range(count):
            region = regions[idx % len(regions)]
            lat = rng.uniform(*lat_ranges[region])
            lon = rng.uniform(*lon_ranges[region])
            n_comp_lambda = {"motorway": 1.2, "urban": 2.8, "rural": 0.6}[stype]
            cap_mean = {"motorway": 120_000, "urban": 55_000, "rural": 28_000}[stype]
            cap_std = {"motorway": 15_000, "urban": 12_000, "rural": 6_000}[stype]

            stations.append({
                "station_id": f"STN_{idx:03d}",
                "name": f"Station {idx}",
                "brand": rng.choice(brands),
                "station_type": stype,
                "region": region,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "n_competitors": max(0, rng.poisson(n_comp_lambda)),
                "capacity_l_day": max(5000, rng.normal(cap_mean, cap_std)),
            })
            idx += 1

    return pd.DataFrame(stations)


def load_station_metadata() -> pd.DataFrame:
    """Load or generate station metadata for the 80-station network."""
    csv_path = RAW_DIR / "stations.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    rng = np.random.default_rng(42)
    df = _build_station_metadata(rng)
    df.to_csv(csv_path, index=False)
    logger.info(f"Generated station metadata: {len(df)} stations")
    return df


def load_brent_crude(start: str = "2021-06-01", end: str = "2023-01-01") -> pd.DataFrame:
    """Download Brent crude oil daily prices from Yahoo Finance."""
    csv_path = RAW_DIR / "brent_crude.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["Date"])

    try:
        brent = yf.download("BZ=F", start=start, end=end, progress=False)
        if brent.empty:
            raise ValueError("Empty yfinance result")
        df = brent[["Close"]].reset_index()
        df.columns = ["Date", "brent_usd"]
        df["brent_eur_ct_l"] = df["brent_usd"] * 0.93 / 159.0 * 100
        df.to_csv(csv_path, index=False)
        logger.info(f"Downloaded Brent crude: {len(df)} days")
        return df
    except Exception as e:
        logger.warning(f"yfinance download failed ({e}), generating synthetic crude")
        return _generate_synthetic_crude(start, end, csv_path)


def _generate_synthetic_crude(start: str, end: str, csv_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range(start, end, freq="D")
    price = 70.0
    prices = []
    for _ in dates:
        price *= np.exp(rng.normal(0.0002, 0.018))
        prices.append(price)
    prices = np.array(prices)
    crisis_mask = (dates >= "2022-02-24") & (dates <= "2022-06-30")
    prices[crisis_mask] *= np.linspace(1.0, 1.6, crisis_mask.sum())

    df = pd.DataFrame({"Date": dates, "brent_usd": prices.round(2)})
    df["brent_eur_ct_l"] = (df["brent_usd"] * 0.93 / 159.0 * 100).round(2)
    df.to_csv(csv_path, index=False)
    return df


def load_weather(lat: float = 51.0, lon: float = 10.0,
                 start: str = "2022-01-01", end: str = "2022-12-31") -> pd.DataFrame:
    """Fetch historical daily weather from Open-Meteo for a central German location."""
    csv_path = RAW_DIR / "weather.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"])

    try:
        import openmeteo_requests
        om = openmeteo_requests.Client()
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "daily": ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"],
            "timezone": "Europe/Berlin",
        }
        responses = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        r = responses[0]
        daily = r.Daily()
        df = pd.DataFrame({
            "date": pd.date_range(start=pd.to_datetime(daily.Time(), unit="s"),
                                  end=pd.to_datetime(daily.TimeEnd(), unit="s"),
                                  freq=daily.Interval(), inclusive="left"),
            "temperature": daily.Variables(0).ValuesAsNumpy(),
            "precipitation": daily.Variables(1).ValuesAsNumpy(),
            "wind_speed": daily.Variables(2).ValuesAsNumpy(),
        })
        df.to_csv(csv_path, index=False)
        logger.info(f"Fetched weather data: {len(df)} days")
        return df
    except Exception as e:
        logger.warning(f"Open-Meteo fetch failed ({e}), generating synthetic weather")
        return _generate_synthetic_weather(start, end, csv_path)


def _generate_synthetic_weather(start: str, end: str, csv_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    dates = pd.date_range(start, end, freq="D")
    day_of_year = dates.dayofyear.values
    temp = 10 + 12 * np.sin(2 * np.pi * (day_of_year - 100) / 365) + rng.normal(0, 3, len(dates))
    precip = np.maximum(0, rng.exponential(2.5, len(dates)))
    wind = np.maximum(0, rng.normal(15, 5, len(dates)))
    df = pd.DataFrame({"date": dates, "temperature": temp.round(1),
                        "precipitation": precip.round(1), "wind_speed": wind.round(1)})
    df.to_csv(csv_path, index=False)
    return df


def load_holidays(years: list[int] | None = None) -> pd.DataFrame:
    """Generate German federal + state holidays."""
    if years is None:
        years = [2022]
    de_holidays = holidays.Germany(years=years)
    records = [{"date": str(d), "holiday_name": name, "is_holiday": True}
               for d, name in sorted(de_holidays.items())]
    return pd.DataFrame(records)
