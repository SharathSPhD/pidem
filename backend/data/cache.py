import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, inspect

from config import settings

logger = logging.getLogger(__name__)

_cache: dict[str, pd.DataFrame] = {}
_engine = None
_redis_client = None

_DATASET_TABLES = {
    "df_stations": "stations",
    "df_prices": "prices",
    "df_volume": "volumes",
    "df_market": "market",
    "df_daily": "daily_panel",
    "df_hourly": "hourly_panel",
}


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    try:
        _engine = create_engine(settings.database_url, future=True)
        with _engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        logger.info("Connected to primary database")
        return _engine
    except Exception as exc:
        logger.warning("Primary database unavailable (%s). Falling back to SQLite.", exc)
        db_path = Path(__file__).parent / "pidem.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = settings.sqlite_fallback_url
        if sqlite_url.endswith("pidem.db"):
            sqlite_url = f"sqlite:///{db_path}"
        _engine = create_engine(sqlite_url, future=True)
        return _engine


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis

        client = redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        _redis_client = client
        logger.info("Connected to Redis cache")
    except Exception as exc:
        logger.warning("Redis unavailable (%s). Continuing without Redis cache.", exc)
        _redis_client = False
    return _redis_client


def get(key: str) -> Optional[pd.DataFrame]:
    if key in _cache:
        return _cache.get(key)

    redis_client = _get_redis()
    if redis_client:
        try:
            raw = redis_client.get(f"pidem:{key}")
            if raw:
                df = pd.DataFrame(json.loads(raw))
                _cache[key] = df
                return df
        except Exception:
            logger.exception("Failed reading %s from Redis", key)
    return None


def put(key: str, df: pd.DataFrame) -> None:
    _cache[key] = df
    redis_client = _get_redis()
    if redis_client:
        try:
            redis_client.set(f"pidem:{key}", df.to_json(orient="records", date_format="iso"))
        except Exception:
            logger.exception("Failed writing %s to Redis", key)


def clear() -> None:
    _cache.clear()


def keys() -> list[str]:
    return list(_cache.keys())


def has(key: str) -> bool:
    return key in _cache


def persist_all(datasets: dict[str, pd.DataFrame]) -> None:
    engine = _get_engine()
    for key, table in _DATASET_TABLES.items():
        df = datasets.get(key)
        if df is None:
            continue
        df.to_sql(table, engine, if_exists="replace", index=False)


def load_all_from_db() -> Optional[dict[str, pd.DataFrame]]:
    engine = _get_engine()
    inspector = inspect(engine)
    required_tables = list(_DATASET_TABLES.values())
    existing = set(inspector.get_table_names())
    if not all(t in existing for t in required_tables):
        return None

    data: dict[str, pd.DataFrame] = {}
    for key, table in _DATASET_TABLES.items():
        data[key] = pd.read_sql_table(table, engine)
    return data
