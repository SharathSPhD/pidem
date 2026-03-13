from fastapi import APIRouter, Query

from data.generator import build_all_datasets
from data.schemas import DatasetInfo

router = APIRouter(prefix="/api/v1/data", tags=["data"])


@router.get("/info", response_model=DatasetInfo)
def dataset_info():
    ds = build_all_datasets()
    df_s = ds["df_stations"]
    df_p = ds["df_prices"]
    df_v = ds["df_volume"]
    dates = sorted(df_p["date"].unique())
    return DatasetInfo(
        n_stations=len(df_s),
        n_days=len(dates),
        date_range=(dates[0], dates[-1]),
        station_types=df_s["station_type"].value_counts().to_dict(),
        total_price_records=len(df_p),
        total_volume_records=len(df_v),
    )


@router.get("/stations")
def get_stations():
    ds = build_all_datasets()
    return ds["df_stations"].to_dict(orient="records")


@router.get("/stations/{station_id}")
def get_station(station_id: str):
    ds = build_all_datasets()
    row = ds["df_stations"][ds["df_stations"]["station_id"] == station_id]
    if row.empty:
        return {"error": f"Station {station_id} not found"}
    return row.iloc[0].to_dict()


@router.get("/prices")
def get_prices(station_id: str | None = None, limit: int = Query(default=500, le=10000)):
    ds = build_all_datasets()
    df = ds["df_prices"]
    if station_id:
        df = df[df["station_id"] == station_id]
    return df.head(limit).to_dict(orient="records")


@router.get("/volume")
def get_volume(station_id: str | None = None, limit: int = Query(default=500, le=10000)):
    ds = build_all_datasets()
    df = ds["df_volume"]
    if station_id:
        df = df[df["station_id"] == station_id]
    return df.head(limit).to_dict(orient="records")


@router.get("/market")
def get_market():
    ds = build_all_datasets()
    return ds["df_market"].to_dict(orient="records")


@router.get("/daily")
def get_daily(station_id: str | None = None, limit: int = Query(default=1000, le=50000)):
    ds = build_all_datasets()
    df = ds["df_daily"]
    if station_id:
        df = df[df["station_id"] == station_id]
    return df.head(limit).to_dict(orient="records")


@router.get("/hourly")
def get_hourly(station_id: str | None = None, limit: int = Query(default=1000, le=50000)):
    ds = build_all_datasets()
    df = ds["df_hourly"]
    if station_id:
        df = df[df["station_id"] == station_id]
    return df.head(limit).to_dict(orient="records")
