from pydantic import BaseModel
from enum import Enum


class StationType(str, Enum):
    MOTORWAY = "motorway"
    URBAN = "urban"
    RURAL = "rural"


class StationSchema(BaseModel):
    station_id: str
    name: str
    brand: str
    station_type: StationType
    region: str
    lat: float
    lon: float
    n_competitors: int
    capacity_l_day: float


class PriceRecord(BaseModel):
    station_id: str
    date: str
    our_price: float
    cogs: float
    gross_margin: float
    comp_a_price: float | None = None
    comp_b_price: float | None = None
    comp_c_price: float | None = None
    comp_d_price: float | None = None
    min_comp_price: float | None = None


class VolumeRecord(BaseModel):
    station_id: str
    date: str
    volume_litres: float
    gross_profit_eur: float


class MarketRecord(BaseModel):
    date: str
    crude_eur: float
    cpi_index: float | None = None
    temperature: float | None = None
    highway_index: float | None = None
    is_holiday: bool = False


class DatasetInfo(BaseModel):
    n_stations: int
    n_days: int
    date_range: tuple[str, str]
    station_types: dict[str, int]
    total_price_records: int
    total_volume_records: int
