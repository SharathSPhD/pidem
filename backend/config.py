from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "Pricing Intelligence Lab"
    debug: bool = True

    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql+psycopg://localhost:5432/pidem"
    sqlite_fallback_url: str = "sqlite:///./data/pidem.db"
    nim_url: str = "http://localhost:8001"

    hf_token: str = ""
    wandb_api_key: str = ""
    ngc_api_key: str = ""

    data_dir: Path = Path(__file__).parent / "data" / "raw"
    n_stations: int = 80
    data_years: tuple[int, ...] = (2022,)

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
