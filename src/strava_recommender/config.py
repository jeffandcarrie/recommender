from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    strava_client_id: int
    strava_client_secret: str
    strava_refresh_token: str = ""
    data_dir: Path = Path.home() / ".strava_recommender"
    cluster_epsilon_meters: float = 500.0
    cluster_distance_epsilon_m: float = 10_000.0
    cluster_elevation_epsilon_per_km: float = 10.0
    cluster_min_samples: int = 2
    model_type: Literal["gradient_boosting", "random_forest", "ridge"] = "gradient_boosting"

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
