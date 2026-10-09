from datetime import datetime

from pydantic import BaseModel


class EnrichedActivity(BaseModel):
    strava_id: int
    start_lat: float
    start_lon: float
    distance_m: float
    elevation_gain_m: float
    elevation_per_km: float = 0.0
    average_speed_mps: float
    average_watts: float | None = None
    weighted_average_watts: float | None = None
    start_time: datetime
    elapsed_seconds: int
    moving_seconds: int
    # Weather at activity start
    temp_c: float = 0.0
    humidity_pct: float = 0.0
    wind_speed_mps: float = 0.0
    wind_deg: int = 0
    pressure_hpa: float = 0.0
    cloud_pct: int = 0
    rain_1h_mm: float = 0.0
    weather_condition: str = ""
    temp_stress: float = 0.0
    # Bearing to farthest point from start (compass degrees, 0=N, 90=E)
    bearing_deg: float | None = None
    # Assigned after clustering
    route_id: int | None = None
