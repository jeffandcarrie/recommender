from pydantic import BaseModel


class Route(BaseModel):
    route_id: int
    centroid_lat: float
    centroid_lon: float
    activity_count: int
    avg_distance_m: float
    avg_elevation_gain_m: float
    avg_elevation_per_km: float = 0.0
    avg_bearing_deg: float | None = None
    label: str | None = None
    location_name: str | None = None
