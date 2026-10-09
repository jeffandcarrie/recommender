import math

import numpy as np
import pandas as pd

from strava_recommender.geo import temp_stress as _temp_stress
from strava_recommender.models.activity import EnrichedActivity
from strava_recommender.models.route import Route


def _circular_encode(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode a circular variable as sin/cos components."""
    radians = 2 * math.pi * values / period
    return np.sin(radians), np.cos(radians)


def _elevation_per_km(elevation_m: float, distance_m: float) -> float:
    """Elevation gain per kilometre."""
    if distance_m <= 0:
        return 0.0
    return elevation_m / (distance_m / 1000)


def build_feature_matrix(
    activities: list[EnrichedActivity],
    routes: list[Route],
) -> pd.DataFrame:
    """Build a feature DataFrame from enriched activities that have route_id assigned.

    Only includes activities with a valid route_id (not None / noise).
    """
    # Filter to activities with assigned routes
    routed = [a for a in activities if a.route_id is not None]
    if not routed:
        return pd.DataFrame()

    route_lookup = {r.route_id: r for r in routes}

    records = []
    for a in routed:
        route = route_lookup[a.route_id]
        hour = a.start_time.hour + a.start_time.minute / 60.0
        wind_rad = math.radians(a.wind_deg)
        hour_rad = 2 * math.pi * hour / 24.0

        bearing_rad = math.radians(a.bearing_deg) if a.bearing_deg is not None else 0.0
        wind_alignment = math.cos(wind_rad - bearing_rad) if a.bearing_deg is not None else 0.0

        t_stress = a.temp_stress
        elev_km = a.elevation_per_km

        # Effective headwind component (positive = headwind, negative = tailwind)
        headwind_speed = a.wind_speed_mps * (-wind_alignment) if a.bearing_deg is not None else 0.0

        # Moving ratio: fraction of time actually pedalling vs stopped
        moving_ratio = a.moving_seconds / a.elapsed_seconds if a.elapsed_seconds > 0 else 1.0

        # Moving speed is the cleaner target — distance / moving_time
        moving_speed_mps = a.distance_m / a.moving_seconds if a.moving_seconds > 0 else a.average_speed_mps

        records.append(
            {
                "distance_m": a.distance_m,
                "elevation_gain_m": a.elevation_gain_m,
                "elevation_per_km": elev_km,
                "weighted_average_watts": a.weighted_average_watts,
                "bearing_deg_sin": math.sin(bearing_rad),
                "bearing_deg_cos": math.cos(bearing_rad),
                "temp_c": a.temp_c,
                "temp_stress": t_stress,
                "elevation_temp_stress": elev_km * t_stress,
                "humidity_pct": a.humidity_pct,
                "wind_speed_mps": a.wind_speed_mps,
                "wind_alignment": wind_alignment,
                "headwind_speed": headwind_speed,
                "rain_1h_mm": a.rain_1h_mm,
                "is_rainy": 1 if a.rain_1h_mm > 0 or a.weather_condition == "Rain" else 0,
                "hour_sin": math.sin(hour_rad),
                "hour_cos": math.cos(hour_rad),
                "day_of_week": a.start_time.weekday(),
                "month": a.start_time.month,
                "moving_ratio": moving_ratio,
                # Targets
                "average_speed_mps": a.average_speed_mps,
                "moving_speed_mps": moving_speed_mps,
                "average_watts": a.average_watts,
            }
        )

    return pd.DataFrame(records)


def build_prediction_features(
    route: Route,
    weather: dict,
    hour: float = 12.0,
    day_of_week: int = 0,
    month: int = 6,
    weighted_average_watts: float = 200.0,
) -> dict:
    """Build a single feature dict for prediction (no target).

    Args:
        weighted_average_watts: Expected effort level (WAP). Defaults to 200W;
            the recommender overrides this with the rider's median WAP.
    """
    wind_rad = math.radians(weather.get("wind_deg", 0))
    hour_rad = 2 * math.pi * hour / 24.0
    bearing_rad = math.radians(route.avg_bearing_deg) if route.avg_bearing_deg is not None else 0.0
    wind_alignment = math.cos(wind_rad - bearing_rad) if route.avg_bearing_deg is not None else 0.0
    wind_speed = weather.get("wind_speed_mps", 0.0)
    headwind_speed = wind_speed * (-wind_alignment) if route.avg_bearing_deg is not None else 0.0

    temp_c = weather.get("temp_c", 15.0)
    t_stress = _temp_stress(temp_c)
    elev_km = _elevation_per_km(route.avg_elevation_gain_m, route.avg_distance_m)

    return {
        "distance_m": route.avg_distance_m,
        "elevation_gain_m": route.avg_elevation_gain_m,
        "elevation_per_km": elev_km,
        "weighted_average_watts": weighted_average_watts,
        "bearing_deg_sin": math.sin(bearing_rad),
        "bearing_deg_cos": math.cos(bearing_rad),
        "temp_c": temp_c,
        "temp_stress": t_stress,
        "elevation_temp_stress": elev_km * t_stress,
        "humidity_pct": weather.get("humidity_pct", 50.0),
        "wind_speed_mps": wind_speed,
        "wind_alignment": wind_alignment,
        "headwind_speed": headwind_speed,
        "rain_1h_mm": weather.get("rain_1h_mm", 0.0),
        "is_rainy": 1 if weather.get("rain_1h_mm", 0) > 0 or weather.get("weather_condition") == "Rain" else 0,
        "hour_sin": math.sin(hour_rad),
        "hour_cos": math.cos(hour_rad),
        "day_of_week": day_of_week,
        "month": month,
        "moving_ratio": 0.95,
    }


FEATURE_COLUMNS = [
    "distance_m",
    "elevation_gain_m",
    "elevation_per_km",
    "weighted_average_watts",
    "bearing_deg_sin",
    "bearing_deg_cos",
    "temp_c",
    "temp_stress",
    "elevation_temp_stress",
    "humidity_pct",
    "wind_speed_mps",
    "wind_alignment",
    "headwind_speed",
    "rain_1h_mm",
    "is_rainy",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "month",
    "moving_ratio",
]

TARGET_SPEED = "moving_speed_mps"
TARGET_POWER = "average_watts"
