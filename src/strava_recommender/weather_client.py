import time
from datetime import datetime, timezone
from typing import Dict, Optional

import httpx

from strava_recommender.storage import load_weather_cache, save_weather_cache

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# In-process cache for current weather, keyed by rounded (lat, lon).
# Open-Meteo updates roughly every 15 min; 10 min keeps us responsive without hammering.
_CURRENT_TTL_SECONDS = 600
_current_cache: Dict[str, tuple[float, dict]] = {}

# WMO weather code to simple condition string
_WMO_CONDITIONS: Dict[int, str] = {
    0: "Clear",
    1: "Clear",
    2: "Clouds",
    3: "Clouds",
    45: "Fog",
    48: "Fog",
    51: "Drizzle",
    53: "Drizzle",
    55: "Drizzle",
    56: "Drizzle",
    57: "Drizzle",
    61: "Rain",
    63: "Rain",
    65: "Rain",
    66: "Rain",
    67: "Rain",
    71: "Snow",
    73: "Snow",
    75: "Snow",
    77: "Snow",
    80: "Rain",
    81: "Rain",
    82: "Rain",
    85: "Snow",
    86: "Snow",
    95: "Thunderstorm",
    96: "Thunderstorm",
    99: "Thunderstorm",
}


def _cache_key(lat: float, lon: float, date_str: str) -> str:
    """Cache key using rounded coords + date."""
    return f"{round(lat, 2)},{round(lon, 2)},{date_str}"


def _wmo_to_condition(code: Optional[int]) -> str:
    """Convert a WMO weather code to a simple condition string."""
    if code is None:
        return ""
    return _WMO_CONDITIONS.get(code, "")


def _parse_current(data: dict) -> dict:
    """Extract weather fields from an Open-Meteo current weather response."""
    current = data.get("current", {})
    return {
        "temp_c": current.get("temperature_2m", 0.0),
        "humidity_pct": current.get("relative_humidity_2m", 0.0),
        "wind_speed_mps": round(current.get("wind_speed_10m", 0.0) / 3.6, 2),
        "wind_deg": current.get("wind_direction_10m", 0),
        "pressure_hpa": current.get("surface_pressure", 0.0),
        "cloud_pct": current.get("cloud_cover", 0),
        "rain_1h_mm": current.get("rain", 0.0),
        "weather_condition": _wmo_to_condition(current.get("weather_code")),
    }


def _parse_hourly(data: dict, target_hour: int) -> dict:
    """Extract weather fields from the closest hour in an Open-Meteo archive response."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    # Find the index closest to the target hour
    idx = min(range(len(times)), key=lambda i: abs(i - target_hour)) if times else 0

    def _get(field: str, default=0.0):
        values = hourly.get(field, [])
        return values[idx] if idx < len(values) and values[idx] is not None else default

    return {
        "temp_c": _get("temperature_2m"),
        "humidity_pct": _get("relative_humidity_2m"),
        "wind_speed_mps": round(_get("wind_speed_10m") / 3.6, 2),
        "wind_deg": int(_get("wind_direction_10m", 0)),
        "pressure_hpa": _get("surface_pressure"),
        "cloud_pct": int(_get("cloud_cover", 0)),
        "rain_1h_mm": _get("rain"),
        "weather_condition": _wmo_to_condition(int(_get("weather_code", -1))),
    }


def get_historical_weather(lat: float, lon: float, unix_timestamp: int) -> dict:
    """Fetch historical weather for a given location and time. Uses cache."""
    dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")
    key = _cache_key(lat, lon, date_str)

    cache = load_weather_cache()
    if key in cache:
        return cache[key]

    response = httpx.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,cloud_cover,rain,weather_code",
        },
        timeout=30,
    )
    response.raise_for_status()

    weather = _parse_hourly(response.json(), dt.hour)

    cache[key] = weather
    save_weather_cache(cache)

    # Rate limit to be polite to the free API
    time.sleep(1.1)
    return weather


def get_current_weather(lat: float, lon: float) -> dict:
    """Fetch current weather for a given location, cached in-process for ~10 min."""
    key = f"{round(lat, 2)},{round(lon, 2)}"
    now = time.monotonic()
    cached = _current_cache.get(key)
    if cached and now - cached[0] < _CURRENT_TTL_SECONDS:
        return cached[1]

    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,cloud_cover,rain,weather_code",
        },
        timeout=30,
    )
    response.raise_for_status()

    weather = _parse_current(response.json())
    _current_cache[key] = (now, weather)
    return weather
