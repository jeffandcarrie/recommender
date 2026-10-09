import os
import tempfile
from datetime import datetime, timezone

import pytest

# Set env vars before any app imports
os.environ.setdefault("STRAVA_CLIENT_ID", "12345")
os.environ.setdefault("STRAVA_CLIENT_SECRET", "test_secret")


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    """Use a temp directory for all data storage during tests."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Clear the lru_cache so settings pick up the new env
    from strava_recommender.config import get_settings
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture()
def sample_activities():
    """Generate a list of sample EnrichedActivity objects around 2 clusters."""
    import random
    from strava_recommender.models.activity import EnrichedActivity

    random.seed(42)
    activities = []

    # Cluster 1: starts near (51.5, -0.1), bearing ~45° (NE)
    # Cluster 2: starts near (51.6, -0.2), bearing ~225° (SW)
    for i in range(40):
        if i < 25:
            lat = 51.5 + random.gauss(0, 0.001)
            lon = -0.1 + random.gauss(0, 0.001)
            base_speed = 7.0
            bearing = 45.0 + random.gauss(0, 5)  # NE direction
        else:
            lat = 51.6 + random.gauss(0, 0.001)
            lon = -0.2 + random.gauss(0, 0.001)
            base_speed = 5.5
            bearing = 225.0 + random.gauss(0, 5)  # SW direction

        temp = random.uniform(5, 25)
        wind = random.uniform(0, 10)
        wap = random.uniform(180, 280)
        # Higher WAP → faster; temp and wind also affect speed
        speed = base_speed + 0.05 * temp - 0.1 * wind + 0.005 * wap + random.gauss(0, 0.3)

        distance = random.uniform(20000, 60000)
        elevation = random.uniform(50, 500)

        # Derive times from distance and speed so moving_speed target has signal
        moving_secs = int(distance / speed)
        elapsed_secs = int(moving_secs / 0.95)  # ~95% moving ratio

        activities.append(
            EnrichedActivity(
                strava_id=i,
                start_lat=lat,
                start_lon=lon,
                distance_m=distance,
                elevation_gain_m=elevation,
                elevation_per_km=elevation / (distance / 1000),
                average_speed_mps=distance / elapsed_secs,
                average_watts=random.uniform(150, 250) if i % 3 == 0 else None,
                weighted_average_watts=wap,
                start_time=datetime(2024, 6, 15, 10 + i % 8, 0, tzinfo=timezone.utc),
                elapsed_seconds=elapsed_secs,
                moving_seconds=moving_secs,
                temp_c=temp,
                temp_stress=abs(temp - 18.0) / 10.0,
                humidity_pct=random.uniform(30, 90),
                wind_speed_mps=wind,
                wind_deg=random.randint(0, 359),
                pressure_hpa=random.uniform(1000, 1030),
                cloud_pct=random.randint(0, 100),
                rain_1h_mm=random.uniform(0, 2) if i % 5 == 0 else 0.0,
                weather_condition="Rain" if i % 5 == 0 else "Clear",
                bearing_deg=bearing,
            )
        )

    return activities


@pytest.fixture()
def sample_weather():
    """Sample weather data dict."""
    return {
        "temp_c": 18.0,
        "humidity_pct": 55.0,
        "wind_speed_mps": 3.5,
        "wind_deg": 220,
        "pressure_hpa": 1015.0,
        "cloud_pct": 30,
        "rain_1h_mm": 0.0,
        "weather_condition": "Clear",
    }
