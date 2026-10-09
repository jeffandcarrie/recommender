from datetime import datetime, timezone

from strava_recommender.models.activity import EnrichedActivity
from strava_recommender.models.route import Route
from strava_recommender.storage import (
    load_activities,
    load_routes,
    load_tokens,
    load_weather_cache,
    save_activities,
    save_routes,
    save_tokens,
    save_weather_cache,
)


def _make_activity(strava_id: int = 1) -> EnrichedActivity:
    return EnrichedActivity(
        strava_id=strava_id,
        start_lat=51.5,
        start_lon=-0.1,
        distance_m=25000.0,
        elevation_gain_m=100.0,
        average_speed_mps=7.0,
        start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        elapsed_seconds=3600,
        moving_seconds=3500,
    )


class TestActivityStorage:
    def test_save_and_load(self, tmp_data_dir):
        activities = [_make_activity(1), _make_activity(2)]
        save_activities(activities)
        loaded = load_activities()
        assert len(loaded) == 2
        assert loaded[0].strava_id == 1
        assert loaded[1].strava_id == 2

    def test_load_empty(self, tmp_data_dir):
        assert load_activities() == []


class TestRouteStorage:
    def test_save_and_load(self, tmp_data_dir):
        routes = [
            Route(route_id=0, centroid_lat=51.5, centroid_lon=-0.1,
                  activity_count=10, avg_distance_m=30000, avg_elevation_gain_m=150),
        ]
        save_routes(routes)
        loaded = load_routes()
        assert len(loaded) == 1
        assert loaded[0].route_id == 0

    def test_load_empty(self, tmp_data_dir):
        assert load_routes() == []


class TestTokenStorage:
    def test_save_and_load(self, tmp_data_dir):
        tokens = {"access_token": "abc", "refresh_token": "def", "expires_at": 123456}
        save_tokens(tokens)
        loaded = load_tokens()
        assert loaded["access_token"] == "abc"

    def test_load_missing(self, tmp_data_dir):
        assert load_tokens() is None


class TestWeatherCache:
    def test_save_and_load(self, tmp_data_dir):
        cache = {"51.5,-0.1,2024-06-15": {"temp_c": 20.0}}
        save_weather_cache(cache)
        loaded = load_weather_cache()
        assert loaded["51.5,-0.1,2024-06-15"]["temp_c"] == 20.0

    def test_load_empty(self, tmp_data_dir):
        assert load_weather_cache() == {}
