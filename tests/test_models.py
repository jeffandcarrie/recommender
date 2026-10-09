from datetime import datetime, timezone

from strava_recommender.models.activity import EnrichedActivity
from strava_recommender.models.route import Route


class TestEnrichedActivity:
    def test_create_minimal(self):
        activity = EnrichedActivity(
            strava_id=1,
            start_lat=51.5,
            start_lon=-0.1,
            distance_m=25000.0,
            elevation_gain_m=100.0,
            average_speed_mps=7.5,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            elapsed_seconds=3600,
            moving_seconds=3500,
        )
        assert activity.strava_id == 1
        assert activity.average_watts is None
        assert activity.route_id is None
        assert activity.weather_condition == ""

    def test_create_with_weather(self):
        activity = EnrichedActivity(
            strava_id=2,
            start_lat=51.5,
            start_lon=-0.1,
            distance_m=30000.0,
            elevation_gain_m=200.0,
            average_speed_mps=6.5,
            average_watts=185.0,
            start_time=datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
            elapsed_seconds=4000,
            moving_seconds=3800,
            temp_c=22.0,
            wind_speed_mps=5.0,
            wind_deg=180,
            weather_condition="Clear",
            route_id=0,
        )
        assert activity.average_watts == 185.0
        assert activity.temp_c == 22.0
        assert activity.route_id == 0

    def test_json_round_trip(self):
        activity = EnrichedActivity(
            strava_id=3,
            start_lat=51.5,
            start_lon=-0.1,
            distance_m=25000.0,
            elevation_gain_m=100.0,
            average_speed_mps=7.0,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            elapsed_seconds=3600,
            moving_seconds=3500,
        )
        data = activity.model_dump(mode="json")
        restored = EnrichedActivity.model_validate(data)
        assert restored.strava_id == activity.strava_id
        assert restored.start_lat == activity.start_lat


class TestRoute:
    def test_create(self):
        route = Route(
            route_id=0,
            centroid_lat=51.5,
            centroid_lon=-0.1,
            activity_count=10,
            avg_distance_m=30000.0,
            avg_elevation_gain_m=150.0,
        )
        assert route.route_id == 0
        assert route.label is None

    def test_create_with_label(self):
        route = Route(
            route_id=1,
            centroid_lat=51.6,
            centroid_lon=-0.2,
            activity_count=5,
            avg_distance_m=40000.0,
            avg_elevation_gain_m=300.0,
            label="Richmond Park Loop",
        )
        assert route.label == "Richmond Park Loop"
