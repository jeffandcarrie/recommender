from datetime import datetime, timezone

from strava_recommender.analysis import (
    build_performance_gains,
    build_weekly_performance_gains,
    calculate_normalized_power,
)
from strava_recommender.models.activity import EnrichedActivity


class TestPerformanceGains:
    def test_calculate_normalized_power(self):
        np_value = calculate_normalized_power([100] * 300, 5 * 60)
        assert np_value == 100.0

    def test_build_performance_gains_ranks_improvement(self):
        activities = [
            EnrichedActivity(
                strava_id=1,
                start_lat=42.0,
                start_lon=-71.0,
                distance_m=5000,
                elevation_gain_m=100,
                average_speed_mps=5.0,
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                elapsed_seconds=1800,
                moving_seconds=1800,
                route_id=10,
            ),
            EnrichedActivity(
                strava_id=2,
                start_lat=42.0,
                start_lon=-71.0,
                distance_m=5000,
                elevation_gain_m=100,
                average_speed_mps=6.0,
                start_time=datetime(2024, 1, 8, tzinfo=timezone.utc),
                elapsed_seconds=1800,
                moving_seconds=1800,
                route_id=10,
            ),
            EnrichedActivity(
                strava_id=3,
                start_lat=42.0,
                start_lon=-71.0,
                distance_m=5000,
                elevation_gain_m=100,
                average_speed_mps=7.0,
                start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
                elapsed_seconds=1800,
                moving_seconds=1800,
                route_id=10,
            ),
        ]

        gains = build_performance_gains(activities)

        assert len(gains) == 1
        assert gains[0]["route_id"] == 10
        assert gains[0]["strava_id"] == 3
        assert gains[0]["improvement_kmh"] > 0
        assert gains[0]["baseline_speed_kmh"] < gains[0]["average_speed_kmh"]

    def test_build_weekly_performance_gains_aggregates_by_week(self):
        activities = [
            EnrichedActivity(
                strava_id=1,
                start_lat=42.0,
                start_lon=-71.0,
                distance_m=5000,
                elevation_gain_m=100,
                average_speed_mps=5.0,
                average_watts=200,
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                elapsed_seconds=1800,
                moving_seconds=1800,
                route_id=10,
            ),
            EnrichedActivity(
                strava_id=2,
                start_lat=42.0,
                start_lon=-71.0,
                distance_m=5000,
                elevation_gain_m=100,
                average_speed_mps=6.0,
                average_watts=220,
                start_time=datetime(2024, 1, 8, tzinfo=timezone.utc),
                elapsed_seconds=1800,
                moving_seconds=1800,
                route_id=10,
            ),
            EnrichedActivity(
                strava_id=3,
                start_lat=42.0,
                start_lon=-71.0,
                distance_m=5000,
                elevation_gain_m=100,
                average_speed_mps=7.0,
                average_watts=260,
                start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
                elapsed_seconds=1800,
                moving_seconds=1800,
                route_id=10,
            ),
            EnrichedActivity(
                strava_id=4,
                start_lat=42.0,
                start_lon=-71.0,
                distance_m=5000,
                elevation_gain_m=100,
                average_speed_mps=6.5,
                average_watts=240,
                start_time=datetime(2024, 1, 22, tzinfo=timezone.utc),
                elapsed_seconds=1800,
                moving_seconds=1800,
                route_id=11,
            ),
        ]

        weeks = build_weekly_performance_gains(activities)

        assert weeks
        assert weeks[0]["route_names"]
        assert weeks[0]["total_improvement_watts"] > 0
        assert weeks[0]["ride_count"] >= 1
        assert "previous_week_hard_sessions" in weeks[0]
        assert weeks[0]["previous_week_hard_sessions"] >= 0
        assert "training_summary" in weeks[0]
        assert "key_driver" in weeks[0]
        assert "hard sessions" in weeks[0]["training_summary"].lower()
        assert "last week" in weeks[0]["training_summary"].lower()
        assert len(weeks) <= 10
