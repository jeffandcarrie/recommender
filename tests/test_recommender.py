from unittest.mock import patch

import pandas as pd

from strava_recommender.clustering import cluster_activities
from strava_recommender.feature_engineering import build_feature_matrix
from strava_recommender.predictor import train as train_model
from strava_recommender.recommender import recommend
from strava_recommender.storage import save_routes


class TestRecommender:
    def _setup_model(self, sample_activities, tmp_data_dir):
        """Helper to train a model and save routes for recommendation tests."""
        updated, routes = cluster_activities(sample_activities)
        save_routes(routes)

        from strava_recommender.storage import save_activities
        save_activities(updated)

        df = build_feature_matrix(updated, routes)
        train_model(df)
        return routes

    def test_recommend_returns_ranked_routes(self, sample_activities, tmp_data_dir, sample_weather):
        routes = self._setup_model(sample_activities, tmp_data_dir)

        with patch("strava_recommender.recommender.get_current_weather", return_value=sample_weather):
            results = recommend(top_n=3)

        assert len(results) <= 3
        assert len(results) >= 1  # Should find at least some routes

        # Should be sorted descending by predicted speed
        speeds = [r["predicted_speed_mps"] for r in results]
        assert speeds == sorted(speeds, reverse=True)

    def test_recommend_includes_weather(self, sample_activities, tmp_data_dir, sample_weather):
        self._setup_model(sample_activities, tmp_data_dir)

        with patch("strava_recommender.recommender.get_current_weather", return_value=sample_weather):
            results = recommend(top_n=1)

        assert "weather" in results[0]
        assert results[0]["weather"]["temp_c"] == 18.0

    def test_recommend_includes_speed_in_kmh(self, sample_activities, tmp_data_dir, sample_weather):
        self._setup_model(sample_activities, tmp_data_dir)

        with patch("strava_recommender.recommender.get_current_weather", return_value=sample_weather):
            results = recommend(top_n=1)

        result = results[0]
        expected_kmh = result["predicted_speed_mps"] * 3.6
        assert abs(result["predicted_speed_kmh"] - expected_kmh) < 0.01

    def test_recommend_includes_matching_activities(self, sample_activities, tmp_data_dir, sample_weather):
        """Each result should include the activities that belong to that route."""
        self._setup_model(sample_activities, tmp_data_dir)

        with patch("strava_recommender.recommender.get_current_weather", return_value=sample_weather):
            results = recommend(top_n=3)

        for result in results:
            assert "activities" in result
            activities = result["activities"]
            route_id = result["route"].route_id
            assert len(activities) > 0, f"Route {route_id} should have matching activities"
            # Every activity should belong to this route
            for act in activities:
                assert act.route_id == route_id
            # Each activity should have a strava_id we can build a link from
            for act in activities:
                assert act.strava_id is not None

    def test_recommend_includes_direction_and_wind(self, sample_activities, tmp_data_dir, sample_weather):
        """Results should include direction and wind alignment for routes with bearings."""
        self._setup_model(sample_activities, tmp_data_dir)

        with patch("strava_recommender.recommender.get_current_weather", return_value=sample_weather):
            results = recommend(top_n=3)

        for result in results:
            route = result["route"]
            if route.avg_bearing_deg is not None:
                assert result["direction"] is not None
                assert result["wind_alignment"] is not None
                assert -1.0 <= result["wind_alignment"] <= 1.0

    def test_recommend_includes_estimated_time(self, sample_activities, tmp_data_dir, sample_weather):
        """Each result should include a positive estimated time."""
        self._setup_model(sample_activities, tmp_data_dir)

        with patch("strava_recommender.recommender.get_current_weather", return_value=sample_weather):
            results = recommend(top_n=3)

        for result in results:
            assert "estimated_time_minutes" in result
            assert result["estimated_time_minutes"] > 0

    def test_recommend_time_available_filters(self, sample_activities, tmp_data_dir, sample_weather):
        """Setting time_available_minutes should exclude routes that take too long."""
        self._setup_model(sample_activities, tmp_data_dir)

        with patch("strava_recommender.recommender.get_current_weather", return_value=sample_weather):
            all_results = recommend(top_n=10)

        with patch("strava_recommender.recommender.get_current_weather", return_value=sample_weather):
            # Use a very short time budget that should exclude at least some routes
            filtered = recommend(top_n=10, time_available_minutes=1)

        assert len(filtered) <= len(all_results)
        for r in filtered:
            assert r["estimated_time_minutes"] <= 1

    def test_recommend_empty_routes(self, tmp_data_dir):
        """Should return empty list when no routes exist."""
        results = recommend(top_n=3)
        assert results == []
