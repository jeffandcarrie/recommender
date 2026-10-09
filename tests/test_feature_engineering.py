import math

import pandas as pd

from strava_recommender.clustering import cluster_activities
from strava_recommender.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    build_prediction_features,
)
from strava_recommender.models.route import Route


class TestBuildFeatureMatrix:
    def test_produces_correct_columns(self, sample_activities):
        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)

        assert not df.empty
        for col in FEATURE_COLUMNS:
            assert col in df.columns
        assert "moving_speed_mps" in df.columns
        assert "average_watts" in df.columns

    def test_excludes_unrouted_activities(self, sample_activities):
        """Activities without a route_id should be excluded."""
        # Set all route_ids to None
        for a in sample_activities:
            a.route_id = None
        routes = []
        df = build_feature_matrix(sample_activities, routes)
        assert df.empty

    def test_headwind_speed(self, sample_activities):
        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)

        # headwind_speed should be wind_speed * -wind_alignment
        for _, row in df.iterrows():
            expected = row["wind_speed_mps"] * (-row["wind_alignment"])
            assert abs(row["headwind_speed"] - expected) < 1e-10

    def test_hour_encoding(self, sample_activities):
        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)

        sin_sq = df["hour_sin"] ** 2
        cos_sq = df["hour_cos"] ** 2
        sums = sin_sq + cos_sq
        assert all(abs(s - 1.0) < 1e-10 for s in sums)

    def test_empty_input(self):
        df = build_feature_matrix([], [])
        assert df.empty


    def test_bearing_encoding(self, sample_activities):
        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)

        # sin²+cos² should equal 1 for bearing direction
        sin_sq = df["bearing_deg_sin"] ** 2
        cos_sq = df["bearing_deg_cos"] ** 2
        sums = sin_sq + cos_sq
        assert all(abs(s - 1.0) < 1e-10 for s in sums)

    def test_wind_alignment_range(self, sample_activities):
        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)

        # Wind alignment should be between -1 and 1
        assert all(-1.0 <= v <= 1.0 for v in df["wind_alignment"])


class TestBuildPredictionFeatures:
    def test_returns_all_feature_columns(self, sample_weather):
        route = Route(
            route_id=0,
            centroid_lat=51.5,
            centroid_lon=-0.1,
            activity_count=10,
            avg_distance_m=30000.0,
            avg_elevation_gain_m=150.0,
            avg_bearing_deg=45.0,
        )
        features = build_prediction_features(route, sample_weather, hour=14.5)

        for col in FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"

    def test_headwind_speed_prediction(self, sample_weather):
        route = Route(
            route_id=0, centroid_lat=51.5, centroid_lon=-0.1,
            activity_count=10, avg_distance_m=30000.0, avg_elevation_gain_m=150.0,
            avg_bearing_deg=90.0,
        )
        features = build_prediction_features(route, sample_weather)
        expected = features["wind_speed_mps"] * (-features["wind_alignment"])
        assert abs(features["headwind_speed"] - expected) < 1e-10

    def test_bearing_encoding_unit_circle(self, sample_weather):
        route = Route(
            route_id=0, centroid_lat=51.5, centroid_lon=-0.1,
            activity_count=10, avg_distance_m=30000.0, avg_elevation_gain_m=150.0,
            avg_bearing_deg=135.0,
        )
        features = build_prediction_features(route, sample_weather)
        assert abs(features["bearing_deg_sin"] ** 2 + features["bearing_deg_cos"] ** 2 - 1.0) < 1e-10

    def test_hour_encoding_unit_circle(self, sample_weather):
        route = Route(
            route_id=0, centroid_lat=51.5, centroid_lon=-0.1,
            activity_count=10, avg_distance_m=30000.0, avg_elevation_gain_m=150.0,
            avg_bearing_deg=0.0,
        )
        features = build_prediction_features(route, sample_weather, hour=6.0)
        assert abs(features["hour_sin"] ** 2 + features["hour_cos"] ** 2 - 1.0) < 1e-10

    def test_wind_alignment_tailwind(self, sample_weather):
        """Wind from 220° on a 220° bearing route should be a tailwind."""
        route = Route(
            route_id=0, centroid_lat=51.5, centroid_lon=-0.1,
            activity_count=10, avg_distance_m=30000.0, avg_elevation_gain_m=150.0,
            avg_bearing_deg=220.0,  # Same as wind_deg in sample_weather
        )
        features = build_prediction_features(route, sample_weather)
        assert features["wind_alignment"] > 0.9  # Near-perfect tailwind

    def test_wind_alignment_headwind(self, sample_weather):
        """Wind from 220° on a 40° bearing route should be a headwind."""
        route = Route(
            route_id=0, centroid_lat=51.5, centroid_lon=-0.1,
            activity_count=10, avg_distance_m=30000.0, avg_elevation_gain_m=150.0,
            avg_bearing_deg=40.0,  # Opposite to wind_deg (220) in sample_weather
        )
        features = build_prediction_features(route, sample_weather)
        assert features["wind_alignment"] < -0.9  # Near-perfect headwind

    def test_no_target_columns(self, sample_weather):
        route = Route(
            route_id=0, centroid_lat=51.5, centroid_lon=-0.1,
            activity_count=10, avg_distance_m=30000.0, avg_elevation_gain_m=150.0,
            avg_bearing_deg=90.0,
        )
        features = build_prediction_features(route, sample_weather)
        assert "moving_speed_mps" not in features
        assert "average_speed_mps" not in features
        assert "average_watts" not in features
