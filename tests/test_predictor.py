import pandas as pd

from strava_recommender.clustering import cluster_activities
from strava_recommender.feature_engineering import FEATURE_COLUMNS, build_feature_matrix
from strava_recommender.predictor import train as train_model


class TestPredictor:
    def test_train_returns_metrics(self, sample_activities):
        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)

        result = train_model(df)

        assert "pipeline" in result
        assert "r2" in result
        assert "mae" in result
        assert "sample_count" in result
        assert result["sample_count"] == len(df)

    def test_train_r2_is_reasonable(self, sample_activities):
        """R² should not be wildly negative on synthetic data with signal."""
        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)

        result = train_model(df)
        # With our synthetic data that has a clear signal, R² should be > -1
        assert result["r2"] is not None
        assert result["r2"] > -1.0

    def test_predict_returns_array(self, sample_activities):
        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)

        result = train_model(df)
        pipeline = result["pipeline"]

        predictions = pipeline.predict(df[FEATURE_COLUMNS])
        assert len(predictions) == len(df)
        assert all(p > 0 for p in predictions)  # Speed should be positive

    def test_prediction_varies_with_conditions(self, sample_activities):
        """Different weather/effort inputs should produce different predictions."""
        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)
        result = train_model(df)
        pipeline = result["pipeline"]

        # Take a single row and vary multiple conditions
        row = df[FEATURE_COLUMNS].iloc[[0]].copy()

        row_easy = row.copy()
        row_easy["wind_speed_mps"] = 0.0
        row_easy["headwind_speed"] = 0.0
        row_easy["weighted_average_watts"] = 250.0

        row_hard = row.copy()
        row_hard["wind_speed_mps"] = 15.0
        row_hard["headwind_speed"] = 15.0
        row_hard["weighted_average_watts"] = 150.0

        pred_easy = pipeline.predict(row_easy)[0]
        pred_hard = pipeline.predict(row_hard)[0]

        # They should differ
        assert pred_easy != pred_hard

    def test_model_save_and_load(self, sample_activities, tmp_data_dir):
        from strava_recommender.storage import load_model

        updated, routes = cluster_activities(sample_activities)
        df = build_feature_matrix(updated, routes)
        train_model(df)

        loaded = load_model("model_speed")
        predictions = loaded.predict(df[FEATURE_COLUMNS])
        assert len(predictions) == len(df)
