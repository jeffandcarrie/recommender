from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from strava_recommender.config import get_settings
from strava_recommender.feature_engineering import FEATURE_COLUMNS, TARGET_SPEED
from strava_recommender.storage import load_model, save_model

def _build_pipeline(model_type: str, n_samples: int = 100) -> Pipeline:
    """Build an sklearn pipeline with preprocessing and a regressor."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLUMNS),
        ]
    )

    if model_type == "random_forest":
        regressor = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    elif model_type == "ridge":
        regressor = Ridge(alpha=1.0)
    else:
        # Scale regularization to dataset size
        min_leaf = max(2, n_samples // 30)
        regressor = GradientBoostingRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.65,
            min_samples_leaf=min_leaf,
            max_features=0.6,
            random_state=42,
        )

    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def train(df: pd.DataFrame, target: str = TARGET_SPEED) -> dict[str, Any]:
    """Train the model and return metrics + the fitted pipeline.

    Returns dict with keys: pipeline, r2, mae, rmse, cv_scores.
    """
    settings = get_settings()

    X = df[FEATURE_COLUMNS]
    y = df[target]

    pipeline = _build_pipeline(settings.model_type, n_samples=len(df))

    # Cross-validation
    n_splits = min(5, len(df))
    if n_splits < 2:
        # Not enough data for CV, just fit
        pipeline.fit(X, y)
        return {
            "pipeline": pipeline,
            "r2": None,
            "mae": None,
            "rmse": None,
            "cv_scores": [],
            "sample_count": len(df),
        }

    cv_r2 = cross_val_score(pipeline, X, y, cv=n_splits, scoring="r2")
    cv_mae = cross_val_score(pipeline, X, y, cv=n_splits, scoring="neg_mean_absolute_error")

    # Fit on full data
    pipeline.fit(X, y)

    # Train R² (how well the model fits the training data)
    train_r2 = float(pipeline.score(X, y))

    # Feature importances from the fitted regressor
    regressor = pipeline.named_steps["regressor"]
    importances: dict[str, float] = {}
    if hasattr(regressor, "feature_importances_"):
        for name, imp in zip(FEATURE_COLUMNS, regressor.feature_importances_):
            importances[name] = float(imp)
    elif hasattr(regressor, "coef_"):
        # Linear models: use absolute coefficient magnitude as importance
        abs_coefs = np.abs(regressor.coef_)
        total = abs_coefs.sum() if abs_coefs.sum() > 0 else 1.0
        for name, coef in zip(FEATURE_COLUMNS, abs_coefs / total):
            importances[name] = float(coef)

    model_name = "model_power" if target != TARGET_SPEED else "model_speed"
    save_model(pipeline, model_name)

    return {
        "pipeline": pipeline,
        "r2": float(np.mean(cv_r2)),
        "train_r2": train_r2,
        "mae": float(-np.mean(cv_mae)),
        "rmse": float(np.sqrt(np.mean(cv_mae**2))),
        "cv_scores": cv_r2.tolist(),
        "sample_count": len(df),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "feature_importances": importances,
    }


def predict(features: pd.DataFrame, model_name: str = "model_speed") -> np.ndarray:
    """Load a saved model and predict on new features."""
    pipeline = load_model(model_name)
    return pipeline.predict(features[FEATURE_COLUMNS])
