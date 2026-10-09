import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import joblib

from strava_recommender.config import get_settings

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from strava_recommender.models.activity import EnrichedActivity
    from strava_recommender.models.route import Route


def _data_dir() -> Path:
    return get_settings().ensure_data_dir()


# --- Activities ---

def save_activities(activities: list["EnrichedActivity"]) -> Path:
    path = _data_dir() / "activities.json"
    path.write_text(json.dumps([a.model_dump(mode="json") for a in activities], indent=2))
    return path


def load_activities() -> list["EnrichedActivity"]:
    from strava_recommender.models.activity import EnrichedActivity

    path = _data_dir() / "activities.json"
    if not path.exists():
        return []
    return [EnrichedActivity.model_validate(a) for a in json.loads(path.read_text())]


# --- Routes ---

def save_routes(routes: list["Route"]) -> Path:
    path = _data_dir() / "routes.json"
    path.write_text(json.dumps([r.model_dump(mode="json") for r in routes], indent=2))
    return path


def load_routes() -> list["Route"]:
    from strava_recommender.models.route import Route

    path = _data_dir() / "routes.json"
    if not path.exists():
        return []
    return [Route.model_validate(r) for r in json.loads(path.read_text())]


# --- ML Model ---

def save_model(pipeline: "Pipeline", name: str = "model_speed") -> Path:
    path = _data_dir() / f"{name}.joblib"
    joblib.dump(pipeline, path)
    return path


def load_model(name: str = "model_speed") -> "Pipeline":
    path = _data_dir() / f"{name}.joblib"
    return joblib.load(path)


# --- Weather Cache ---

def _weather_cache_path() -> Path:
    return _data_dir() / "weather_cache.json"


def load_weather_cache() -> dict[str, dict]:
    path = _weather_cache_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_weather_cache(cache: dict[str, dict]) -> None:
    _weather_cache_path().write_text(json.dumps(cache, indent=2))


# --- Tokens ---

def save_tokens(tokens: dict) -> Path:
    path = _data_dir() / "tokens.json"
    path.write_text(json.dumps(tokens, indent=2))
    return path


def load_tokens() -> Optional[dict]:
    path = _data_dir() / "tokens.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
