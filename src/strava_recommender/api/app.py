import math
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from strava_recommender.geo import bearing_to_compass
from strava_recommender.models.activity import EnrichedActivity
from strava_recommender.storage import load_activities, load_routes
from strava_recommender.weather_client import get_current_weather


Metric = Literal[
    "average_watts",
    "weighted_average_watts",
    "average_speed_kmh",
    "distance_km",
    "elevation_gain_m",
    "moving_minutes",
]


class ActivityOut(BaseModel):
    strava_id: int
    start_time: str
    distance_km: float
    elevation_gain_m: float
    average_speed_kmh: float
    average_watts: float | None
    weighted_average_watts: float | None
    moving_minutes: float
    temp_c: float
    wind_speed_mps: float
    wind_deg: int
    weather_condition: str
    bearing_deg: float | None
    headwind_mps: float | None
    route_id: int | None


class TimeseriesPoint(BaseModel):
    date: str
    value: float
    count: int


class HeadwindPoint(BaseModel):
    strava_id: int
    headwind_mps: float
    average_speed_kmh: float
    average_watts: float | None
    distance_km: float


class RouteWeather(BaseModel):
    temp_c: float
    wind_speed_mps: float
    wind_deg: int
    weather_condition: str
    wind_alignment: float | None
    direction: str | None


class RouteOut(BaseModel):
    route_id: int
    centroid_lat: float
    centroid_lon: float
    activity_count: int
    avg_distance_m: float
    avg_elevation_gain_m: float
    avg_elevation_per_km: float
    avg_bearing_deg: float | None
    label: str | None
    location_name: str | None
    weather: RouteWeather | None = None


def _headwind(activity: EnrichedActivity) -> float | None:
    if activity.bearing_deg is None or not activity.weather_condition:
        return None
    wind_rad = math.radians(activity.wind_deg)
    bearing_rad = math.radians(activity.bearing_deg)
    alignment = math.cos(wind_rad - bearing_rad)
    return activity.wind_speed_mps * (-alignment)


def _to_out(a: EnrichedActivity) -> ActivityOut:
    return ActivityOut(
        strava_id=a.strava_id,
        start_time=a.start_time.isoformat(),
        distance_km=a.distance_m / 1000,
        elevation_gain_m=a.elevation_gain_m,
        average_speed_kmh=a.average_speed_mps * 3.6,
        average_watts=a.average_watts,
        weighted_average_watts=a.weighted_average_watts,
        moving_minutes=a.moving_seconds / 60,
        temp_c=a.temp_c,
        wind_speed_mps=a.wind_speed_mps,
        wind_deg=a.wind_deg,
        weather_condition=a.weather_condition,
        bearing_deg=a.bearing_deg,
        headwind_mps=_headwind(a),
        route_id=a.route_id,
    )


def _metric_value(a: EnrichedActivity, metric: Metric) -> float | None:
    if metric == "average_watts":
        return a.average_watts
    if metric == "weighted_average_watts":
        return a.weighted_average_watts
    if metric == "average_speed_kmh":
        return a.average_speed_mps * 3.6
    if metric == "distance_km":
        return a.distance_m / 1000
    if metric == "elevation_gain_m":
        return a.elevation_gain_m
    if metric == "moving_minutes":
        return a.moving_seconds / 60
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="Strava Recommender")

    # CORS for vite dev server (5173)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/activities", response_model=list[ActivityOut])
    def get_activities() -> list[ActivityOut]:
        activities = load_activities()
        activities.sort(key=lambda a: a.start_time)
        return [_to_out(a) for a in activities]

    @app.get("/api/routes", response_model=list[RouteOut])
    def get_routes(with_weather: bool = False) -> list[RouteOut]:
        routes = load_routes()
        if not with_weather:
            return [RouteOut(**r.model_dump()) for r in routes]

        # Dedupe weather lookups by rounded centroid; the client cache also handles repeats.
        weather_by_key: dict[str, dict] = {}
        out: list[RouteOut] = []
        for r in routes:
            key = f"{round(r.centroid_lat, 2)},{round(r.centroid_lon, 2)}"
            if key not in weather_by_key:
                try:
                    weather_by_key[key] = get_current_weather(r.centroid_lat, r.centroid_lon)
                except Exception:
                    weather_by_key[key] = {}
            w = weather_by_key[key]
            rw: RouteWeather | None = None
            if w:
                wind_alignment: float | None = None
                direction: str | None = None
                if r.avg_bearing_deg is not None:
                    wind_rad = math.radians(w.get("wind_deg", 0))
                    bearing_rad = math.radians(r.avg_bearing_deg)
                    wind_alignment = math.cos(wind_rad - bearing_rad)
                    direction = bearing_to_compass(r.avg_bearing_deg)
                rw = RouteWeather(
                    temp_c=w.get("temp_c", 0.0),
                    wind_speed_mps=w.get("wind_speed_mps", 0.0),
                    wind_deg=int(w.get("wind_deg", 0)),
                    weather_condition=w.get("weather_condition", ""),
                    wind_alignment=wind_alignment,
                    direction=direction,
                )
            out.append(RouteOut(**r.model_dump(), weather=rw))
        return out

    @app.get("/api/stats/timeseries", response_model=list[TimeseriesPoint])
    def get_timeseries(
        metric: Metric = Query("average_watts"),
        bucket: Literal["month", "week"] = Query("month"),
    ) -> list[TimeseriesPoint]:
        activities = load_activities()
        buckets: dict[str, list[float]] = {}
        for a in activities:
            value = _metric_value(a, metric)
            if value is None:
                continue
            if bucket == "month":
                key = a.start_time.strftime("%Y-%m")
            else:
                iso = a.start_time.isocalendar()
                key = f"{iso.year}-W{iso.week:02d}"
            buckets.setdefault(key, []).append(value)

        return [
            TimeseriesPoint(date=k, value=sum(v) / len(v), count=len(v))
            for k, v in sorted(buckets.items())
        ]

    @app.get("/api/stats/headwind-scatter", response_model=list[HeadwindPoint])
    def get_headwind_scatter() -> list[HeadwindPoint]:
        points: list[HeadwindPoint] = []
        for a in load_activities():
            hw = _headwind(a)
            if hw is None:
                continue
            points.append(
                HeadwindPoint(
                    strava_id=a.strava_id,
                    headwind_mps=hw,
                    average_speed_kmh=a.average_speed_mps * 3.6,
                    average_watts=a.average_watts,
                    distance_km=a.distance_m / 1000,
                )
            )
        return points

    @app.get("/api/recommend")
    def get_recommend(
        top_n: int = 3,
        time_available_minutes: float | None = None,
    ) -> list[dict]:
        try:
            from strava_recommender.recommender import recommend
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        results = recommend(top_n=top_n, time_available_minutes=time_available_minutes)
        out = []
        for r in results:
            route = r["route"]
            activities = r.get("activities", [])
            sample_activities = sorted(
                activities, key=lambda a: a.start_time, reverse=True
            )[:5]
            out.append(
                {
                    "route_id": route.route_id,
                    "label": route.label,
                    "location_name": route.location_name,
                    "avg_distance_km": route.avg_distance_m / 1000,
                    "avg_elevation_gain_m": route.avg_elevation_gain_m,
                    "predicted_speed_kmh": r["predicted_speed_kmh"],
                    "estimated_time_minutes": r["estimated_time_minutes"],
                    "direction": r.get("direction"),
                    "wind_alignment": r.get("wind_alignment"),
                    "weather": r.get("weather"),
                    "sample_activities": [
                        {
                            "strava_id": a.strava_id,
                            "start_time": a.start_time.isoformat(),
                            "distance_km": a.distance_m / 1000,
                            "average_speed_kmh": a.average_speed_mps * 3.6,
                        }
                        for a in sample_activities
                    ],
                }
            )
        return out

    # Serve the built React bundle if it exists. Vite builds to web/dist.
    web_dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    if web_dist.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=web_dist / "assets"),
            name="assets",
        )

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(web_dist / "index.html")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str) -> FileResponse:
            # Anything not matching /api/* falls back to index.html for SPA routing
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404)
            target = web_dist / full_path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(web_dist / "index.html")

    return app


app = create_app()
