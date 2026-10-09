import math
from datetime import datetime, timezone

import pandas as pd
from rich.console import Console

from strava_recommender.feature_engineering import FEATURE_COLUMNS, build_prediction_features
from strava_recommender.geo import bearing_to_compass
from strava_recommender.models.route import Route
from strava_recommender.storage import load_activities, load_model, load_routes
from strava_recommender.weather_client import get_current_weather

console = Console()


def recommend(top_n: int = 3, time_available_minutes: float | None = None) -> list[dict]:
    """Recommend the best routes based on current weather conditions.

    Args:
        top_n: Maximum number of routes to return.
        time_available_minutes: If set, only include routes completable within this time.

    Returns a list of dicts with route info, predicted speed, weather, and estimated time.
    """
    routes = load_routes()
    if not routes:
        console.print("[red]No routes found. Run 'sync' and 'cluster' first.[/red]")
        return []

    pipeline = load_model("model_speed")

    # Build a lookup of route_id -> activities for matching
    all_activities = load_activities()
    activities_by_route: dict[int, list] = {}
    for a in all_activities:
        if a.route_id is not None:
            activities_by_route.setdefault(a.route_id, []).append(a)

    # Compute rider's typical WAP from historical data
    wap_values = [a.weighted_average_watts for a in all_activities if a.weighted_average_watts]
    median_wap = float(sorted(wap_values)[len(wap_values) // 2]) if wap_values else 200.0
    console.print(f"Using median WAP: [bold]{median_wap:.0f}W[/bold]")

    now = datetime.now(tz=timezone.utc)
    hour = now.hour + now.minute / 60.0
    day_of_week = now.weekday()
    month = now.month

    results: list[dict] = []

    # Fetch weather for each route centroid
    # Group nearby routes to reduce API calls
    weather_cache: dict[str, dict] = {}

    for route in routes:
        cache_key = f"{round(route.centroid_lat, 2)},{round(route.centroid_lon, 2)}"
        if cache_key not in weather_cache:
            weather_cache[cache_key] = get_current_weather(
                route.centroid_lat, route.centroid_lon
            )
            weather = weather_cache[cache_key]
            console.print(f"Predicting for (weather: {weather['weather_condition']}, {weather['temp_c']}°C)")
        weather = weather_cache[cache_key]

        features = build_prediction_features(
            route=route,
            weather=weather,
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            weighted_average_watts=median_wap,
        )

        features_df = pd.DataFrame([features])[FEATURE_COLUMNS]
        predicted_speed = float(pipeline.predict(features_df)[0])

        # Convert m/s to km/h for display
        predicted_kmh = predicted_speed * 3.6

        # Compute wind alignment for display
        wind_alignment = None
        direction = None
        if route.avg_bearing_deg is not None:
            wind_rad = math.radians(weather.get("wind_deg", 0))
            bearing_rad = math.radians(route.avg_bearing_deg)
            wind_alignment = math.cos(wind_rad - bearing_rad)
            direction = bearing_to_compass(route.avg_bearing_deg)

        route_activities = activities_by_route.get(route.route_id, [])

        # Model predicts moving speed; estimate total time with ~95% moving ratio
        moving_ratio = 0.95
        estimated_time_min = (
            (route.avg_distance_m / predicted_speed / moving_ratio / 60.0)
            if predicted_speed > 0
            else float("inf")
        )

        results.append(
            {
                "route": route,
                "predicted_speed_mps": predicted_speed,
                "predicted_speed_kmh": predicted_kmh,
                "estimated_time_minutes": estimated_time_min,
                "weather": weather,
                "direction": direction,
                "wind_alignment": wind_alignment,
                "location": route.location_name or "Unknown",
                "activities": route_activities,
            }
        )

    # Filter by time budget if specified — pick routes that fill the available time
    # Allow routes between 70% and 100% of the time budget
    if time_available_minutes is not None:
        min_time = time_available_minutes * 0.7
        max_time = time_available_minutes
        results = [r for r in results if min_time <= r["estimated_time_minutes"] <= max_time]
        if not results:
            console.print(
                f"[yellow]No routes match a {time_available_minutes:.0f}-minute window "
                f"({min_time:.0f}–{max_time:.0f} min). Try adjusting your time budget.[/yellow]"
            )
            return []

    # Sort by predicted speed descending
    results.sort(key=lambda r: r["predicted_speed_mps"], reverse=True)
    return results[:top_n]


def hindsight(top_n: int = 10) -> list[dict]:
    """Analyse past rides and find the worst route choices.

    For each activity, predicts speed on every route using that day's actual
    weather, then compares the chosen route to the best alternative. Returns
    the rides with the biggest speed gap (i.e. worst choices), sorted by gap.
    """
    routes = load_routes()
    if not routes:
        console.print("[red]No routes found. Run 'cluster' first.[/red]")
        return []

    pipeline = load_model("model_speed")
    all_activities = load_activities()

    # Only analyse activities that have route, weather, and power data
    candidates = [
        a for a in all_activities
        if a.route_id is not None and a.weather_condition and a.weighted_average_watts
    ]
    if not candidates:
        console.print("[red]No activities with route + weather + power data.[/red]")
        return []

    route_lookup = {r.route_id: r for r in routes}
    results = []

    for activity in candidates:
        if activity.route_id not in route_lookup:
            continue

        # Build weather dict from the activity's stored weather
        weather = {
            "temp_c": activity.temp_c,
            "humidity_pct": activity.humidity_pct,
            "wind_speed_mps": activity.wind_speed_mps,
            "wind_deg": activity.wind_deg,
            "pressure_hpa": activity.pressure_hpa,
            "cloud_pct": activity.cloud_pct,
            "rain_1h_mm": activity.rain_1h_mm,
            "weather_condition": activity.weather_condition,
        }
        hour = activity.start_time.hour + activity.start_time.minute / 60.0
        day_of_week = activity.start_time.weekday()
        month = activity.start_time.month
        wap = activity.weighted_average_watts

        # Predict speed on every route for this day's conditions
        route_speeds: dict[int, float] = {}
        for route in routes:
            features = build_prediction_features(
                route=route,
                weather=weather,
                hour=hour,
                day_of_week=day_of_week,
                month=month,
                weighted_average_watts=wap,
            )
            features_df = pd.DataFrame([features])[FEATURE_COLUMNS]
            route_speeds[route.route_id] = float(pipeline.predict(features_df)[0])

        chosen_route = route_lookup[activity.route_id]
        chosen_speed = route_speeds[activity.route_id]
        best_route_id = max(route_speeds, key=route_speeds.get)
        best_speed = route_speeds[best_route_id]
        best_route = route_lookup[best_route_id]

        gap_kmh = (best_speed - chosen_speed) * 3.6

        # Only include if there was a meaningfully better option
        if gap_kmh > 0.5:
            results.append({
                "activity": activity,
                "chosen_route": chosen_route,
                "best_route": best_route,
                "chosen_speed_kmh": chosen_speed * 3.6,
                "best_speed_kmh": best_speed * 3.6,
                "actual_speed_kmh": activity.average_speed_mps * 3.6,
                "gap_kmh": gap_kmh,
                "weather": weather,
            })

    results.sort(key=lambda r: r["gap_kmh"], reverse=True)
    return results[:top_n]
