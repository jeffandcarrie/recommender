from collections import defaultdict
from datetime import date, datetime
from statistics import median

from strava_recommender.models.activity import EnrichedActivity
from strava_recommender.models.route import Route


def calculate_normalized_power(watts: list[float], window_seconds: int) -> float | None:
    """Approximate a rolling normalized-power value using a 1-second sample cadence."""
    values = [float(v) for v in watts if v is not None and v >= 0]
    if not values:
        return None

    if window_seconds <= 0:
        return None

    if len(values) <= window_seconds:
        window = values
    else:
        window = values[-window_seconds:]

    if not window:
        return None

    return round((sum(v ** 4 for v in window) / len(window)) ** 0.25, 2)


def _power_value(activity: EnrichedActivity) -> float | None:
    if activity.np_20m_watts is not None:
        return activity.np_20m_watts
    if activity.np_5m_watts is not None:
        return activity.np_5m_watts
    if activity.weighted_average_watts is not None:
        return activity.weighted_average_watts
    if activity.average_watts is not None:
        return activity.average_watts
    return activity.average_speed_mps * 3.6


def build_performance_gains(
    activities: list[EnrichedActivity],
    routes: list[Route] | None = None,
) -> list[dict]:
    """Rank rides by how much higher their power was than that route's historical norm."""
    if not activities:
        return []

    route_lookup = {route.route_id: route for route in (routes or [])}
    by_route: dict[int, list[EnrichedActivity]] = defaultdict(list)
    for activity in activities:
        if activity.route_id is not None:
            by_route[activity.route_id].append(activity)

    gains: list[dict] = []
    for route_id, route_activities in by_route.items():
        if len(route_activities) < 2:
            continue

        for activity in route_activities:
            current_power = _power_value(activity)
            if current_power is None:
                continue

            others = [a for a in route_activities if a.strava_id != activity.strava_id]
            other_powers = [p for p in (_power_value(a) for a in others) if p is not None]
            if not other_powers:
                continue

            baseline_power = median(other_powers)
            improvement_watts = current_power - baseline_power
            if improvement_watts <= 0:
                continue

            current_speed_kmh = activity.average_speed_mps * 3.6
            baseline_speed_kmh = median(a.average_speed_mps * 3.6 for a in others)
            improvement_kmh = current_speed_kmh - baseline_speed_kmh

            route = route_lookup.get(route_id)
            gains.append(
                {
                    "route_id": route_id,
                    "label": route.label if route else None,
                    "location_name": route.location_name if route else None,
                    "strava_id": activity.strava_id,
                    "start_time": activity.start_time.isoformat(),
                    "current_power_watts": current_power,
                    "baseline_power_watts": baseline_power,
                    "improvement_watts": improvement_watts,
                    "average_speed_kmh": current_speed_kmh,
                    "current_speed_kmh": current_speed_kmh,
                    "baseline_speed_kmh": baseline_speed_kmh,
                    "improvement_kmh": improvement_kmh,
                    "power_metric": "20m NP" if activity.np_20m_watts is not None else "5m NP" if activity.np_5m_watts is not None else "weighted avg watts" if activity.weighted_average_watts is not None else "speed fallback",
                    "distance_km": activity.distance_m / 1000,
                    "elevation_gain_m": activity.elevation_gain_m,
                    "weighted_average_watts": activity.weighted_average_watts,
                    "weather_condition": activity.weather_condition,
                    "temp_c": activity.temp_c,
                }
            )

    gains.sort(key=lambda row: row["improvement_watts"], reverse=True)
    return gains[:10]


def _week_key_for_activity(activity: EnrichedActivity) -> str:
    iso = activity.start_time.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _activity_power_value(activity: EnrichedActivity) -> float | None:
    for candidate in (
        activity.np_20m_watts,
        activity.np_5m_watts,
        activity.weighted_average_watts,
        activity.average_watts,
    ):
        if candidate is not None:
            return float(candidate)
    return None


def _hard_session_count(activities: list[EnrichedActivity]) -> int:
    count = 0
    for activity in activities:
        power = _activity_power_value(activity)
        if power is not None and power >= 250:
            count += 1
            continue
        if activity.average_watts is not None and activity.average_watts >= 220:
            count += 1
            continue
        if activity.average_speed_mps * 3.6 >= 38 and activity.distance_m >= 4000:
            count += 1
    return count


def _build_key_driver(row: dict) -> str:
    delta_hard = row["hard_session_count"] - row["previous_week_hard_sessions"]
    delta_distance = row["total_distance_km"] - row["previous_week_total_distance_km"]

    if delta_hard >= 2:
        return "Threshold block"
    if delta_hard > 0:
        return "More hard sessions"
    if delta_distance > 0:
        return "Volume bump"
    if row["hard_session_count"] > 0:
        return "Hard effort week"
    return "Steady week"


def _build_training_summary(row: dict) -> str:
    current_rides = row["ride_count"]
    current_hard = row["hard_session_count"]
    prev_rides = row["previous_week_ride_count"]
    prev_hard = row["previous_week_hard_sessions"]
    prev_distance = row["previous_week_total_distance_km"]
    current_distance = row["total_distance_km"]
    delta_rides = current_rides - prev_rides
    delta_hard = current_hard - prev_hard
    delta_distance = current_distance - prev_distance

    if prev_rides == 0:
        return (
            f"This looks like a strong build week: {current_rides} rides and {current_hard} hard sessions "
            f"across {current_distance:.0f} km."
        )

    ride_sentence = (
        f"This week had {current_rides} rides and {current_hard} hard sessions, "
        f"compared with {prev_rides} rides and {prev_hard} hard sessions last week"
    )
    change_sentence = []
    if delta_rides > 0:
        change_sentence.append(f"{delta_rides} more ride{'s' if delta_rides != 1 else ''}")
    elif delta_rides < 0:
        change_sentence.append(f"{abs(delta_rides)} fewer rides")
    if delta_hard > 0:
        change_sentence.append(f"{delta_hard} more hard session{'s' if delta_hard != 1 else ''}")
    elif delta_hard < 0:
        change_sentence.append(f"{abs(delta_hard)} fewer hard sessions")
    if delta_distance > 0:
        change_sentence.append(f"{delta_distance:.0f} km more")
    elif delta_distance < 0:
        change_sentence.append(f"{abs(delta_distance):.0f} km less")

    key_driver = _build_key_driver(row).lower()
    if change_sentence:
        return (
            f"{ride_sentence}; this looks like a {key_driver} week with "
            f"{', '.join(change_sentence)}."
        )
    return f"{ride_sentence}; the pattern stayed broadly similar to last week."


def build_weekly_performance_gains(
    activities: list[EnrichedActivity],
    routes: list[Route] | None = None,
) -> list[dict]:
    """Aggregate ride-level performance gains into the strongest weeks for a rider."""
    gains = build_performance_gains(activities, routes)
    if not gains:
        return []

    by_week: dict[str, list[dict]] = defaultdict(list)
    for gain in gains:
        start_time = datetime.fromisoformat(gain["start_time"])
        iso = start_time.isocalendar()
        week_key = f"{iso.year}-W{iso.week:02d}"
        by_week[week_key].append(gain)

    week_activity_totals: dict[str, list[EnrichedActivity]] = defaultdict(list)
    for activity in activities:
        week_activity_totals[_week_key_for_activity(activity)].append(activity)

    all_week_keys = sorted(set(by_week) | set(week_activity_totals))
    summaries: list[dict] = []
    for index, week_key in enumerate(all_week_keys):
        week_gains = by_week.get(week_key, [])
        week_activities = week_activity_totals.get(week_key, [])
        if not week_gains or not week_activities:
            continue

        iso_year, iso_week = week_key.split("-W")
        week_start = date.fromisocalendar(int(iso_year), int(iso_week), 1)

        route_names: list[str] = []
        seen: set[str] = set()
        for gain in week_gains:
            label = gain.get("label") or f"Route {gain['route_id']}"
            location = gain.get("location_name")
            route_name = label if not location else f"{label} · {location}"
            if route_name not in seen:
                route_names.append(route_name)
                seen.add(route_name)

        total_improvement = sum(float(gain["improvement_watts"]) for gain in week_gains)
        total_distance_km = sum(activity.distance_m / 1000 for activity in week_activities)
        total_elevation_m = sum(activity.elevation_gain_m for activity in week_activities)
        avg_power = sum(
            value for value in (
                _activity_power_value(activity) for activity in week_activities
            ) if value is not None
        ) / max(
            1,
            sum(1 for value in (_activity_power_value(activity) for activity in week_activities) if value is not None),
        )
        hard_sessions = _hard_session_count(week_activities)

        previous_week_key = None
        if index > 0:
            previous_week_key = all_week_keys[index - 1]

        prev = {
            "previous_week_label": previous_week_key,
            "previous_week_ride_count": 0,
            "previous_week_hard_sessions": 0,
            "previous_week_total_distance_km": 0.0,
            "previous_week_avg_power_watts": 0.0,
        }
        if previous_week_key and previous_week_key in week_activity_totals:
            prev_activities = week_activity_totals[previous_week_key]
            prev["previous_week_ride_count"] = len(prev_activities)
            prev["previous_week_hard_sessions"] = _hard_session_count(prev_activities)
            prev["previous_week_total_distance_km"] = round(sum(a.distance_m / 1000 for a in prev_activities), 1)
            power_values = [
                value for value in (_activity_power_value(a) for a in prev_activities) if value is not None
            ]
            if power_values:
                prev["previous_week_avg_power_watts"] = round(sum(power_values) / len(power_values), 1)

        row = {
            "week": week_key,
            "week_start": week_start.isoformat(),
            "route_names": route_names,
            "ride_count": len(week_activities),
            "total_improvement_watts": round(total_improvement, 2),
            "avg_improvement_watts": round(total_improvement / len(week_gains), 2),
            "peak_improvement_watts": round(
                max(float(gain["improvement_watts"]) for gain in week_gains), 2
            ),
            "total_distance_km": round(total_distance_km, 1),
            "total_elevation_m": round(total_elevation_m, 1),
            "avg_power_watts": round(avg_power, 1),
            "hard_session_count": hard_sessions,
            "previous_week_label": prev["previous_week_label"],
            "previous_week_ride_count": prev["previous_week_ride_count"],
            "previous_week_hard_sessions": prev["previous_week_hard_sessions"],
            "previous_week_total_distance_km": prev["previous_week_total_distance_km"],
            "previous_week_avg_power_watts": prev["previous_week_avg_power_watts"],
        }
        row["key_driver"] = _build_key_driver(row)
        row["training_summary"] = _build_training_summary(row)
        summaries.append(row)

    summaries.sort(key=lambda row: row["total_improvement_watts"], reverse=True)
    return summaries[:10]
