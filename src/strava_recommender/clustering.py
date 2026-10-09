import math

import numpy as np
from sklearn.cluster import DBSCAN

from strava_recommender.config import get_settings
from strava_recommender.geo import angular_distance
from strava_recommender.models.activity import EnrichedActivity
from strava_recommender.models.route import Route

EARTH_RADIUS_M = 6_371_000

# Weights for combined distance metric (must sum to 1.0)
BEARING_WEIGHT = 0.55
DISTANCE_WEIGHT = 0.10
LOCATION_WEIGHT = 0.20
ELEVATION_WEIGHT = 0.15

# Default angular epsilon for bearing clustering
BEARING_EPSILON_DEG = 10.0

# Distance buckets (metres)
DISTANCE_BUCKETS = [15_000, 25_000, 35_000, 45_000, 60_000]


def _distance_bucket(distance_m: float) -> int:
    """Assign a ride to a distance bucket."""
    for i, threshold in enumerate(DISTANCE_BUCKETS):
        if distance_m < threshold:
            return i
    return len(DISTANCE_BUCKETS)


def _combined_distance(
    activities: list[EnrichedActivity],
    eps_meters: float,
    distance_eps_m: float,
    elevation_eps_per_km: float,
    bearing_eps_deg: float = BEARING_EPSILON_DEG,
) -> np.ndarray:
    """Build a precomputed distance matrix combining four dimensions.

    Components (all normalised to [0, 1] before weighting):
    - Bearing similarity (40%) — angular distance between route directions
    - Distance similarity (25%) — absolute difference in ride distance
    - Location proximity (20%) — haversine distance between start points
    - Elevation similarity (15%) — absolute difference in elevation per km

    The DBSCAN epsilon is set to 1.0 on the combined metric.
    """
    n = len(activities)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = activities[i], activities[j]

            # Bearing component
            if ai.bearing_deg is not None and aj.bearing_deg is not None:
                ang = angular_distance(ai.bearing_deg, aj.bearing_deg)
                bearing_norm = ang / bearing_eps_deg
            else:
                bearing_norm = 2.0

            # Location component (haversine)
            lat1, lon1 = math.radians(ai.start_lat), math.radians(ai.start_lon)
            lat2, lon2 = math.radians(aj.start_lat), math.radians(aj.start_lon)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            haversine_m = EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            location_norm = haversine_m / eps_meters

            # Distance component — bucket-based: 0 if same bucket, 1 if different
            distance_norm = 0.0 if _distance_bucket(ai.distance_m) == _distance_bucket(aj.distance_m) else 1.0

            # Elevation component
            elevation_norm = abs(ai.elevation_per_km - aj.elevation_per_km) / elevation_eps_per_km

            combined = (
                BEARING_WEIGHT * bearing_norm
                + LOCATION_WEIGHT * location_norm
                + DISTANCE_WEIGHT * distance_norm
                + ELEVATION_WEIGHT * elevation_norm
            )
            dist[i, j] = combined
            dist[j, i] = combined

    return dist


def _mean_bearing(bearings: list[float]) -> float:
    """Compute the circular mean of a list of bearings in degrees."""
    sin_sum = sum(math.sin(math.radians(b)) for b in bearings)
    cos_sum = sum(math.cos(math.radians(b)) for b in bearings)
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360) % 360


def cluster_activities(
    activities: list[EnrichedActivity],
) -> tuple[list[EnrichedActivity], list[Route]]:
    """Cluster activities by direction, distance, elevation, and start location.

    Uses a combined distance metric where:
    - Bearing similarity is the primary dimension (40%)
    - Distance similarity is secondary (25%)
    - Start location proximity is tertiary (20%)
    - Elevation profile similarity is quaternary (15%)

    Activities without a bearing fall back to location-only clustering.

    Returns updated activities (with route_id assigned) and a list of Route objects.
    """
    if not activities:
        return [], []

    settings = get_settings()

    # Split into activities with and without bearing data
    with_bearing = [a for a in activities if a.bearing_deg is not None]
    without_bearing = [a for a in activities if a.bearing_deg is None]

    labels_map: dict[int, int | None] = {}  # index in original list -> route_id
    next_route_id = 0

    # --- Cluster activities that have bearing data ---
    if len(with_bearing) >= settings.cluster_min_samples:
        dist_matrix = _combined_distance(
            with_bearing,
            eps_meters=settings.cluster_epsilon_meters,
            distance_eps_m=settings.cluster_distance_epsilon_m,
            elevation_eps_per_km=settings.cluster_elevation_epsilon_per_km,
        )
        db = DBSCAN(
            eps=0.3,  # tight threshold to prevent chain-linking
            min_samples=settings.cluster_min_samples,
            metric="precomputed",
        )
        bearing_labels = db.fit_predict(dist_matrix)

        for act, label in zip(with_bearing, bearing_labels):
            idx = activities.index(act)
            if label >= 0:
                labels_map[idx] = int(label)
            else:
                labels_map[idx] = None

        if len(bearing_labels) > 0 and bearing_labels.max() >= 0:
            next_route_id = int(bearing_labels.max()) + 1

    else:
        for act in with_bearing:
            labels_map[activities.index(act)] = None

    # --- Fallback: cluster activities without bearing by location only ---
    if len(without_bearing) >= settings.cluster_min_samples:
        coords = np.array([[a.start_lat, a.start_lon] for a in without_bearing])
        coords_rad = np.radians(coords)
        eps_rad = settings.cluster_epsilon_meters / EARTH_RADIUS_M
        db = DBSCAN(
            eps=eps_rad,
            min_samples=settings.cluster_min_samples,
            metric="haversine",
        )
        loc_labels = db.fit_predict(coords_rad)

        for act, label in zip(without_bearing, loc_labels):
            idx = activities.index(act)
            if label >= 0:
                labels_map[idx] = int(next_route_id + label)
            else:
                labels_map[idx] = None
    else:
        for act in without_bearing:
            labels_map[activities.index(act)] = None

    # --- Build updated activities ---
    updated_activities = []
    for i, activity in enumerate(activities):
        activity_copy = activity.model_copy()
        activity_copy.route_id = labels_map.get(i)
        updated_activities.append(activity_copy)

    # --- Build route summaries ---
    route_ids = {rid for rid in labels_map.values() if rid is not None}
    routes: list[Route] = []
    for route_id in sorted(route_ids):
        cluster_acts = [a for a in updated_activities if a.route_id == route_id]
        lats = [a.start_lat for a in cluster_acts]
        lons = [a.start_lon for a in cluster_acts]
        bearings = [a.bearing_deg for a in cluster_acts if a.bearing_deg is not None]

        routes.append(
            Route(
                route_id=route_id,
                centroid_lat=float(np.mean(lats)),
                centroid_lon=float(np.mean(lons)),
                activity_count=len(cluster_acts),
                avg_distance_m=float(np.mean([a.distance_m for a in cluster_acts])),
                avg_elevation_gain_m=float(np.mean([a.elevation_gain_m for a in cluster_acts])),
                avg_elevation_per_km=float(np.mean([a.elevation_per_km for a in cluster_acts])),
                avg_bearing_deg=_mean_bearing(bearings) if bearings else None,
            )
        )

    return updated_activities, routes
