"""Geographic utilities: polyline decoding, bearing, distance, and geocoding."""

import math

import httpx

EARTH_RADIUS_M = 6_371_000


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline into a list of (lat, lon) tuples."""
    points: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lon = 0
    length = len(encoded)

    while index < length:
        # Decode latitude
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lat += (~(result >> 1) if result & 1 else result >> 1)

        # Decode longitude
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lon += (~(result >> 1) if result & 1 else result >> 1)

        points.append((lat / 1e5, lon / 1e5))

    return points


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance in meters between two points."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the initial bearing (compass degrees, 0=N, 90=E) from point 1 to point 2."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def farthest_point_bearing(
    start_lat: float, start_lon: float, polyline_points: list[tuple[float, float]]
) -> float | None:
    """Find the bearing from start to the farthest point along a route.

    Returns compass degrees (0=N, 90=E), or None if no valid points.
    """
    if not polyline_points:
        return None

    max_dist = 0.0
    farthest = None

    for lat, lon in polyline_points:
        dist = haversine_distance(start_lat, start_lon, lat, lon)
        if dist > max_dist:
            max_dist = dist
            farthest = (lat, lon)

    if farthest is None or max_dist < 50:  # Less than 50m means essentially same point
        return None

    return bearing_degrees(start_lat, start_lon, farthest[0], farthest[1])


def angular_distance(deg1: float, deg2: float) -> float:
    """Calculate the smallest angular distance between two compass bearings (0-180)."""
    diff = abs(deg1 - deg2) % 360
    return diff if diff <= 180 else 360 - diff


def bearing_to_compass(deg: float) -> str:
    """Convert a bearing in degrees to a compass direction label."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(deg / 45) % 8
    return directions[index]


IDEAL_TEMP_C = 18.0  # Comfortable cycling temperature


def temp_stress(temp_c: float) -> float:
    """How far temperature is from ideal riding conditions.

    Returns 0 at ideal temp, increases for both cold and hot.
    Normalised so that 10°C away from ideal = 1.0.
    """
    return abs(temp_c - IDEAL_TEMP_C) / 10.0


def classify_temp(temp_c: float) -> str:
    """Classify temperature conditions for cycling.

    Returns one of: Ideal, Mild, Cold, Hot, Extreme.
    """
    stress = temp_stress(temp_c)
    if stress < 0.3:
        return "Ideal"
    if stress < 0.8:
        if temp_c < IDEAL_TEMP_C:
            return "Cool"
        return "Warm"
    if stress < 1.3:
        if temp_c < IDEAL_TEMP_C:
            return "Cold"
        return "Hot"
    if temp_c < IDEAL_TEMP_C:
        return "Very Cold"
    return "Very Hot"


def classify_elevation(elevation_m: float, distance_m: float) -> str:
    """Classify a route's hilliness based on elevation gain per distance.

    Reference: ~1000 ft / 20 mi (~9.5 m/km) is considered hilly.

    Returns one of: Flat, Rolling, Hilly, Very Hilly.
    """
    if distance_m <= 0:
        return "Flat"
    gain_per_km = elevation_m / (distance_m / 1000)
    if gain_per_km < 5:
        return "Flat"
    if gain_per_km < 9:
        return "Rolling"
    if gain_per_km < 16:
        return "Hilly"
    return "Very Hilly"


def classify_difficulty(elevation_m: float, distance_m: float, temp_c: float) -> str:
    """Overall route difficulty considering terrain and temperature stress.

    Combines elevation classification with temperature to flag when
    hilly routes become harder due to extreme temperatures.
    """
    terrain = classify_elevation(elevation_m, distance_m)
    temp_label = classify_temp(temp_c)

    # Extreme temps make hilly routes significantly harder
    is_hilly = terrain in ("Hilly", "Very Hilly")
    is_extreme_temp = temp_label in ("Cold", "Hot", "Very Cold", "Very Hot")

    if is_hilly and is_extreme_temp:
        return f"{terrain} ({temp_label} — consider flat route)"
    return terrain


def reverse_geocode(lat: float, lon: float) -> str:
    """Return the nearest city/town/village name for a coordinate using Nominatim.

    Queries at full detail (zoom=18) so the response includes all address
    levels, then walks from the most-local place name up to broader admin
    areas until it finds something useful.  Returns "Unknown" only on
    network/parsing failure.
    """
    try:
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 18,
                "addressdetails": 1,
            },
            headers={"User-Agent": "strava-recommender/1.0"},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        address = data.get("address", {})
        # Walk from most-local to broadest admin level
        for key in (
            "village",
            "town",
            "city",
            "hamlet",
            "suburb",
            "municipality",
            "city_district",
            "borough",
            "county",
            "state_district",
            "state",
        ):
            if key in address:
                return address[key]
        # Last resort: first segment of the display name
        display = data.get("display_name", "")
        if display:
            return display.split(",")[0].strip()
        return "Unknown"
    except Exception:
        return "Unknown"


def geocode_routes(routes: list) -> None:
    """Resolve location_name for routes that don't have one yet.

    Mutates routes in-place.  Deduplicates nearby centroids and respects
    Nominatim's 1 req/s rate limit.
    """
    import time as _time

    cache: dict[str, str] = {}
    for route in routes:
        if route.location_name:
            continue
        cache_key = f"{round(route.centroid_lat, 2)},{round(route.centroid_lon, 2)}"
        if cache_key not in cache:
            if cache:  # not the first request — respect rate limit
                _time.sleep(1.1)
            cache[cache_key] = reverse_geocode(route.centroid_lat, route.centroid_lon)
        route.location_name = cache[cache_key]
