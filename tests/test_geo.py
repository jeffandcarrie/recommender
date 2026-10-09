import math

from strava_recommender.geo import (
    angular_distance,
    bearing_degrees,
    bearing_to_compass,
    classify_difficulty,
    classify_elevation,
    classify_temp,
    decode_polyline,
    farthest_point_bearing,
    haversine_distance,
    temp_stress,
)


class TestDecodePolyline:
    def test_simple_polyline(self):
        # Encoded polyline for a simple path
        # This is the Google example: (38.5, -120.2), (40.7, -120.95), (43.252, -126.453)
        encoded = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
        points = decode_polyline(encoded)
        assert len(points) == 3
        assert abs(points[0][0] - 38.5) < 0.001
        assert abs(points[0][1] - (-120.2)) < 0.001

    def test_empty_polyline(self):
        assert decode_polyline("") == []


class TestHaversineDistance:
    def test_same_point(self):
        assert haversine_distance(51.5, -0.1, 51.5, -0.1) == 0.0

    def test_known_distance(self):
        # London to Paris is approximately 344 km
        dist = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        assert 340_000 < dist < 350_000


class TestBearingDegrees:
    def test_due_north(self):
        bearing = bearing_degrees(51.5, -0.1, 52.5, -0.1)
        assert abs(bearing - 0.0) < 1.0

    def test_due_east(self):
        bearing = bearing_degrees(0.0, 0.0, 0.0, 1.0)
        assert abs(bearing - 90.0) < 1.0

    def test_due_south(self):
        bearing = bearing_degrees(52.5, -0.1, 51.5, -0.1)
        assert abs(bearing - 180.0) < 1.0


class TestFarthestPointBearing:
    def test_farthest_point_north(self):
        # Route goes north then comes back
        points = [(51.5, -0.1), (51.6, -0.1), (51.7, -0.1), (51.5, -0.1)]
        bearing = farthest_point_bearing(51.5, -0.1, points)
        assert bearing is not None
        assert abs(bearing - 0.0) < 5.0  # ~north

    def test_farthest_point_east(self):
        points = [(51.5, -0.1), (51.5, 0.0), (51.5, 0.1), (51.5, -0.1)]
        bearing = farthest_point_bearing(51.5, -0.1, points)
        assert bearing is not None
        assert abs(bearing - 90.0) < 5.0  # ~east

    def test_empty_points(self):
        assert farthest_point_bearing(51.5, -0.1, []) is None

    def test_too_close_returns_none(self):
        # All points within 50m of start
        points = [(51.5, -0.1), (51.50001, -0.10001)]
        assert farthest_point_bearing(51.5, -0.1, points) is None


class TestAngularDistance:
    def test_same_direction(self):
        assert angular_distance(45.0, 45.0) == 0.0

    def test_opposite(self):
        assert angular_distance(0.0, 180.0) == 180.0

    def test_wraps_around(self):
        assert abs(angular_distance(350.0, 10.0) - 20.0) < 1e-10

    def test_symmetric(self):
        assert angular_distance(30.0, 90.0) == angular_distance(90.0, 30.0)


class TestBearingToCompass:
    def test_cardinal_directions(self):
        assert bearing_to_compass(0.0) == "N"
        assert bearing_to_compass(90.0) == "E"
        assert bearing_to_compass(180.0) == "S"
        assert bearing_to_compass(270.0) == "W"

    def test_intercardinal_directions(self):
        assert bearing_to_compass(45.0) == "NE"
        assert bearing_to_compass(135.0) == "SE"
        assert bearing_to_compass(225.0) == "SW"
        assert bearing_to_compass(315.0) == "NW"

    def test_near_boundary(self):
        # 22° is closer to N than NE
        assert bearing_to_compass(22.0) == "N"
        # 23° is closer to NE than N
        assert bearing_to_compass(23.0) == "NE"


class TestClassifyElevation:
    def test_flat(self):
        # 100m gain over 40km = 2.5 m/km
        assert classify_elevation(100, 40_000) == "Flat"

    def test_rolling(self):
        # 280m gain over 40km = 7 m/km
        assert classify_elevation(280, 40_000) == "Rolling"

    def test_hilly(self):
        # ~1000ft / 20mi = 305m / 32km = 9.5 m/km
        assert classify_elevation(305, 32_000) == "Hilly"

    def test_very_hilly(self):
        # 600m gain over 30km = 20 m/km
        assert classify_elevation(600, 30_000) == "Very Hilly"

    def test_zero_distance(self):
        assert classify_elevation(100, 0) == "Flat"


class TestTempStress:
    def test_ideal_temp(self):
        assert temp_stress(18.0) == 0.0

    def test_cold(self):
        assert temp_stress(8.0) == 1.0

    def test_hot(self):
        assert temp_stress(28.0) == 1.0

    def test_symmetric(self):
        # Equidistant from ideal should give same stress
        assert temp_stress(13.0) == temp_stress(23.0)


class TestClassifyTemp:
    def test_ideal(self):
        assert classify_temp(18.0) == "Ideal"
        assert classify_temp(17.0) == "Ideal"

    def test_cool(self):
        assert classify_temp(12.0) == "Cool"

    def test_warm(self):
        assert classify_temp(24.0) == "Warm"

    def test_cold(self):
        assert classify_temp(7.0) == "Cold"

    def test_hot(self):
        assert classify_temp(30.0) == "Hot"

    def test_very_cold(self):
        assert classify_temp(-2.0) == "Very Cold"

    def test_very_hot(self):
        assert classify_temp(35.0) == "Very Hot"


class TestClassifyDifficulty:
    def test_flat_any_temp(self):
        # Flat routes stay flat regardless of temp
        assert classify_difficulty(100, 40_000, 5.0) == "Flat"

    def test_hilly_ideal_temp(self):
        # Hilly at ideal temp is just hilly
        assert classify_difficulty(305, 32_000, 18.0) == "Hilly"

    def test_hilly_cold_warns(self):
        # Hilly + cold should suggest flat route
        result = classify_difficulty(305, 32_000, 5.0)
        assert "Hilly" in result
        assert "consider flat" in result

    def test_hilly_hot_warns(self):
        # Hilly + hot should suggest flat route
        result = classify_difficulty(305, 32_000, 32.0)
        assert "Hilly" in result
        assert "consider flat" in result

    def test_rolling_cold_no_warn(self):
        # Rolling terrain shouldn't trigger the warning
        assert classify_difficulty(280, 40_000, 5.0) == "Rolling"
