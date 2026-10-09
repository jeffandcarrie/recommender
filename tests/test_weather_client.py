import httpx
import respx

from strava_recommender.weather_client import (
    _cache_key,
    _parse_current,
    _parse_hourly,
    _wmo_to_condition,
    get_current_weather,
    get_historical_weather,
)


class TestCacheKey:
    def test_rounds_coordinates(self):
        key = _cache_key(51.5074, -0.1278, "2024-06-15")
        assert key == "51.51,-0.13,2024-06-15"

    def test_same_area_same_key(self):
        key1 = _cache_key(51.5050, -0.1260, "2024-06-15")
        key2 = _cache_key(51.5090, -0.1290, "2024-06-15")
        assert key1 == key2

    def test_different_dates_different_keys(self):
        key1 = _cache_key(51.5, -0.1, "2024-06-15")
        key2 = _cache_key(51.5, -0.1, "2024-06-16")
        assert key1 != key2


class TestWmoToCondition:
    def test_clear(self):
        assert _wmo_to_condition(0) == "Clear"
        assert _wmo_to_condition(1) == "Clear"

    def test_rain(self):
        assert _wmo_to_condition(61) == "Rain"
        assert _wmo_to_condition(80) == "Rain"

    def test_clouds(self):
        assert _wmo_to_condition(2) == "Clouds"
        assert _wmo_to_condition(3) == "Clouds"

    def test_none(self):
        assert _wmo_to_condition(None) == ""

    def test_unknown_code(self):
        assert _wmo_to_condition(999) == ""


class TestParseCurrent:
    def test_parse_current(self):
        data = {
            "current": {
                "temperature_2m": 18.5,
                "relative_humidity_2m": 65,
                "wind_speed_10m": 15.12,  # km/h
                "wind_direction_10m": 220,
                "surface_pressure": 1015,
                "cloud_cover": 40,
                "rain": 0.5,
                "weather_code": 61,
            }
        }
        result = _parse_current(data)
        assert result["temp_c"] == 18.5
        assert result["humidity_pct"] == 65
        assert result["wind_speed_mps"] == round(15.12 / 3.6, 2)
        assert result["wind_deg"] == 220
        assert result["pressure_hpa"] == 1015
        assert result["cloud_pct"] == 40
        assert result["rain_1h_mm"] == 0.5
        assert result["weather_condition"] == "Rain"

    def test_parse_missing_fields(self):
        data = {"current": {}}
        result = _parse_current(data)
        assert result["temp_c"] == 0.0
        assert result["weather_condition"] == ""


class TestParseHourly:
    def test_parse_target_hour(self):
        data = {
            "hourly": {
                "time": [f"2024-06-15T{h:02d}:00" for h in range(24)],
                "temperature_2m": [15.0 + h * 0.5 for h in range(24)],
                "relative_humidity_2m": [50] * 24,
                "wind_speed_10m": [7.2] * 24,  # km/h
                "wind_direction_10m": [180] * 24,
                "surface_pressure": [1013] * 24,
                "cloud_cover": [10] * 24,
                "rain": [0.0] * 24,
                "weather_code": [0] * 24,
            }
        }
        result = _parse_hourly(data, target_hour=12)
        assert result["temp_c"] == 21.0  # 15.0 + 12 * 0.5
        assert result["wind_speed_mps"] == round(7.2 / 3.6, 2)
        assert result["weather_condition"] == "Clear"

    def test_parse_empty_hourly(self):
        data = {"hourly": {"time": []}}
        result = _parse_hourly(data, target_hour=0)
        assert result["temp_c"] == 0.0
        assert result["weather_condition"] == ""


@respx.mock
class TestGetHistoricalWeather:
    def test_fetches_and_caches(self, tmp_data_dir):
        respx.get("https://archive-api.open-meteo.com/v1/archive").mock(
            return_value=httpx.Response(
                200,
                json={
                    "hourly": {
                        "time": [f"2024-06-15T{h:02d}:00" for h in range(24)],
                        "temperature_2m": [20.0] * 24,
                        "relative_humidity_2m": [60] * 24,
                        "wind_speed_10m": [10.8] * 24,
                        "wind_direction_10m": [90] * 24,
                        "surface_pressure": [1010] * 24,
                        "cloud_cover": [20] * 24,
                        "rain": [0.0] * 24,
                        "weather_code": [3] * 24,
                    }
                },
            )
        )

        # Unix timestamp for 2024-06-15 12:00 UTC
        ts = 1718452800
        result = get_historical_weather(51.5, -0.1, ts)
        assert result["temp_c"] == 20.0
        assert result["weather_condition"] == "Clouds"

        # Second call should use cache (no HTTP call)
        respx.get("https://archive-api.open-meteo.com/v1/archive").mock(
            side_effect=RuntimeError("Should not be called")
        )
        cached = get_historical_weather(51.5, -0.1, ts)
        assert cached["temp_c"] == 20.0


@respx.mock
class TestGetCurrentWeather:
    def test_fetches_current(self, tmp_data_dir):
        respx.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 15.0,
                        "relative_humidity_2m": 70,
                        "wind_speed_10m": 18.0,
                        "wind_direction_10m": 270,
                        "surface_pressure": 1020,
                        "cloud_cover": 80,
                        "rain": 1.2,
                        "weather_code": 61,
                    }
                },
            )
        )

        result = get_current_weather(51.5, -0.1)
        assert result["temp_c"] == 15.0
        assert result["rain_1h_mm"] == 1.2
        assert result["weather_condition"] == "Rain"
