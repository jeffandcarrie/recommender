import http.server
import threading
import urllib.parse
import webbrowser

import httpx
from rich.console import Console

from strava_recommender.analysis import calculate_normalized_power

from strava_recommender.config import get_settings
from strava_recommender.geo import decode_polyline, farthest_point_bearing
from strava_recommender.models.activity import EnrichedActivity
from strava_recommender.storage import load_tokens, save_tokens

console = Console()

STRAVA_API = "https://www.strava.com/api/v3"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"


def _refresh_tokens(refresh_token: str) -> dict:
    """Exchange a refresh token for new access + refresh tokens."""
    settings = get_settings()
    response = httpx.post(
        STRAVA_TOKEN_URL,
        data={
            "client_id": settings.strava_client_id,
            "client_secret": settings.strava_client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _get_access_token() -> str:
    """Get a valid access token, refreshing if needed."""
    settings = get_settings()

    # Try stored tokens first
    tokens = load_tokens()
    if tokens and tokens.get("refresh_token"):
        refreshed = _refresh_tokens(tokens["refresh_token"])
        save_tokens({
            "access_token": refreshed["access_token"],
            "refresh_token": refreshed["refresh_token"],
            "expires_at": refreshed["expires_at"],
        })
        return refreshed["access_token"]

    # Fall back to refresh token from env
    if settings.strava_refresh_token:
        refreshed = _refresh_tokens(settings.strava_refresh_token)
        save_tokens({
            "access_token": refreshed["access_token"],
            "refresh_token": refreshed["refresh_token"],
            "expires_at": refreshed["expires_at"],
        })
        return refreshed["access_token"]

    raise RuntimeError(
        "No tokens found. Run 'strava-recommender auth' first to authorize with Strava."
    )


def authorize() -> None:
    """Run the initial OAuth2 authorization flow with a local callback server."""
    settings = get_settings()
    settings.ensure_data_dir()
    redirect_uri = "http://localhost:8000/callback"

    auth_url = (
        f"{STRAVA_AUTH_URL}?client_id={settings.strava_client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope=read,activity:read_all"
    )

    code_holder: dict[str, str | None] = {"code": None}

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            code_holder["code"] = params.get("code", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authorization successful!</h1><p>You can close this tab.</p>")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("localhost", 8000), CallbackHandler)
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    console.print(f"\nOpening browser for Strava authorization...\n{auth_url}\n")
    webbrowser.open(auth_url)

    server_thread.join(timeout=120)
    server.server_close()

    if not code_holder["code"]:
        raise RuntimeError("Did not receive authorization code. Please try again.")

    # Exchange code for tokens
    response = httpx.post(
        STRAVA_TOKEN_URL,
        data={
            "client_id": settings.strava_client_id,
            "client_secret": settings.strava_client_secret,
            "code": code_holder["code"],
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    response.raise_for_status()
    token_data = response.json()

    save_tokens({
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "expires_at": token_data["expires_at"],
    })
    console.print("[green]Authorization successful! Tokens saved.[/green]")


def _fetch_activity_power_stream(activity_id: int, access_token: str) -> list[float]:
    """Fetch the power stream for a ride and return the raw watt values."""
    response = httpx.get(
        f"{STRAVA_API}/activities/{activity_id}/streams",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"keys": "time,watts", "key_by_type": True},
        timeout=30,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()

    data = response.json()
    watts_data = data.get("watts")
    if not watts_data:
        return []
    if isinstance(watts_data, dict):
        series = watts_data.get("data", [])
    else:
        series = watts_data
    return [float(v) for v in series if v is not None]


def fetch_cycling_activities() -> list[EnrichedActivity]:
    """Fetch all cycling activities from Strava."""
    from datetime import datetime

    access_token = _get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    activities: list[EnrichedActivity] = []
    page = 1
    per_page = 100

    while True:
        response = httpx.get(
            f"{STRAVA_API}/athlete/activities",
            headers=headers,
            params={"page": page, "per_page": per_page},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        for act in data:
            if act.get("type") != "Ride":
                continue

            start_latlng = act.get("start_latlng")
            if not start_latlng or len(start_latlng) < 2:
                continue

            bearing = None
            summary_polyline = (act.get("map") or {}).get("summary_polyline")
            if summary_polyline:
                points = decode_polyline(summary_polyline)
                bearing = farthest_point_bearing(start_latlng[0], start_latlng[1], points)

            distance = act.get("distance", 0.0)
            elevation = act.get("total_elevation_gain", 0.0)
            elev_per_km = (elevation / (distance / 1000)) if distance > 0 else 0.0

            activity = EnrichedActivity(
                strava_id=act["id"],
                start_lat=start_latlng[0],
                start_lon=start_latlng[1],
                distance_m=distance,
                elevation_gain_m=elevation,
                elevation_per_km=elev_per_km,
                average_speed_mps=act.get("average_speed", 0.0),
                average_watts=act.get("average_watts"),
                weighted_average_watts=act.get("weighted_average_watts"),
                start_time=datetime.fromisoformat(act["start_date"]),
                elapsed_seconds=act.get("elapsed_time", 0),
                moving_seconds=act.get("moving_time", 0),
                bearing_deg=bearing,
            )

            power_values = _fetch_activity_power_stream(activity.strava_id, access_token)
            if power_values:
                activity.np_5m_watts = calculate_normalized_power(power_values, 5 * 60)
                activity.np_20m_watts = calculate_normalized_power(power_values, 20 * 60)

            activities.append(activity)

        page += 1

    return activities
