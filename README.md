# Strava Recommender

This project recommends cycling routes using your Strava ride history and weather data.

## Requirements

- Python 3.11+
- uv

## Install dependencies

From the project root:

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root with your Strava credentials:

```env
STRAVA_CLIENT_ID=12345
STRAVA_CLIENT_SECRET=your_strava_client_secret
# Optional if you already have a refresh token you want to use:
# STRAVA_REFRESH_TOKEN=your_refresh_token
```

The app reads settings from `.env` automatically.

## First-time authorization

Authenticate with Strava so the app can fetch your activities:

```bash
uv run strava-recommender auth
```

This opens a browser for the Strava OAuth flow and saves the tokens in your local data directory.

## Run the sync

To fetch your cycling activities and enrich them with historical weather data:

```bash
uv run strava-recommender sync
```

This command:

- fetches recent Strava rides
- filters to cycling activities
- enriches each activity with weather data
- stores the results locally

## Useful follow-up commands

```bash
uv run strava-recommender cluster
uv run strava-recommender train
```

The `cluster` step groups rides into route patterns, and `train` trains the ML model.

## Run the backend

Start the FastAPI backend from the project root:

```bash
uv run uvicorn strava_recommender.api.app:app --reload --host 127.0.0.1 --port 8000
```

This serves the API at:

```text
http://127.0.0.1:8000
```

## Run the UI

In a second terminal, start the Vite frontend:

```bash
cd web
npm install
npm run dev -- --host 0.0.0.0
```

The UI runs at:

```text
http://localhost:5173
```

The frontend is configured to proxy `/api` requests to the backend at `http://127.0.0.1:8000`.

### Frontend configuration

Create a `web/.env` file to override the default map center and chart defaults without editing the code:

```env
VITE_MAP_CENTER_LAT=42.0837
VITE_MAP_CENTER_LON=-71.3967
VITE_MAP_ZOOM=11
VITE_DEFAULT_METRIC=distance_km
VITE_DEFAULT_BUCKET=week
```

These values control the initial map area and the default metrics chart selection. You can change them to your own city or the area you want to share with others.

## Full app flow

```bash
uv sync
uv run strava-recommender auth
uv run strava-recommender sync
uv run uvicorn strava_recommender.api.app:app --reload --host 127.0.0.1 --port 8000
cd web
npm install
npm run dev -- --host 0.0.0.0
```

Then open:

```text
http://localhost:5173
```
