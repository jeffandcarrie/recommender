from datetime import timezone

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="ML-powered cycling route recommender based on Strava + weather data.")
console = Console()


@app.command()
def auth() -> None:
    """Authorize with Strava (first-time setup)."""
    from strava_recommender.strava_client import authorize

    authorize()


@app.command()
def sync() -> None:
    """Fetch cycling activities from Strava and enrich with weather data."""
    from strava_recommender.strava_client import fetch_cycling_activities
    from strava_recommender.storage import load_activities, save_activities
    from strava_recommender.weather_client import get_historical_weather

    console.print("Fetching cycling activities from Strava...")
    new_activities = fetch_cycling_activities()
    console.print(f"Found [bold]{len(new_activities)}[/bold] cycling activities.")

    # Merge with existing (avoid duplicates by strava_id)
    existing = load_activities()
    existing_ids = {a.strava_id for a in existing}
    # Keep existing enriched activities, only add new ones
    to_enrich = [a for a in new_activities if a.strava_id not in existing_ids]

    console.print(f"[bold]{len(to_enrich)}[/bold] new activities to enrich with weather data.")

    for i, activity in enumerate(to_enrich, 1):
        console.print(
            f"  [{i}/{len(to_enrich)}] Enriching activity {activity.strava_id} "
            f"({activity.start_time.strftime('%Y-%m-%d')})...",
            end="",
        )
        try:
            from strava_recommender.geo import temp_stress

            unix_ts = int(activity.start_time.replace(tzinfo=timezone.utc).timestamp())
            weather = get_historical_weather(activity.start_lat, activity.start_lon, unix_ts)
            for key, value in weather.items():
                setattr(activity, key, value)
            activity.temp_stress = temp_stress(activity.temp_c)
            console.print(" [green]OK[/green]")
        except Exception as e:
            console.print(f" [red]FAILED: {e}[/red]")

    all_activities = existing + to_enrich
    save_activities(all_activities)
    console.print(f"[green]Saved {len(all_activities)} total activities.[/green]")


@app.command()
def cluster() -> None:
    """Cluster activities into routes by start location."""
    from strava_recommender.clustering import cluster_activities
    from strava_recommender.storage import load_activities, save_activities, save_routes

    activities = load_activities()
    if not activities:
        console.print("[red]No activities found. Run 'sync' first.[/red]")
        raise typer.Exit(1)

    console.print(f"Clustering {len(activities)} activities...")
    updated_activities, routes = cluster_activities(activities)

    from strava_recommender.geo import bearing_to_compass, classify_elevation, geocode_routes

    console.print("Resolving route locations...")
    geocode_routes(routes)

    save_activities(updated_activities)
    save_routes(routes)

    assigned = sum(1 for a in updated_activities if a.route_id is not None)
    console.print(f"[green]Found {len(routes)} routes from {assigned} activities.[/green]")
    noise = len(updated_activities) - assigned
    if noise:
        console.print(f"[yellow]{noise} activities not assigned to any route (too isolated).[/yellow]")

    # Build lookup of route_id -> most recent activity for Strava links
    route_latest: dict[int, object] = {}
    for a in updated_activities:
        if a.route_id is not None:
            if a.route_id not in route_latest or a.start_time > route_latest[a.route_id].start_time:
                route_latest[a.route_id] = a

    table = Table(title="Routes", show_lines=False)
    table.add_column("ID", style="cyan")
    table.add_column("Location")
    table.add_column("Activities", justify="right")
    table.add_column("Direction", justify="center")
    table.add_column("Terrain", justify="center")
    table.add_column("Avg Distance (km)", justify="right")
    table.add_column("Avg Elevation (m)", justify="right")
    table.add_column("Label")
    table.add_column("Latest Activity", style="dim", no_wrap=True)

    for r in routes:
        direction = bearing_to_compass(r.avg_bearing_deg) if r.avg_bearing_deg is not None else "-"
        terrain = classify_elevation(r.avg_elevation_gain_m, r.avg_distance_m)
        latest = route_latest.get(r.route_id)
        link = f"https://www.strava.com/activities/{latest.strava_id}" if latest else "-"
        table.add_row(
            str(r.route_id),
            r.location_name or "-",
            str(r.activity_count),
            direction,
            terrain,
            f"{r.avg_distance_m / 1000:.1f}",
            f"{r.avg_elevation_gain_m:.0f}",
            r.label or "-",
            link,
        )

    console.print(table)


@app.command()
def train() -> None:
    """Train the ML model to predict cycling performance."""
    from strava_recommender.feature_engineering import TARGET_SPEED, build_feature_matrix
    from strava_recommender.predictor import train as train_model
    from strava_recommender.storage import load_activities, load_routes

    activities = load_activities()
    routes = load_routes()

    if not routes:
        console.print("[red]No routes found. Run 'cluster' first.[/red]")
        raise typer.Exit(1)

    df = build_feature_matrix(activities, routes)
    if df.empty:
        console.print("[red]No activities with assigned routes. Need more data.[/red]")
        raise typer.Exit(1)

    # Filter to activities with power data for better model quality
    has_power = df["weighted_average_watts"].notna().sum()
    total = len(df)
    if has_power >= 10:
        df = df.dropna(subset=["weighted_average_watts"])
        console.print(
            f"Training on [bold]{len(df)}[/bold] activities with power data "
            f"(of {total} total) across [bold]{len(routes)}[/bold] routes..."
        )
    else:
        console.print(
            f"[yellow]Only {has_power} activities have power data — training on all {total} "
            f"activities (power feature will be imputed).[/yellow]"
        )
        # Fill missing WAP with median for activities without power
        median_wap = df["weighted_average_watts"].median()
        if median_wap and not np.isnan(median_wap):
            df["weighted_average_watts"] = df["weighted_average_watts"].fillna(median_wap)
        else:
            df["weighted_average_watts"] = df["weighted_average_watts"].fillna(200.0)
        console.print(f"Training on [bold]{len(df)}[/bold] activities across [bold]{len(routes)}[/bold] routes...")

    # Train speed model
    result = train_model(df, target=TARGET_SPEED)

    table = Table(title="Model Performance (Speed)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Samples", str(result["sample_count"]))
    if result["r2"] is not None:
        table.add_row("R² (CV)", f"{result['r2']:.3f}")
        table.add_row("R² (Train)", f"{result['train_r2']:.3f}")
        table.add_row("MAE (m/s)", f"{result['mae']:.3f}")
        table.add_row("MAE (km/h)", f"{result['mae'] * 3.6:.2f}")
        mean_kmh = result["target_mean"] * 3.6
        std_kmh = result["target_std"] * 3.6
        table.add_row("Target mean", f"{mean_kmh:.1f} km/h")
        table.add_row("Target std", f"{std_kmh:.1f} km/h")

        # Diagnose issues
        gap = result["train_r2"] - result["r2"]
        if gap > 0.3:
            console.print("[yellow]Warning: Large gap between train and CV R² — model is overfitting.[/yellow]")
        if result["target_std"] < 0.3:
            console.print("[yellow]Warning: Very low speed variance — your rides are very consistent, "
                          "leaving little for the model to explain.[/yellow]")
    else:
        table.add_row("R²", "N/A (too few samples for CV)")

    console.print(table)

    # Feature importances
    importances = result.get("feature_importances", {})
    if importances:
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        imp_table = Table(title="Feature Importances (top 10)")
        imp_table.add_column("Feature", style="cyan")
        imp_table.add_column("Importance", justify="right")
        for name, imp in sorted_imp[:10]:
            bar = "█" * int(imp * 50)
            imp_table.add_row(name, f"{imp:.3f} {bar}")
        console.print(imp_table)

    # Train power model if data exists
    has_power = df["average_watts"].notna().sum()
    if has_power >= 5:
        df_power = df.dropna(subset=["average_watts"])
        result_power = train_model(df_power, target="average_watts")
        console.print(f"\n[green]Power model trained on {len(df_power)} activities (R²: {result_power['r2']:.3f}).[/green]")
    elif has_power > 0:
        console.print(f"\n[yellow]Only {has_power} activities with power data — need at least 5 to train power model.[/yellow]")

    console.print("[green]Model saved.[/green]")


@app.command()
def recommend(
    top_n: int = typer.Option(3, help="Number of routes to recommend"),
    time_available: float = typer.Option(None, help="Maximum time available in minutes"),
) -> None:
    """Recommend the best routes for current weather conditions."""
    from strava_recommender.geo import classify_difficulty, classify_temp
    from strava_recommender.recommender import recommend as do_recommend

    if time_available:
        hours = int(time_available // 60)
        mins = int(time_available % 60)
        time_str = f"{hours}h{mins:02d}m" if hours > 0 else f"{mins}m"
        console.print(f"Filtering for routes completable within [bold]{time_str}[/bold]")
    console.print("Fetching current weather and predicting performance...\n")
    results = do_recommend(top_n=top_n, time_available_minutes=time_available)

    if not results:
        return

    # Summary table
    table = Table(title="Route Recommendations")
    table.add_column("Rank", style="bold cyan", justify="center")
    table.add_column("Route", justify="right")
    table.add_column("Location")
    table.add_column("Label")
    table.add_column("Direction", justify="center")
    table.add_column("Terrain", justify="center")
    table.add_column("Wind", justify="center")
    table.add_column("Predicted", justify="right", style="green")
    table.add_column("Est. Time", justify="right")
    table.add_column("Temp", justify="right")
    table.add_column("Conditions")
    table.add_column("Example Route", style="dim", no_wrap=True)

    for i, r in enumerate(results, 1):
        route = r["route"]
        weather = r["weather"]
        direction = r.get("direction") or "-"
        alignment = r.get("wind_alignment")
        if alignment is not None:
            if alignment > 0.5:
                wind_label = "[green]Tail[/green]"
            elif alignment < -0.5:
                wind_label = "[red]Head[/red]"
            else:
                wind_label = "[yellow]Cross[/yellow]"
        else:
            wind_label = "-"
        temp_c = weather["temp_c"]
        terrain = classify_difficulty(route.avg_elevation_gain_m, route.avg_distance_m, temp_c)
        temp_label = classify_temp(temp_c)

        # Most recent activity link for quick route lookup
        activities = r.get("activities", [])
        if activities:
            most_recent = max(activities, key=lambda a: a.start_time)
            example_link = f"https://www.strava.com/activities/{most_recent.strava_id}"
        else:
            example_link = "-"

        est_min = r.get("estimated_time_minutes", 0)
        est_h = int(est_min // 60)
        est_m = int(est_min % 60)
        est_str = f"{est_h}h{est_m:02d}m" if est_h > 0 else f"{est_m}m"

        table.add_row(
            str(i),
            str(route.route_id),
            r.get("location", ""),
            route.label or "-",
            direction,
            terrain,
            wind_label,
            f"{r['predicted_speed_kmh']:.1f} km/h",
            est_str,
            f"{temp_c:.0f}°C ({temp_label})",
            weather.get("weather_condition", ""),
            example_link,
        )

    console.print(table)

    # Detailed breakdown per route
    for i, r in enumerate(results, 1):
        route = r["route"]
        weather = r["weather"]
        direction = r.get("direction") or "-"
        alignment = r.get("wind_alignment")
        if alignment is not None:
            if alignment > 0.5:
                wind_label = "[green]Tailwind[/green]"
            elif alignment < -0.5:
                wind_label = "[red]Headwind[/red]"
            else:
                wind_label = "[yellow]Crosswind[/yellow]"
        else:
            wind_label = "-"

        temp_c = weather["temp_c"]
        difficulty = classify_difficulty(route.avg_elevation_gain_m, route.avg_distance_m, temp_c)
        temp_label = classify_temp(temp_c)
        elev_ft = route.avg_elevation_gain_m * 3.281
        dist_mi = route.avg_distance_m / 1609.34

        est_min = r.get("estimated_time_minutes", 0)
        est_h = int(est_min // 60)
        est_m = int(est_min % 60)
        est_time = f"{est_h}h{est_m:02d}m" if est_h > 0 else f"{est_m}m"

        location = r.get("location", "Unknown")
        console.print(f"\n[bold cyan]#{i}[/bold cyan] Route {route.route_id} — {route.label or 'Unlabelled'} ({location})")
        console.print(
            f"  Direction: [bold]{direction}[/bold]  |  Terrain: [bold]{difficulty}[/bold] "
            f"({elev_ft:.0f} ft / {dist_mi:.1f} mi)  |  Wind: {wind_label}  |  Est. time: [bold]{est_time}[/bold]"
        )
        console.print(
            f"  Predicted: [green]{r['predicted_speed_kmh']:.1f} km/h[/green]  |  "
            f"Weather: {weather.get('weather_condition', '')} "
            f"{temp_c:.0f}°C ({temp_label}), wind {weather['wind_speed_mps']:.1f} m/s"
        )

        activities = r.get("activities", [])
        if activities:
            # Sort by date descending so most recent rides are first
            sorted_acts = sorted(activities, key=lambda a: a.start_time, reverse=True)
            console.print(f"  [dim]Matching activities ({len(sorted_acts)}):[/dim]")
            for act in sorted_acts[:5]:
                date_str = act.start_time.strftime("%Y-%m-%d")
                dist_km = act.distance_m / 1000
                console.print(
                    f"    {date_str}  {dist_km:.1f} km  "
                    f"{act.average_speed_mps * 3.6:.1f} km/h  "
                    f"https://www.strava.com/activities/{act.strava_id}"
                )
            if len(sorted_acts) > 5:
                console.print(f"    [dim]... and {len(sorted_acts) - 5} more[/dim]")


@app.command()
def hindsight(
    top_n: int = typer.Option(10, help="Number of worst choices to show"),
) -> None:
    """Analyse past rides and find your worst route choices."""
    from strava_recommender.geo import bearing_to_compass, classify_temp
    from strava_recommender.recommender import hindsight as do_hindsight

    console.print("Analysing route choices across all rides...\n")
    results = do_hindsight(top_n=top_n)

    if not results:
        console.print("[green]No bad choices found — you always picked well![/green]")
        return

    table = Table(title=f"Top {len(results)} Worst Route Choices")
    table.add_column("Rank", style="bold cyan", justify="center")
    table.add_column("Date")
    table.add_column("You Rode")
    table.add_column("Should Have")
    table.add_column("Actual", justify="right")
    table.add_column("Predicted", justify="right")
    table.add_column("Best Route", justify="right")
    table.add_column("Gap", justify="right", style="red")
    table.add_column("Conditions")
    table.add_column("Activity", style="dim", no_wrap=True)

    for i, r in enumerate(results, 1):
        activity = r["activity"]
        chosen = r["chosen_route"]
        best = r["best_route"]
        weather = r["weather"]

        chosen_dir = bearing_to_compass(chosen.avg_bearing_deg) if chosen.avg_bearing_deg else "-"
        best_dir = bearing_to_compass(best.avg_bearing_deg) if best.avg_bearing_deg else "-"
        chosen_label = f"{chosen.location_name or 'Route ' + str(chosen.route_id)} ({chosen_dir})"
        best_label = f"{best.location_name or 'Route ' + str(best.route_id)} ({best_dir})"

        temp_label = classify_temp(weather["temp_c"])
        wind_str = f"{weather['wind_speed_mps']:.0f}m/s"
        conditions = f"{weather['weather_condition']} {weather['temp_c']:.0f}°C ({temp_label}), {wind_str}"

        link = f"https://www.strava.com/activities/{activity.strava_id}"

        table.add_row(
            str(i),
            activity.start_time.strftime("%Y-%m-%d"),
            chosen_label,
            best_label,
            f"{r['actual_speed_kmh']:.1f} km/h",
            f"{r['chosen_speed_kmh']:.1f} km/h",
            f"{r['best_speed_kmh']:.1f} km/h",
            f"+{r['gap_kmh']:.1f} km/h",
            conditions,
            link,
        )

    console.print(table)

    # Show patterns
    console.print("\n[bold]Patterns:[/bold]")
    # What conditions led to bad choices?
    hot_count = sum(1 for r in results if r["weather"]["temp_c"] > 25)
    cold_count = sum(1 for r in results if r["weather"]["temp_c"] < 5)
    windy_count = sum(1 for r in results if r["weather"]["wind_speed_mps"] > 5)
    rainy_count = sum(1 for r in results if r["weather"]["rain_1h_mm"] > 0)

    if hot_count > len(results) * 0.3:
        console.print(f"  [yellow]Heat was a factor in {hot_count}/{len(results)} bad choices — consider flatter routes on hot days[/yellow]")
    if cold_count > len(results) * 0.3:
        console.print(f"  [blue]Cold was a factor in {cold_count}/{len(results)} bad choices — consider flatter routes on cold days[/blue]")
    if windy_count > len(results) * 0.3:
        console.print(f"  [yellow]Wind was a factor in {windy_count}/{len(results)} bad choices — pay more attention to wind direction[/yellow]")
    if rainy_count > len(results) * 0.3:
        console.print(f"  [blue]Rain was a factor in {rainy_count}/{len(results)} bad choices[/blue]")

    avg_gap = sum(r["gap_kmh"] for r in results) / len(results)
    console.print(f"  Average speed left on the table: [red]{avg_gap:.1f} km/h[/red]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Run the FastAPI dashboard server."""
    import uvicorn

    uvicorn.run(
        "strava_recommender.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def status() -> None:
    """Show current data and model status."""
    from pathlib import Path

    from strava_recommender.config import get_settings
    from strava_recommender.storage import load_activities, load_routes

    settings = get_settings()
    activities = load_activities()
    routes = load_routes()
    enriched = sum(1 for a in activities if a.weather_condition)
    with_route = sum(1 for a in activities if a.route_id is not None)

    table = Table(title="Status")
    table.add_column("Item", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Data directory", str(settings.data_dir))
    table.add_row("Total activities", str(len(activities)))
    table.add_row("Weather-enriched", str(enriched))
    table.add_row("Assigned to routes", str(with_route))
    table.add_row("Routes", str(len(routes)))

    model_path = settings.data_dir / "model_speed.joblib"
    if model_path.exists():
        import datetime

        mtime = datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
        table.add_row("Speed model", f"trained {mtime.strftime('%Y-%m-%d %H:%M')}")
    else:
        table.add_row("Speed model", "[red]not trained[/red]")

    power_path = settings.data_dir / "model_power.joblib"
    if power_path.exists():
        import datetime

        mtime = datetime.datetime.fromtimestamp(power_path.stat().st_mtime)
        table.add_row("Power model", f"trained {mtime.strftime('%Y-%m-%d %H:%M')}")
    else:
        table.add_row("Power model", "[dim]not trained[/dim]")

    console.print(table)


if __name__ == "__main__":
    app()
