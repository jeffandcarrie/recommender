/// <reference types="vite/client" />

import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  Tooltip as LeafletTooltip,
} from "react-leaflet";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import {
  Bucket,
  ClusteredRoute,
  Metric,
  Recommendation,
  fetchActivities,
  fetchHeadwindScatter,
  fetchPerformanceGains,
  fetchRecommend,
  fetchRoutes,
  fetchTimeseries,
  fetchWeeklyPerformanceGains,
} from "./api";
import {
  UnitSystem,
  UnitsContext,
  c,
  distUnit,
  elevUnit,
  km,
  kmh,
  m,
  mps,
  speedUnit,
  tempUnit,
  useUnits,
  windUnit,
} from "./units";

type MetricMeta = {
  value: Metric;
  label: string;
  unit: (u: UnitSystem) => string;
  convert: (v: number, u: UnitSystem) => number;
};

const readNumberEnv = (name: string, fallback: number) => {
  const value = Number(import.meta.env[name]);
  return Number.isFinite(value) ? value : fallback;
};

const readMetricEnv = (name: string, fallback: Metric): Metric => {
  const valid: Metric[] = [
    "average_watts",
    "weighted_average_watts",
    "average_speed_kmh",
    "distance_km",
    "elevation_gain_m",
    "moving_minutes",
  ];
  const value = import.meta.env[name] as string | undefined;
  return valid.includes(value as Metric) ? (value as Metric) : fallback;
};

const readBucketEnv = (name: string, fallback: Bucket): Bucket => {
  const value = import.meta.env[name] as string | undefined;
  return value === "month" || value === "week" ? (value as Bucket) : fallback;
};

const APP_CONFIG = {
  mapCenter: [
    readNumberEnv("VITE_MAP_CENTER_LAT", 42.0837),
    readNumberEnv("VITE_MAP_CENTER_LON", -71.3967),
  ] as [number, number],
  mapZoom: Math.round(readNumberEnv("VITE_MAP_ZOOM", 11)),
  defaultMetric: readMetricEnv("VITE_DEFAULT_METRIC", "distance_km"),
  defaultBucket: readBucketEnv("VITE_DEFAULT_BUCKET", "week"),
};

const METRICS: MetricMeta[] = [
  { value: "average_watts", label: "Average Watts", unit: () => "W", convert: (v) => v },
  { value: "weighted_average_watts", label: "Weighted Avg Watts", unit: () => "W", convert: (v) => v },
  { value: "average_speed_kmh", label: "Average Speed", unit: speedUnit, convert: kmh },
  { value: "distance_km", label: "Distance", unit: distUnit, convert: km },
  { value: "elevation_gain_m", label: "Elevation Gain", unit: elevUnit, convert: m },
  { value: "moving_minutes", label: "Moving Time", unit: () => "min", convert: (v) => v },
];

function TimeseriesCard() {
  const units = useUnits();
  const [metric, setMetric] = useState<Metric>(APP_CONFIG.defaultMetric);
  const [bucket, setBucket] = useState<Bucket>(APP_CONFIG.defaultBucket);
  const { data, isLoading, error } = useQuery({
    queryKey: ["timeseries", metric, bucket],
    queryFn: () => fetchTimeseries(metric, bucket),
  });
  const meta = METRICS.find((m) => m.value === metric)!;
  const unit = meta.unit(units);
  const points = data?.map((p) => ({ ...p, value: meta.convert(p.value, units) }));

  return (
    <div className="card">
      <div className="controls">
        <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
          {METRICS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
        <select value={bucket} onChange={(e) => setBucket(e.target.value as Bucket)}>
          <option value="month">Monthly</option>
          <option value="week">Weekly</option>
        </select>
      </div>
      <h2>
        {meta.label} ({unit}) — {bucket === "month" ? "monthly" : "weekly"}{" "}
        {metric === "distance_km" || metric === "elevation_gain_m" || metric === "moving_minutes"
          ? "total"
          : "avg"}
      </h2>
      {isLoading && <div className="muted">Loading…</div>}
      {error && <div className="error">{(error as Error).message}</div>}
      {points && points.length === 0 && (
        <div className="muted">No data for this metric yet.</div>
      )}
      {points && points.length > 0 && (
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={points} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid stroke="#232834" strokeDasharray="3 3" />
            <XAxis dataKey="date" stroke="#9aa0a8" fontSize={11} />
            <YAxis stroke="#9aa0a8" fontSize={11} />
            <Tooltip
              contentStyle={{ background: "#161a22", border: "1px solid #2a3040" }}
              formatter={(value: number) => `${value.toFixed(1)} ${unit}`}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#fc4c02"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

function HeadwindScatterCard() {
  const units = useUnits();
  const { data, isLoading, error } = useQuery({
    queryKey: ["headwind"],
    queryFn: fetchHeadwindScatter,
  });

  const points = data?.map((p) => ({
    ...p,
    headwind: mps(p.headwind_mps, units),
    speed: kmh(p.average_speed_kmh, units),
    distance: km(p.distance_km, units),
  }));

  return (
    <div className="card">
      <h2>Speed vs. Headwind</h2>
      <p className="muted" style={{ marginTop: 0, fontSize: 12 }}>
        Positive headwind = wind against you. Each dot = one ride.
      </p>
      {isLoading && <div className="muted">Loading…</div>}
      {error && <div className="error">{(error as Error).message}</div>}
      {points && points.length === 0 && (
        <div className="muted">No weather-enriched activities with bearing data yet.</div>
      )}
      {points && points.length > 0 && (
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid stroke="#232834" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="headwind"
              name="Headwind"
              unit={` ${windUnit(units)}`}
              stroke="#9aa0a8"
              fontSize={11}
            />
            <YAxis
              type="number"
              dataKey="speed"
              name="Avg speed"
              unit={` ${speedUnit(units)}`}
              stroke="#9aa0a8"
              fontSize={11}
            />
            <ZAxis type="number" dataKey="distance" range={[40, 240]} name="Distance" />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{ background: "#161a22", border: "1px solid #2a3040" }}
              formatter={(value: number, name: string) => {
                if (name === "Headwind") return `${value.toFixed(1)} ${windUnit(units)}`;
                if (name === "Avg speed") return `${value.toFixed(1)} ${speedUnit(units)}`;
                if (name === "Distance") return `${value.toFixed(1)} ${distUnit(units)}`;
                return value;
              }}
            />
            <Scatter data={points} fill="#fc4c02" fillOpacity={0.6} />
          </ScatterChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

function SummaryCard() {
  const units = useUnits();
  const { data } = useQuery({ queryKey: ["activities"], queryFn: fetchActivities });

  if (!data) return null;

  const total = data.length;
  const totalKm = data.reduce((s, a) => s + a.distance_km, 0);
  const totalElev = data.reduce((s, a) => s + a.elevation_gain_m, 0);
  const withPower = data.filter((a) => a.average_watts != null);
  const avgWatts =
    withPower.length > 0
      ? withPower.reduce((s, a) => s + (a.average_watts ?? 0), 0) / withPower.length
      : null;
  const avgSpeed =
    total > 0 ? data.reduce((s, a) => s + a.average_speed_kmh, 0) / total : 0;

  return (
    <div className="card">
      <h2>Lifetime</h2>
      <div className="stat-grid">
        <div className="stat">
          <div className="label">Rides</div>
          <div className="value">{total}</div>
        </div>
        <div className="stat">
          <div className="label">Distance</div>
          <div className="value">
            {km(totalKm, units).toFixed(0)} {distUnit(units)}
          </div>
        </div>
        <div className="stat">
          <div className="label">Elevation</div>
          <div className="value">
            {m(totalElev, units).toFixed(0)} {elevUnit(units)}
          </div>
        </div>
        <div className="stat">
          <div className="label">Avg speed</div>
          <div className="value">
            {kmh(avgSpeed, units).toFixed(1)} {speedUnit(units)}
          </div>
        </div>
        <div className="stat">
          <div className="label">Avg watts</div>
          <div className="value">
            {avgWatts != null ? `${avgWatts.toFixed(0)} W` : "—"}
          </div>
        </div>
      </div>
    </div>
  );
}

function windLabel(alignment: number | null): string {
  if (alignment == null) return "—";
  if (alignment > 0.5) return "Tailwind";
  if (alignment < -0.5) return "Headwind";
  return "Crosswind";
}

function compassFromDeg(deg: number): string {
  const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return dirs[Math.round(((deg % 360) / 45)) % 8];
}

function formatTime(minutes: number): string {
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return h > 0 ? `${h}h${m.toString().padStart(2, "0")}m` : `${m}m`;
}

function formatIsoWeek(dateInput: string): string {
  const date = new Date(dateInput);
  if (Number.isNaN(date.getTime())) return "—";

  const isoDate = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const day = isoDate.getUTCDay() || 7;
  isoDate.setUTCDate(isoDate.getUTCDate() + 4 - day);

  const yearStart = new Date(Date.UTC(isoDate.getUTCFullYear(), 0, 1));
  const weekNumber = Math.ceil((((isoDate.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  return `${isoDate.getUTCFullYear()}-W${weekNumber}`;
}

function parseDuration(input: string): number | null {
  const s = input.trim().toLowerCase();
  if (!s) return null;
  // Accept "1h30m", "90m", "90", "1.5h", "1:30"
  const colon = s.match(/^(\d+):(\d{1,2})$/);
  if (colon) return Number(colon[1]) * 60 + Number(colon[2]);
  const hm = s.match(/^(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+)m)?$/);
  if (hm && (hm[1] || hm[2])) return (Number(hm[1] || 0)) * 60 + Number(hm[2] || 0);
  const numeric = Number(s);
  if (!Number.isNaN(numeric)) return numeric;
  return null;
}

function PerformanceGainCard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["performance-gains"],
    queryFn: fetchPerformanceGains,
  });

  return (
    <div className="card">
      <h2>Performance gains by ride type</h2>
      <p className="muted" style={{ marginTop: 0, fontSize: 12 }}>
        Biggest positive gains versus each route&apos;s historical baseline. Higher is better.
      </p>
      {isLoading && <div className="muted">Loading gains…</div>}
      {error && <div className="error">{(error as Error).message}</div>}
      {data && data.length === 0 && (
        <div className="muted">Not enough route history yet to compare gains.</div>
      )}
      {data && data.length > 0 && (
        <div className="routes-list">
          {data.slice(0, 8).map((row) => (
            <div key={row.strava_id} className="route-row">
              <div className="route-row-head">
                <span className="route-row-title">
                  {row.label || `Route ${row.route_id}`}
                  {row.location_name && (
                    <span className="muted"> — {row.location_name}</span>
                  )}
                </span>
                <span className="muted" style={{ fontSize: 12 }}>
                  +{row.improvement_watts.toFixed(0)} W
                </span>
              </div>
              <div className="route-row-weather">
                <span>{new Date(row.start_time).toLocaleDateString()}</span>
                <span>
                  {row.power_metric} {row.current_power_watts.toFixed(0)} W vs {row.baseline_power_watts.toFixed(0)} W
                </span>
                <span>
                  {row.weather_condition || "—"} · {row.temp_c.toFixed(0)}°C
                </span>
                {row.weighted_average_watts != null && (
                  <span>{row.weighted_average_watts.toFixed(0)} W avg effort</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function WeeklyPerformanceGainCard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["weekly-performance-gains"],
    queryFn: fetchWeeklyPerformanceGains,
  });

  const chartData = data
    ? data
        .slice()
        .sort(
          (a, b) => new Date(a.week_start).getTime() - new Date(b.week_start).getTime(),
        )
        .map((row) => ({
          week: formatIsoWeek(row.week_start),
          volume_km: row.total_distance_km,
          gain_watts: row.total_improvement_watts,
        }))
    : [];

  return (
    <div className="card">
      <h2>Best power-gain weeks</h2>
      <p className="muted" style={{ marginTop: 0, fontSize: 12 }}>
        Week-start dates. Each point represents one full ISO week of riding.
      </p>
      {isLoading && <div className="muted">Loading weekly gains…</div>}
      {error && <div className="error">{(error as Error).message}</div>}
      {data && data.length === 0 && (
        <div className="muted">Not enough weekly route history yet to compare gains.</div>
      )}
      {chartData.length > 0 && (
        <div style={{ height: 180, marginBottom: 18 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
              <CartesianGrid stroke="#232834" strokeDasharray="3 3" />
              <XAxis dataKey="week" stroke="#9aa0a8" fontSize={11} />
              <YAxis yAxisId="left" stroke="#9aa0a8" fontSize={11} />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#9aa0a8"
                fontSize={11}
              />
              <Tooltip
                contentStyle={{ background: "#161a22", border: "1px solid #2a3040" }}
                formatter={(value: number, name: string) => {
                  if (name === "gain_watts") return `${value.toFixed(0)} W`;
                  if (name === "volume_km") return `${value.toFixed(0)} km`;
                  return value;
                }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="gain_watts"
                stroke="#fc4c02"
                strokeWidth={2}
                dot={{ r: 3 }}
                name="gain_watts"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="volume_km"
                stroke="#0f8f47"
                strokeWidth={2}
                dot={{ r: 2 }}
                name="volume_km"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {data && data.length > 0 && (
        <div className="routes-list">
          {data.slice(0, 10).map((row) => (
            <div key={row.week} className="route-row">
              <div className="route-row-head">
                <span className="route-row-title">{formatIsoWeek(row.week_start)}</span>
                <span className="muted" style={{ fontSize: 12 }}>
                  +{row.total_improvement_watts.toFixed(0)} W
                </span>
              </div>
              <div className="route-row-weather">
                <span>{row.key_driver}</span>
                <span>{row.ride_count} rides</span>
                <span>{row.hard_session_count} hard sessions</span>
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                {row.training_summary}
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>
                Route mix: {row.route_names.slice(0, 3).join(" · ") || "Route mix"}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RecommendationItem({ rec, rank }: { rec: Recommendation; rank: number }) {
  const units = useUnits();
  const latest = rec.sample_activities[0];
  return (
    <div className="rec">
      <div className="rec-head">
        <span className="rec-rank">#{rank}</span>
        <span className="rec-title">
          {rec.label || `Route ${rec.route_id}`}
          {rec.location_name && (
            <span className="muted"> — {rec.location_name}</span>
          )}
        </span>
        <span className="rec-speed">
          {kmh(rec.predicted_speed_kmh, units).toFixed(1)} {speedUnit(units)}
        </span>
      </div>
      <div className="rec-meta">
        <span>
          {km(rec.avg_distance_km, units).toFixed(1)} {distUnit(units)}
        </span>
        <span>
          {m(rec.avg_elevation_gain_m, units).toFixed(0)} {elevUnit(units)} elev
        </span>
        <span>{formatTime(rec.estimated_time_minutes)}</span>
        {rec.direction && <span>{rec.direction}</span>}
        <span>{windLabel(rec.wind_alignment)}</span>
        {rec.weather && (
          <span>
            {rec.weather.weather_condition}{" "}
            {c(rec.weather.temp_c, units).toFixed(0)}
            {tempUnit(units)} · wind {mps(rec.weather.wind_speed_mps, units).toFixed(1)}{" "}
            {windUnit(units)} {compassFromDeg(rec.weather.wind_deg)}
          </span>
        )}
      </div>
      {rec.sample_activities.length > 0 && (
        <div className="rec-links">
          <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
            Recent rides on this route:
          </div>
          {latest && (
            <a
              href={`https://www.strava.com/activities/${latest.strava_id}`}
              target="_blank"
              rel="noreferrer"
            >
              latest: {new Date(latest.start_time).toLocaleDateString()}
            </a>
          )}
          {rec.sample_activities.slice(1).map((a) => (
            <a
              key={a.strava_id}
              href={`https://www.strava.com/activities/${a.strava_id}`}
              target="_blank"
              rel="noreferrer"
            >
              {new Date(a.start_time).toLocaleDateString()}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function RecommendCard() {
  const [topN, setTopN] = useState(3);
  const [duration, setDuration] = useState("");
  const [durationError, setDurationError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () => {
      let minutes: number | undefined;
      if (duration.trim()) {
        const parsed = parseDuration(duration);
        if (parsed == null || parsed <= 0) {
          throw new Error(`Couldn't parse duration "${duration}". Try "90m", "1h30m", or "1:30".`);
        }
        minutes = parsed;
      }
      return fetchRecommend(topN, minutes);
    },
    onError: (e) => setDurationError((e as Error).message),
    onMutate: () => setDurationError(null),
  });

  return (
    <div className="card">
      <h2>Recommendations</h2>
      <div className="controls">
        <select
          value={topN}
          onChange={(e) => setTopN(Number(e.target.value))}
        >
          {[1, 3, 5, 10].map((n) => (
            <option key={n} value={n}>
              Top {n}
            </option>
          ))}
        </select>
        <input
          type="text"
          placeholder="duration (e.g. 1h30m)"
          value={duration}
          onChange={(e) => setDuration(e.target.value)}
          style={{
            background: "#0f1115",
            color: "#e6e8ec",
            border: "1px solid #2a3040",
            borderRadius: 4,
            padding: "6px 10px",
            fontSize: 13,
            width: 180,
          }}
        />
        <button
          className="btn"
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
        >
          {mutation.isPending ? "Computing…" : "Recommend"}
        </button>
      </div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
        Leave duration blank for any length. Routes within 70–100% of your time budget are returned.
      </div>
      {durationError && <div className="error">{durationError}</div>}
      {mutation.data && mutation.data.length === 0 && !mutation.isPending && (
        <div className="muted">No recommendations match your criteria.</div>
      )}
      {mutation.data &&
        mutation.data.map((rec, i) => (
          <RecommendationItem key={rec.route_id} rec={rec} rank={i + 1} />
        ))}
    </div>
  );
}

// wind_deg is the direction the wind is coming FROM. We project the tail in the
// direction the wind is blowing TOWARD, scaled to the map by approximate metres-per-degree.
function windTailLatLng(r: ClusteredRoute): [number, number] | null {
  if (!r.weather) return null;
  const speed = r.weather.wind_speed_mps;
  // Make the tail clearly visible and long enough to read from a distance.
  const meters = Math.min(35_000, 2500 + speed * 2500);
  const towardRad = ((r.weather.wind_deg + 180) * Math.PI) / 180;
  const dLat = (meters * Math.cos(towardRad)) / 111_320;
  const dLon =
    (meters * Math.sin(towardRad)) /
    (111_320 * Math.cos((r.centroid_lat * Math.PI) / 180));
  return [r.centroid_lat + dLat, r.centroid_lon + dLon];
}

function RoutesMap({
  routes,
  hoveredId,
  onHover,
}: {
  routes: ClusteredRoute[];
  hoveredId: number | null;
  onHover: (id: number | null) => void;
}) {
  return (
    <MapContainer
      className="routes-map"
      center={APP_CONFIG.mapCenter}
      zoom={APP_CONFIG.mapZoom}
      scrollWheelZoom={true}
    >
      <TileLayer
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        maxZoom={19}
      />
      {routes.map((r) => {
        const isHovered = hoveredId === r.route_id;
        const radius = 5 + Math.min(10, Math.log2(r.activity_count + 1) * 2);
        const tail = windTailLatLng(r);
        return (
          <CircleMarker
            key={r.route_id}
            center={[r.centroid_lat, r.centroid_lon]}
            radius={radius}
            pathOptions={{
              color: isHovered ? "#ffd1bf" : "#161a22",
              weight: isHovered ? 2 : 1,
              fillColor: "#fc4c02",
              fillOpacity: isHovered ? 1 : 0.8,
            }}
            eventHandlers={{
              mouseover: () => onHover(r.route_id),
              mouseout: () => onHover(null),
            }}
          >
            <LeafletTooltip direction="top" offset={[0, -radius]}>
              {r.label || `Route ${r.route_id}`}
              {r.weather && (
                <>
                  {" — "}
                  {r.weather.weather_condition || "—"}{" "}
                  {Math.round(r.weather.temp_c)}°C, wind{" "}
                  {r.weather.wind_speed_mps.toFixed(1)} m/s
                </>
              )}
            </LeafletTooltip>
            {tail && (
              <Polyline
                positions={[[r.centroid_lat, r.centroid_lon], tail]}
                pathOptions={{ color: "#0f8f47", weight: 8, opacity: 1, lineCap: "round", lineJoin: "round" }}
              />
            )}
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}

function RoutesCard() {
  const units = useUnits();
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const { data, isLoading, error } = useQuery({
    queryKey: ["routes", "with-weather"],
    queryFn: () => fetchRoutes(true),
    staleTime: 5 * 60 * 1000,
  });

  const sorted = data
    ? [...data].sort((a, b) => b.activity_count - a.activity_count)
    : [];

  return (
    <div className="card">
      <h2>Route starting points</h2>
      <p className="muted" style={{ marginTop: 0, fontSize: 12 }}>
        Current weather at each cluster's centroid. Green tail = wind blowing from that
        direction. Dot size scales with ride count.
      </p>
      {isLoading && <div className="muted">Loading current weather…</div>}
      {error && <div className="error">{(error as Error).message}</div>}
      {sorted.length > 0 && (
        <>
          <RoutesMap routes={sorted} hoveredId={hoveredId} onHover={setHoveredId} />
          <div className="routes-list">
            {sorted.map((r) => (
              <div
                key={r.route_id}
                className={`route-row${hoveredId === r.route_id ? " hovered" : ""}`}
                onMouseEnter={() => setHoveredId(r.route_id)}
                onMouseLeave={() => setHoveredId(null)}
              >
                <div className="route-row-head">
                  <span className="route-row-title">
                    {r.label || `Route ${r.route_id}`}
                    {r.location_name && (
                      <span className="muted"> — {r.location_name}</span>
                    )}
                  </span>
                  <span className="muted" style={{ fontSize: 12 }}>
                    {r.activity_count} ride{r.activity_count === 1 ? "" : "s"} ·{" "}
                    {km(r.avg_distance_m / 1000, units).toFixed(1)} {distUnit(units)}
                  </span>
                </div>
                {r.weather ? (
                  <div className="route-row-weather">
                    <span>
                      {r.weather.weather_condition || "—"}{" "}
                      {c(r.weather.temp_c, units).toFixed(0)}
                      {tempUnit(units)}
                    </span>
                    <span>
                      wind {mps(r.weather.wind_speed_mps, units).toFixed(1)}{" "}
                      {windUnit(units)} from {compassFromDeg(r.weather.wind_deg)}
                    </span>
                    {r.weather.direction && (
                      <span>
                        ride heads {r.weather.direction} ({windLabel(r.weather.wind_alignment)})
                      </span>
                    )}
                  </div>
                ) : (
                  <div className="muted" style={{ fontSize: 12 }}>
                    Weather unavailable
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function UnitsToggle({
  units,
  setUnits,
}: {
  units: UnitSystem;
  setUnits: (u: UnitSystem) => void;
}) {
  return (
    <div className="units-toggle">
      <button
        className={units === "metric" ? "active" : ""}
        onClick={() => setUnits("metric")}
      >
        Metric
      </button>
      <button
        className={units === "imperial" ? "active" : ""}
        onClick={() => setUnits("imperial")}
      >
        Imperial
      </button>
    </div>
  );
}

export function App() {
  const [units, setUnits] = useState<UnitSystem>(() => {
    const stored = localStorage.getItem("units");
    return stored === "imperial" ? "imperial" : "metric";
  });

  useEffect(() => {
    localStorage.setItem("units", units);
  }, [units]);

  return (
    <UnitsContext.Provider value={units}>
      <div className="app">
        <div className="header">
          <div>
            <h1>Strava Recommender</h1>
            <div className="subtitle">Cycling performance dashboard</div>
          </div>
          <UnitsToggle units={units} setUnits={setUnits} />
        </div>
        <SummaryCard />
        <PerformanceGainCard />
        <WeeklyPerformanceGainCard />
        <RecommendCard />
        <RoutesCard />
        <TimeseriesCard />
        <HeadwindScatterCard />
      </div>
    </UnitsContext.Provider>
  );
}
