export type Metric =
  | "average_watts"
  | "weighted_average_watts"
  | "average_speed_kmh"
  | "distance_km"
  | "elevation_gain_m"
  | "moving_minutes";

export type Bucket = "month" | "week";

export type TimeseriesPoint = {
  date: string;
  value: number;
  count: number;
};

export type HeadwindPoint = {
  strava_id: number;
  headwind_mps: number;
  average_speed_kmh: number;
  average_watts: number | null;
  distance_km: number;
};

export type Activity = {
  strava_id: number;
  start_time: string;
  distance_km: number;
  elevation_gain_m: number;
  average_speed_kmh: number;
  average_watts: number | null;
  weighted_average_watts: number | null;
  moving_minutes: number;
  temp_c: number;
  wind_speed_mps: number;
  wind_deg: number;
  weather_condition: string;
  bearing_deg: number | null;
  headwind_mps: number | null;
  route_id: number | null;
};

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

export const fetchTimeseries = (metric: Metric, bucket: Bucket) =>
  get<TimeseriesPoint[]>(
    `/api/stats/timeseries?metric=${metric}&bucket=${bucket}`,
  );

export const fetchHeadwindScatter = () =>
  get<HeadwindPoint[]>(`/api/stats/headwind-scatter`);

export const fetchActivities = () => get<Activity[]>(`/api/activities`);

export type RouteWeather = {
  temp_c: number;
  wind_speed_mps: number;
  wind_deg: number;
  weather_condition: string;
  wind_alignment: number | null;
  direction: string | null;
};

export type ClusteredRoute = {
  route_id: number;
  centroid_lat: number;
  centroid_lon: number;
  activity_count: number;
  avg_distance_m: number;
  avg_elevation_gain_m: number;
  avg_elevation_per_km: number;
  avg_bearing_deg: number | null;
  label: string | null;
  location_name: string | null;
  weather: RouteWeather | null;
};

export const fetchRoutes = (withWeather = false) =>
  get<ClusteredRoute[]>(
    `/api/routes${withWeather ? "?with_weather=true" : ""}`,
  );

export type Recommendation = {
  route_id: number;
  label: string | null;
  location_name: string | null;
  avg_distance_km: number;
  avg_elevation_gain_m: number;
  predicted_speed_kmh: number;
  estimated_time_minutes: number;
  direction: string | null;
  wind_alignment: number | null;
  weather: {
    temp_c: number;
    wind_speed_mps: number;
    wind_deg: number;
    weather_condition: string;
  } | null;
  sample_activities: {
    strava_id: number;
    start_time: string;
    distance_km: number;
    average_speed_kmh: number;
  }[];
};

export const fetchRecommend = (topN: number, timeAvailableMinutes?: number) => {
  const params = new URLSearchParams({ top_n: String(topN) });
  if (timeAvailableMinutes != null) {
    params.set("time_available_minutes", String(timeAvailableMinutes));
  }
  return get<Recommendation[]>(`/api/recommend?${params}`);
};
