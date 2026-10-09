from datetime import datetime, timezone

from strava_recommender.clustering import cluster_activities
from strava_recommender.models.activity import EnrichedActivity


def _make_activity(
    strava_id: int, lat: float, lon: float, bearing: float | None = None
) -> EnrichedActivity:
    return EnrichedActivity(
        strava_id=strava_id,
        start_lat=lat,
        start_lon=lon,
        distance_m=25000.0,
        elevation_gain_m=100.0,
        average_speed_mps=7.0,
        start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        elapsed_seconds=3600,
        moving_seconds=3500,
        bearing_deg=bearing,
    )


class TestClustering:
    def test_two_clusters(self, sample_activities):
        updated, routes = cluster_activities(sample_activities)
        assert len(routes) >= 2
        assert all(r.activity_count >= 2 for r in routes)
        assigned = sum(1 for a in updated if a.route_id is not None)
        assert assigned >= len(sample_activities) * 0.5  # at least half assigned

    def test_empty_input(self):
        updated, routes = cluster_activities([])
        assert updated == []
        assert routes == []

    def test_single_activity_is_noise(self):
        """A single activity can't form a cluster with min_samples=2."""
        activities = [_make_activity(0, 51.5, -0.1, bearing=90.0)]
        updated, routes = cluster_activities(activities)
        assert len(routes) == 0
        assert updated[0].route_id is None

    def test_same_location_different_bearing_separate_clusters(self):
        """Activities from the same start but different directions should separate."""
        activities = [
            # NE direction (~45°)
            _make_activity(0, 51.500, -0.100, bearing=44.0),
            _make_activity(1, 51.500, -0.100, bearing=45.0),
            _make_activity(2, 51.500, -0.100, bearing=46.0),
            # SW direction (~225°, opposite)
            _make_activity(3, 51.500, -0.100, bearing=224.0),
            _make_activity(4, 51.500, -0.100, bearing=225.0),
            _make_activity(5, 51.500, -0.100, bearing=226.0),
        ]
        updated, routes = cluster_activities(activities)
        assert len(routes) == 2

        # Verify activities with similar bearings got the same route_id
        ne_ids = {a.route_id for a in updated if a.bearing_deg < 100}
        sw_ids = {a.route_id for a in updated if a.bearing_deg > 100}
        assert len(ne_ids) == 1
        assert len(sw_ids) == 1
        assert ne_ids != sw_ids

    def test_similar_bearing_same_location_clusters_together(self):
        """Activities with similar bearing and location should cluster."""
        activities = [
            _make_activity(0, 51.500, -0.100, bearing=90.0),
            _make_activity(1, 51.500, -0.100, bearing=92.0),
            _make_activity(2, 51.500, -0.100, bearing=88.0),
        ]
        updated, routes = cluster_activities(activities)
        assert len(routes) == 1
        assert routes[0].activity_count == 3

    def test_no_bearing_falls_back_to_location(self):
        """Activities without bearing should cluster by location only."""
        activities = [
            _make_activity(0, 51.500, -0.100, bearing=None),
            _make_activity(1, 51.501, -0.101, bearing=None),
            _make_activity(2, 51.500, -0.099, bearing=None),
        ]
        updated, routes = cluster_activities(activities)
        assert len(routes) == 1
        assert routes[0].activity_count == 3
        assert routes[0].avg_bearing_deg is None

    def test_route_has_avg_bearing(self, sample_activities):
        _, routes = cluster_activities(sample_activities)
        for route in routes:
            assert route.avg_bearing_deg is not None
            assert 0 <= route.avg_bearing_deg < 360

    def test_route_centroids_are_reasonable(self, sample_activities):
        _, routes = cluster_activities(sample_activities)
        for route in routes:
            assert -90 <= route.centroid_lat <= 90
            assert -180 <= route.centroid_lon <= 180
            assert route.avg_distance_m > 0
            assert route.avg_elevation_gain_m >= 0
