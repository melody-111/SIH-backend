import datetime
import math
from typing import Dict, Any, List, Optional, Tuple
from app.schemas.gis_map import (
    RouteAnalysisRequest,
    RouteAnalysisResponse,
    RouteWaypoint,
    GeoJSONGeometry
)
from app.connectors.open_meteo_marine import OpenMeteoMarineConnector
from app.connectors.open_meteo_weather import OpenMeteoWeatherConnector
from app.services.official_warning_provider import OfficialWarningProvider
from app.core.config import settings

class RouteIntelligenceService:
    """
    Maritime route intelligence service.
    
    Calculates geodesic Great Circle marine waypoints, true nautical distances,
    estimated transit times, live wave/wind/SST overlays along waypoints, and
    proximity to official cyclone/marine warning zones.
    
    Upholds strict safety rules:
    - Never declares a maritime route 'completely safe'.
    - Never presents straight lines as navigable shipping channels without disclaimer.
    - Transparently documents input data sources and environmental timestamps.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RouteIntelligenceService, cls).__new__(cls)
            cls._instance._marine_connector = OpenMeteoMarineConnector()
            cls._instance._weather_connector = OpenMeteoWeatherConnector()
            cls._instance._warning_provider = OfficialWarningProvider()
        return cls._instance

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance between two points in kilometers."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2)**2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Initial compass bearing in degrees (0-360)."""
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlon_r = math.radians(lon2 - lon1)

        y = math.sin(dlon_r) * math.cos(lat2_r)
        x = (math.cos(lat1_r) * math.sin(lat2_r) -
             math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def _interpolate_great_circle_waypoints(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        num_intermediate: int = 4
    ) -> List[Tuple[float, float]]:
        """
        Generates intermediate waypoints along the Great Circle arc.
        """
        points = [(lat1, lon1)]
        if num_intermediate <= 0:
            points.append((lat2, lon2))
            return points

        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)

        # Angular distance
        d = 2 * math.asin(math.sqrt(
            math.sin((lat2_r - lat1_r) / 2)**2 +
            math.cos(lat1_r) * math.cos(lat2_r) * math.sin((lon2_r - lon1_r) / 2)**2
        ))

        if d < 1e-6:
            points.append((lat2, lon2))
            return points

        for i in range(1, num_intermediate + 1):
            f = i / (num_intermediate + 1)
            A = math.sin((1 - f) * d) / math.sin(d)
            B = math.sin(f * d) / math.sin(d)
            x = A * math.cos(lat1_r) * math.cos(lon1_r) + B * math.cos(lat2_r) * math.cos(lon2_r)
            y = A * math.cos(lat1_r) * math.sin(lon1_r) + B * math.cos(lat2_r) * math.sin(lon2_r)
            z = A * math.sin(lat1_r) + B * math.sin(lat2_r)
            pt_lat = math.atan2(z, math.sqrt(x**2 + y**2))
            pt_lon = math.atan2(y, x)
            points.append((math.degrees(pt_lat), math.degrees(pt_lon)))

        points.append((lat2, lon2))
        return points

    async def analyze_route(self, request: RouteAnalysisRequest) -> RouteAnalysisResponse:
        """
        Conducts maritime route corridor analysis with live environmental and hazard overlay.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        o_lat = request.origin_latitude
        o_lon = request.origin_longitude
        d_lat = request.destination_latitude
        d_lon = request.destination_longitude

        # 1. Total geodesic distance
        total_dist_km = self._haversine_distance(o_lat, o_lon, d_lat, d_lon)
        total_dist_nm = round(total_dist_km * 0.539957, 2)

        # 2. Transit time estimate (assuming knots)
        speed_kmh = request.vessel_speed_knots * 1.852
        transit_hours = total_dist_km / max(speed_kmh, 1.0)
        est_transit_minutes = int(transit_hours * 60)

        # 3. Interpolate waypoints
        num_intermediate = 3 if total_dist_km > 20.0 else 1
        raw_waypoints = self._interpolate_great_circle_waypoints(o_lat, o_lon, d_lat, d_lon, num_intermediate)

        # 4. LineString GeoJSON coordinates [lon, lat]
        geojson_coords = [[round(lon, 4), round(lat, 4)] for lat, lon in raw_waypoints]

        # 5. Check environmental conditions along origin and destination
        origin_marine = await self._marine_connector.fetch_marine(o_lat, o_lon)
        dest_marine = await self._marine_connector.fetch_marine(d_lat, d_lon)
        origin_weather = await self._weather_connector.fetch_weather(o_lat, o_lon)

        o_wave = origin_marine.data.get("current", {}).get("wave_height") if origin_marine.success else None
        d_wave = dest_marine.data.get("current", {}).get("wave_height") if dest_marine.success else None
        o_wind = origin_weather.data.get("current", {}).get("wind_speed_10m") if origin_weather.success else None
        o_sst = origin_marine.data.get("current", {}).get("sea_surface_temperature") if origin_marine.success else None

        # Build waypoints data
        waypoints: List[RouteWaypoint] = []
        cum_dist = 0.0

        for idx, (wpt_lat, wpt_lon) in enumerate(raw_waypoints):
            if idx > 0:
                prev_lat, prev_lon = raw_waypoints[idx - 1]
                cum_dist += self._haversine_distance(prev_lat, prev_lon, wpt_lat, wpt_lon)

            # Compass bearing to next point
            if idx < len(raw_waypoints) - 1:
                next_lat, next_lon = raw_waypoints[idx + 1]
                bearing = self._calculate_bearing(wpt_lat, wpt_lon, next_lat, next_lon)
            else:
                bearing = self._calculate_bearing(raw_waypoints[idx - 1][0], raw_waypoints[idx - 1][1], wpt_lat, wpt_lon)

            # Interpolate estimated environmental conditions along route
            ratio = idx / max(len(raw_waypoints) - 1, 1)
            interp_wave = None
            if o_wave is not None and d_wave is not None:
                interp_wave = round(o_wave * (1 - ratio) + d_wave * ratio, 2)
            elif o_wave is not None:
                interp_wave = o_wave

            waypoints.append(
                RouteWaypoint(
                    waypoint_index=idx,
                    latitude=round(wpt_lat, 4),
                    longitude=round(wpt_lon, 4),
                    distance_from_origin_km=round(cum_dist, 1),
                    bearing_degrees=round(bearing, 1),
                    wave_height_m=interp_wave,
                    wind_speed_kmh=round(o_wind, 1) if o_wind is not None else None,
                    sst_celsius=round(o_sst, 1) if o_sst is not None else None
                )
            )

        # 6. Proximity analysis against active official warnings
        all_warnings = self._warning_provider.get_all_warnings(active_only=True)
        warnings_along_route: List[Dict[str, Any]] = []

        for w in all_warnings:
            if w.affected_coordinates:
                w_lat = w.affected_coordinates.get("latitude")
                w_lon = w.affected_coordinates.get("longitude")
                if w_lat is not None and w_lon is not None:
                    # Check distance from every waypoint to warning center
                    min_wpt_dist = min(self._haversine_distance(wpt.latitude, wpt.longitude, w_lat, w_lon) for wpt in waypoints)
                    if min_wpt_dist <= (w.coverage_radius_km + 25.0):
                        warnings_along_route.append({
                            "warning_id": w.warning_id,
                            "warning_type": w.warning_type,
                            "source_severity": w.source_severity,
                            "normalized_severity": w.normalized_severity,
                            "bulletin": w.bulletin_text,
                            "min_distance_to_route_km": round(min_wpt_dist, 1),
                            "intersects_route": min_wpt_dist <= w.coverage_radius_km
                        })

        # 7. Documented Departure Safety Rules
        max_wave = max([w.wave_height_m for w in waypoints if w.wave_height_m is not None], default=None)
        max_wind = max([w.wind_speed_kmh for w in waypoints if w.wind_speed_kmh is not None], default=None)

        has_intersecting_warning = any(w.get("intersects_route") for w in warnings_along_route)
        is_safe = True
        recommendation = "Route corridor is within standard operational limits. Maintain visual lookout and radio watch."

        if has_intersecting_warning:
            is_safe = False
            recommendation = "UNSAFE: Route intersects an active official severe weather warning area. Departure is strongly discouraged."
        elif max_wave and max_wave >= settings.WAVE_HEIGHT_WARNING_M:
            is_safe = False
            recommendation = f"CAUTION: Significant wave height ({max_wave} m) exceeds safety threshold ({settings.WAVE_HEIGHT_WARNING_M} m). Small vessels should avoid open water."
        elif max_wind and max_wind >= settings.WIND_SPEED_WARNING_KMH:
            is_safe = False
            recommendation = f"CAUTION: Wind speed ({max_wind} km/h) exceeds safe threshold ({settings.WIND_SPEED_WARNING_KMH} km/h)."

        return RouteAnalysisResponse(
            origin={"latitude": o_lat, "longitude": o_lon},
            destination={"latitude": d_lat, "longitude": d_lon},
            total_distance_km=round(total_dist_km, 2),
            total_distance_nautical_miles=total_dist_nm,
            estimated_transit_minutes=est_transit_minutes,
            cruising_speed_knots=request.vessel_speed_knots,
            route_geometry=GeoJSONGeometry(
                type="LineString",
                coordinates=geojson_coords
            ),
            waypoints=waypoints,
            warnings_along_route=warnings_along_route,
            max_wave_height_m=max_wave,
            max_wind_speed_kmh=max_wind,
            is_safe_to_depart=is_safe,
            departure_recommendation=recommendation,
            data_sources=["Open-Meteo Marine (CMEMS)", "Open-Meteo Weather (ECMWF)", "IMD/INCOIS Official Bulletins"],
            retrieval_timestamp=now_iso,
            navigational_limitation="Geodesic maritime corridor calculation with live environmental overlay. Does NOT replace official nautical navigational charts, local bathymetric surveys, or port traffic rules. Departure is not recommended when warnings intersect the corridor."
        )
