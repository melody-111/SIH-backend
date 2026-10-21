import datetime
import math
from typing import Dict, Any, List, Optional, Tuple
from app.schemas.gis_map import (
    GeoJSONGeometry,
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    MapLayerInfo,
    CoastalLocationItem
)
from app.services.marine_data_service import KNOWN_COASTAL_LOCATIONS
from app.connectors.open_meteo_marine import OpenMeteoMarineConnector
from app.connectors.open_meteo_weather import OpenMeteoWeatherConnector
from app.services.official_warning_provider import OfficialWarningProvider
from app.services.offline_cache import OfflineMarineCache

AVAILABLE_MAP_LAYERS: List[MapLayerInfo] = [
    MapLayerInfo(
        layer_id="sst",
        name="Sea Surface Temperature (SST)",
        description="Gridded coastal water temperatures derived from satellite radiance and ocean models.",
        provider="Open-Meteo Marine / Copernicus Marine Service (CMEMS)",
        units="°C",
        data_type="OBSERVATION",
        is_live_supported=True,
        is_offline_supported=True,
        status="ACTIVE"
    ),
    MapLayerInfo(
        layer_id="waves",
        name="Significant Wave Height & Swell",
        description="Combined sea wave height, primary swell wave height, and wave period.",
        provider="Open-Meteo Marine / ECMWF WAM",
        units="m",
        data_type="OBSERVATION",
        is_live_supported=True,
        is_offline_supported=True,
        status="ACTIVE"
    ),
    MapLayerInfo(
        layer_id="wind",
        name="Surface Wind Velocity",
        description="10-meter coastal wind speed and wind gust indicators.",
        provider="Open-Meteo Weather / ECMWF IFS",
        units="km/h",
        data_type="OBSERVATION",
        is_live_supported=True,
        is_offline_supported=True,
        status="ACTIVE"
    ),
    MapLayerInfo(
        layer_id="warnings",
        name="Official Severe Weather & Cyclone Warnings",
        description="Government bulletins and coastal danger zones issued by meteorological authorities.",
        provider="India Meteorological Department (IMD) / INCOIS",
        units="coverage radius",
        data_type="OFFICIAL_ADVISORY",
        is_live_supported=True,
        is_offline_supported=True,
        status="ACTIVE"
    ),
    MapLayerInfo(
        layer_id="pfz",
        name="Potential Fishing Zones (PFZ)",
        description="Oceanographic thermal front and chlorophyll advisory zones.",
        provider="INCOIS (Indian National Centre for Ocean Information Services)",
        units="prospective zone",
        data_type="DERIVED_PRODUCT",
        is_live_supported=False,
        is_offline_supported=False,
        status="REQUIRES_AUTH"
    )
]

class MarineGISService:
    """
    Modular Marine GIS service.
    
    Generates standards-compliant WGS84 GeoJSON [longitude, latitude] features
    for Sea Surface Temperature, wave height, winds, and official warning zones.
    Strictly avoids fabricating unverified PFZ or satellite layers.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MarineGISService, cls).__new__(cls)
            cls._instance._marine_connector = OpenMeteoMarineConnector()
            cls._instance._weather_connector = OpenMeteoWeatherConnector()
            cls._instance._warning_provider = OfficialWarningProvider()
            cls._instance._cache = OfflineMarineCache()
        return cls._instance

    def get_layer_catalog(self) -> List[MapLayerInfo]:
        """Returns the list of available modular map layers and their capabilities."""
        return AVAILABLE_MAP_LAYERS

    def validate_bounds(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates geographical bounding box constraints and request limits.
        """
        if not (-90.0 <= min_lat <= 90.0) or not (-90.0 <= max_lat <= 90.0):
            return False, "Latitude values must be between -90.0 and 90.0."
        if not (-180.0 <= min_lon <= 180.0) or not (-180.0 <= max_lon <= 180.0):
            return False, "Longitude values must be between -180.0 and 180.0."
        if min_lat > max_lat:
            return False, "min_lat cannot be greater than max_lat."
        if min_lon > max_lon:
            return False, "min_lon cannot be greater than max_lon."

        # Request area limit to avoid overloading external numerical servers (max 15x15 deg)
        if (max_lat - min_lat) > 15.0 or (max_lon - min_lon) > 15.0:
            return False, "Geographical bounding box exceeds maximum request limit of 15° x 15°."

        return True, None

    def _create_circle_polygon(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        num_points: int = 24
    ) -> List[List[float]]:
        """
        Generates a closed GeoJSON Polygon ring [[lon, lat], ...] around a center coordinate.
        Note: GeoJSON requires [longitude, latitude] coordinate ordering!
        """
        coords: List[List[float]] = []
        R = 6371.0
        d = radius_km / R

        lat1 = math.radians(center_lat)
        lon1 = math.radians(center_lon)

        for i in range(num_points):
            bearing = (2 * math.pi * i) / num_points
            lat2 = math.asin(
                math.sin(lat1) * math.cos(d) +
                math.cos(lat1) * math.sin(d) * math.cos(bearing)
            )
            lon2 = lon1 + math.atan2(
                math.sin(bearing) * math.sin(d) * math.cos(lat1),
                math.cos(d) - math.sin(lat1) * math.sin(lat2)
            )
            # GeoJSON order: [longitude, latitude]
            coords.append([round(math.degrees(lon2), 4), round(math.degrees(lat2), 4)])

        # Close the ring
        coords.append(coords[0])
        return coords

    async def get_layer_geojson(
        self,
        layer_id: str,
        center_lat: float,
        center_lon: float,
        radius_km: float = 50.0
    ) -> GeoJSONFeatureCollection:
        """
        Retrieves real vector GeoJSON features for the specified layer.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        features: List[GeoJSONFeature] = []

        # ---------------------------------------------------------------------
        # Layer: SST (Sea Surface Temperature)
        # ---------------------------------------------------------------------
        if layer_id == "sst":
            marine_res = await self._marine_connector.fetch_marine(center_lat, center_lon)
            if marine_res.success and marine_res.data:
                current = marine_res.data.get("current", {})
                sst_val = current.get("sea_surface_temperature")
                time_str = current.get("time", now_iso)

                if sst_val is not None:
                    # GeoJSON Point [lon, lat]
                    feature = GeoJSONFeature(
                        geometry=GeoJSONGeometry(
                            type="Point",
                            coordinates=[round(center_lon, 4), round(center_lat, 4)]
                        ),
                        properties={
                            "layer": "sst",
                            "name": "Sea Surface Temperature",
                            "value": float(sst_val),
                            "unit": "°C",
                            "observation_time": time_str,
                            "provider": "Open-Meteo Marine / Copernicus CMEMS",
                            "quality": "VALID",
                            "is_cached": False
                        }
                    )
                    features.append(feature)

            return GeoJSONFeatureCollection(
                features=features,
                metadata={
                    "layer_id": "sst",
                    "retrieval_timestamp": now_iso,
                    "provider": "Open-Meteo Marine / Copernicus Marine Service",
                    "unit": "°C",
                    "coverage_radius_km": radius_km,
                    "feature_count": len(features)
                }
            )

        # ---------------------------------------------------------------------
        # Layer: Waves (Wave Height, Swell & Period)
        # ---------------------------------------------------------------------
        elif layer_id == "waves":
            marine_res = await self._marine_connector.fetch_marine(center_lat, center_lon)
            if marine_res.success and marine_res.data:
                current = marine_res.data.get("current", {})
                wave_h = current.get("wave_height")
                swell_h = current.get("swell_wave_height")
                period = current.get("wave_period")
                time_str = current.get("time", now_iso)

                if wave_h is not None:
                    feature = GeoJSONFeature(
                        geometry=GeoJSONGeometry(
                            type="Point",
                            coordinates=[round(center_lon, 4), round(center_lat, 4)]
                        ),
                        properties={
                            "layer": "waves",
                            "name": "Wave Observations",
                            "wave_height": float(wave_h),
                            "swell_wave_height": float(swell_h) if swell_h is not None else None,
                            "wave_period": float(period) if period is not None else None,
                            "unit": "m",
                            "observation_time": time_str,
                            "provider": "Open-Meteo Marine / ECMWF WAM",
                            "quality": "VALID"
                        }
                    )
                    features.append(feature)

            return GeoJSONFeatureCollection(
                features=features,
                metadata={
                    "layer_id": "waves",
                    "retrieval_timestamp": now_iso,
                    "provider": "Open-Meteo Marine / ECMWF",
                    "unit": "m",
                    "feature_count": len(features)
                }
            )

        # ---------------------------------------------------------------------
        # Layer: Wind (10m Wind Speed, Gusts, Direction)
        # ---------------------------------------------------------------------
        elif layer_id == "wind":
            weather_res = await self._weather_connector.fetch_weather(center_lat, center_lon)
            if weather_res.success and weather_res.data:
                current = weather_res.data.get("current", {})
                wind_spd = current.get("wind_speed_10m")
                wind_gst = current.get("wind_gusts_10m")
                wind_dir = current.get("wind_direction_10m")
                time_str = current.get("time", now_iso)

                if wind_spd is not None:
                    feature = GeoJSONFeature(
                        geometry=GeoJSONGeometry(
                            type="Point",
                            coordinates=[round(center_lon, 4), round(center_lat, 4)]
                        ),
                        properties={
                            "layer": "wind",
                            "name": "Surface Wind Field",
                            "wind_speed": float(wind_spd),
                            "wind_gusts": float(wind_gst) if wind_gst is not None else None,
                            "wind_direction": float(wind_dir) if wind_dir is not None else None,
                            "unit": "km/h",
                            "observation_time": time_str,
                            "provider": "Open-Meteo Weather / ECMWF IFS",
                            "quality": "VALID"
                        }
                    )
                    features.append(feature)

            return GeoJSONFeatureCollection(
                features=features,
                metadata={
                    "layer_id": "wind",
                    "retrieval_timestamp": now_iso,
                    "provider": "Open-Meteo Weather / ECMWF",
                    "unit": "km/h",
                    "feature_count": len(features)
                }
            )

        # ---------------------------------------------------------------------
        # Layer: Official Marine & Cyclone Warnings
        # ---------------------------------------------------------------------
        elif layer_id == "warnings":
            warnings = self._warning_provider.get_warnings_for_location(center_lat, center_lon, include_expired=False)
            for w in warnings:
                if w.affected_coordinates:
                    w_lat = w.affected_coordinates.get("latitude")
                    w_lon = w.affected_coordinates.get("longitude")
                    if w_lat is not None and w_lon is not None:
                        # Create circular coverage polygon
                        ring = self._create_circle_polygon(w_lat, w_lon, w.coverage_radius_km)
                        feature = GeoJSONFeature(
                            geometry=GeoJSONGeometry(
                                type="Polygon",
                                coordinates=[ring]
                            ),
                            properties={
                                "layer": "warnings",
                                "warning_id": w.warning_id,
                                "warning_type": w.warning_type,
                                "source": w.source,
                                "source_severity": w.source_severity,
                                "normalized_severity": w.normalized_severity,
                                "bulletin_text": w.bulletin_text,
                                "valid_from": w.valid_from,
                                "valid_until": w.valid_until,
                                "radius_km": w.coverage_radius_km
                            }
                        )
                        features.append(feature)

            return GeoJSONFeatureCollection(
                features=features,
                metadata={
                    "layer_id": "warnings",
                    "retrieval_timestamp": now_iso,
                    "provider": "Official Warning Provider (IMD/INCOIS)",
                    "feature_count": len(features)
                }
            )

        # ---------------------------------------------------------------------
        # Layer: PFZ (Potential Fishing Zones) - Honest Unauthenticated Handling
        # ---------------------------------------------------------------------
        elif layer_id == "pfz":
            # Per strict project rules: Never fabricate live PFZ coordinates or fish abundance!
            return GeoJSONFeatureCollection(
                features=[],
                metadata={
                    "layer_id": "pfz",
                    "retrieval_timestamp": now_iso,
                    "provider": "INCOIS (Indian National Centre for Ocean Information Services)",
                    "status": "REQUIRES_AUTH",
                    "message": "Official INCOIS PFZ dataset requires institutional API key (INCOIS_API_KEY). Zero simulated coordinates displayed.",
                    "feature_count": 0
                }
            )

        return GeoJSONFeatureCollection(features=[], metadata={"layer_id": layer_id, "error": "Unknown layer"})

    def search_coastal_locations(
        self,
        query: str,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None
    ) -> List[CoastalLocationItem]:
        """
        Searches supported coastal ports and maritime hubs.
        Supports query text matching and coordinate parsing.
        """
        results: List[CoastalLocationItem] = []
        q = (query or "").lower().strip()

        # Check if query is comma-separated coordinates (e.g. "15.35, 73.80")
        if "," in q:
            parts = [p.strip() for p in q.split(",")]
            if len(parts) == 2:
                try:
                    c_lat = float(parts[0])
                    c_lon = float(parts[1])
                    if (-90.0 <= c_lat <= 90.0) and (-180.0 <= c_lon <= 180.0):
                        results.append(
                            CoastalLocationItem(
                                id="coord_custom",
                                name=f"Custom Coordinates ({c_lat:.4f}, {c_lon:.4f})",
                                latitude=c_lat,
                                longitude=c_lon,
                                distance_from_query_km=0.0
                            )
                        )
                        return results
                except ValueError:
                    pass

        # Search coastal reference catalog
        for hub_id, hub in KNOWN_COASTAL_LOCATIONS.items():
            if not q or q in hub_id or q in hub["name"].lower():
                dist = None
                if user_lat is not None and user_lon is not None:
                    # Haversine distance
                    dlat = math.radians(hub["lat"] - user_lat)
                    dlon = math.radians(hub["lon"] - user_lon)
                    a = math.sin(dlat / 2)**2 + math.cos(math.radians(user_lat)) * math.cos(math.radians(hub["lat"])) * math.sin(dlon / 2)**2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                    dist = round(6371.0 * c, 1)

                results.append(
                    CoastalLocationItem(
                        id=hub_id,
                        name=hub["name"],
                        latitude=hub["lat"],
                        longitude=hub["lon"],
                        distance_from_query_km=dist
                    )
                )

        # Sort by distance if user location provided
        if user_lat is not None and user_lon is not None:
            results.sort(key=lambda item: item.distance_from_query_km if item.distance_from_query_km is not None else float("inf"))

        return results
