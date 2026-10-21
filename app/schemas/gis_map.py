from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class GeoJSONGeometry(BaseModel):
    type: str  # "Point", "LineString", "Polygon"
    coordinates: Any  # [lon, lat] for Point, [[lon, lat], ...] for LineString, etc.

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: Dict[str, Any] = {}

class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature] = []
    metadata: Dict[str, Any] = {}

class MapLayerInfo(BaseModel):
    layer_id: str
    name: str
    description: str
    provider: str
    units: Optional[str] = None
    data_type: str  # "OBSERVATION", "FORECAST", "DERIVED_PRODUCT", "OFFICIAL_ADVISORY"
    is_live_supported: bool = True
    is_offline_supported: bool = True
    status: str = "ACTIVE"  # "ACTIVE", "REQUIRES_AUTH", "UNAVAILABLE"

from pydantic import BaseModel, Field, model_validator

class RouteAnalysisRequest(BaseModel):
    origin_latitude: float = Field(..., ge=-90.0, le=90.0)
    origin_longitude: float = Field(..., ge=-180.0, le=180.0)
    destination_latitude: float = Field(..., ge=-90.0, le=90.0)
    destination_longitude: float = Field(..., ge=-180.0, le=180.0)
    vessel_speed_knots: float = Field(10.0, ge=1.0, le=50.0)

    @model_validator(mode="before")
    @classmethod
    def handle_coordinate_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            if "start_latitude" in mapped and "origin_latitude" not in mapped:
                mapped["origin_latitude"] = mapped["start_latitude"]
            if "start_longitude" in mapped and "origin_longitude" not in mapped:
                mapped["origin_longitude"] = mapped["start_longitude"]
            if "end_latitude" in mapped and "destination_latitude" not in mapped:
                mapped["destination_latitude"] = mapped["end_latitude"]
            if "end_longitude" in mapped and "destination_longitude" not in mapped:
                mapped["destination_longitude"] = mapped["end_longitude"]
            return mapped
        return data

class RouteWaypoint(BaseModel):
    waypoint_index: int
    latitude: float
    longitude: float
    distance_from_origin_km: float
    bearing_degrees: float
    wave_height_m: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    sst_celsius: Optional[float] = None
    active_hazard_warning: Optional[str] = None

class RouteAnalysisResponse(BaseModel):
    origin: Dict[str, float]
    destination: Dict[str, float]
    total_distance_km: float
    total_distance_nautical_miles: float
    estimated_transit_minutes: int
    cruising_speed_knots: float
    route_geometry: GeoJSONGeometry
    waypoints: List[RouteWaypoint] = []
    warnings_along_route: List[Dict[str, Any]] = []
    max_wave_height_m: Optional[float] = None
    max_wind_speed_kmh: Optional[float] = None
    is_safe_to_depart: bool
    departure_recommendation: str
    data_sources: List[str]
    retrieval_timestamp: str
    navigational_limitation: str

class CoastalLocationItem(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    distance_from_query_km: Optional[float] = None
