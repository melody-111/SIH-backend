from fastapi import APIRouter, Query, HTTPException, Body
from typing import Optional, List, Dict, Any
from app.schemas.gis_map import (
    MapLayerInfo,
    GeoJSONFeatureCollection,
    RouteAnalysisRequest,
    RouteAnalysisResponse,
    CoastalLocationItem
)
from app.services.marine_gis_service import MarineGISService
from app.services.route_intelligence_service import RouteIntelligenceService

router = APIRouter(prefix="/map", tags=["map"])

gis_service = MarineGISService()
route_service = RouteIntelligenceService()

@router.get("/layers", response_model=List[MapLayerInfo])
async def get_map_layers():
    """
    Returns the catalog of available modular marine GIS layers,
    including provider attribution, units, and capability status.
    """
    return gis_service.get_layer_catalog()

@router.get("/layer-data", response_model=GeoJSONFeatureCollection)
async def get_layer_geojson_data(
    layer_id: str = Query(..., description="Layer identifier: sst, waves, wind, warnings, pfz"),
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Center latitude"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Center longitude"),
    radius_km: float = Query(50.0, ge=1.0, le=500.0, description="Coverage radius in km")
):
    """
    Returns standards-compliant GeoJSON FeatureCollection for the requested marine layer.
    Strictly preserves WGS84 [longitude, latitude] coordinate ordering.
    """
    try:
        return await gis_service.get_layer_geojson(
            layer_id=layer_id.lower().strip(),
            center_lat=latitude,
            center_lon=longitude,
            radius_km=radius_km
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed generating layer GeoJSON: {str(e)}")

@router.post("/route", response_model=RouteAnalysisResponse)
async def analyze_maritime_route(request: RouteAnalysisRequest):
    """
    Calculates geodesic maritime route corridor waypoints, estimated transit time,
    and overlays live wave/wind/SST conditions and active official warnings along the route.
    """
    try:
        return await route_service.analyze_route(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route analysis failed: {str(e)}")

@router.get("/search-locations", response_model=List[CoastalLocationItem])
async def search_coastal_locations(
    query: str = Query("", description="Search term or comma-separated latitude, longitude"),
    latitude: Optional[float] = Query(None, ge=-90.0, le=90.0),
    longitude: Optional[float] = Query(None, ge=-180.0, le=180.0)
):
    """
    Searches supported Indian maritime ports and coastal locations.
    Calculates distance from user coordinates when provided.
    """
    return gis_service.search_coastal_locations(
        query=query,
        user_lat=latitude,
        user_lon=longitude
    )
