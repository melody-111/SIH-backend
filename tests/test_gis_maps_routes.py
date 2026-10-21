import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.marine_gis_service import MarineGISService
from app.services.route_intelligence_service import RouteIntelligenceService
from app.schemas.gis_map import RouteAnalysisRequest

client = TestClient(app)

# -----------------------------------------------------------------------------
# 1. Coordinate Validation & Bounding Box Checks
# -----------------------------------------------------------------------------
def test_coordinate_validation_and_bounds():
    gis = MarineGISService()

    # Valid bounding box (Goa coastal zone: 1° x 1°)
    valid, err = gis.validate_bounds(min_lat=15.0, min_lon=73.0, max_lat=16.0, max_lon=74.0)
    assert valid is True
    assert err is None

    # Invalid latitude
    valid_bad_lat, err_lat = gis.validate_bounds(min_lat=-95.0, min_lon=73.0, max_lat=16.0, max_lon=74.0)
    assert valid_bad_lat is False
    assert "Latitude values must be between -90.0 and 90.0" in err_lat

    # Excessively large request (>15x15 deg)
    valid_huge, err_huge = gis.validate_bounds(min_lat=0.0, min_lon=60.0, max_lat=20.0, max_lon=80.0)
    assert valid_huge is False
    assert "exceeds maximum request limit" in err_huge

# -----------------------------------------------------------------------------
# 2. GeoJSON Geometry & Coordinate Order ([lon, lat])
# -----------------------------------------------------------------------------
def test_geojson_coordinate_order_and_polygon_rings():
    gis = MarineGISService()
    circle = gis._create_circle_polygon(center_lat=15.3493, center_lon=73.8000, radius_km=50.0)
    
    # Must have multiple coordinates
    assert len(circle) >= 20
    
    # First and last coordinate must be identical to close the ring
    assert circle[0] == circle[-1]

    # Coordinate ordering must be [longitude, latitude]
    for pt in circle:
        lon, lat = pt
        assert 70.0 <= lon <= 77.0  # Longitude in Arabian Sea near Goa
        assert 12.0 <= lat <= 18.0  # Latitude near Goa

# -----------------------------------------------------------------------------
# 3. Layer Catalog & Metadata
# -----------------------------------------------------------------------------
def test_map_layers_catalog():
    res = client.get("/api/map/layers")
    assert res.status_code == 200
    layers = res.json()
    layer_ids = [l["layer_id"] for l in layers]
    assert "sst" in layer_ids
    assert "waves" in layer_ids
    assert "wind" in layer_ids
    assert "warnings" in layer_ids
    assert "pfz" in layer_ids

    # Check SST layer metadata
    sst = next(l for l in layers if l["layer_id"] == "sst")
    assert sst["units"] == "°C"
    assert "Copernicus" in sst["provider"]
    assert sst["status"] == "ACTIVE"

# -----------------------------------------------------------------------------
# 4. Live SST & Wave GeoJSON Layers
# -----------------------------------------------------------------------------
def test_live_sst_layer_geojson():
    res = client.get("/api/map/layer-data", params={"layer_id": "sst", "latitude": 15.3493, "longitude": 73.8000})
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert data["metadata"]["layer_id"] == "sst"
    assert data["metadata"]["unit"] == "°C"
    
    if len(data["features"]) > 0:
        feat = data["features"][0]
        assert feat["geometry"]["type"] == "Point"
        lon, lat = feat["geometry"]["coordinates"]
        assert 73.0 <= lon <= 75.0
        assert 14.0 <= lat <= 16.0
        assert feat["properties"]["value"] > 20.0  # Warm tropical Arabian sea SST

def test_live_waves_layer_geojson():
    res = client.get("/api/map/layer-data", params={"layer_id": "waves", "latitude": 15.3493, "longitude": 73.8000})
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert data["metadata"]["layer_id"] == "waves"
    if len(data["features"]) > 0:
        feat = data["features"][0]
        assert feat["properties"]["wave_height"] >= 0.0

# -----------------------------------------------------------------------------
# 5. Official Warnings Layer (Polygon Overlays)
# -----------------------------------------------------------------------------
def test_official_warnings_layer_geojson():
    res = client.get("/api/map/layer-data", params={"layer_id": "warnings", "latitude": 15.3493, "longitude": 73.8000})
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) >= 1  # Active Goa baseline warning bulletin
    feat = data["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    assert "bulletin_text" in feat["properties"]
    assert "source_severity" in feat["properties"]

# -----------------------------------------------------------------------------
# 6. PFZ Safety Rule (Zero Fabricated Coordinates)
# -----------------------------------------------------------------------------
def test_pfz_layer_unauthenticated_safety():
    res = client.get("/api/map/layer-data", params={"layer_id": "pfz", "latitude": 15.3493, "longitude": 73.8000})
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    # Must NOT fabricate points when unauthenticated
    assert len(data["features"]) == 0
    assert data["metadata"]["status"] == "REQUIRES_AUTH"
    assert "INCOIS_API_KEY" in data["metadata"]["message"]

# -----------------------------------------------------------------------------
# 7. Route Intelligence: Geodesic Corridor, Distance & Waypoints
# -----------------------------------------------------------------------------
@pytest.mark.anyio
async def test_route_analysis_and_distance_calculation():
    route_svc = RouteIntelligenceService()
    req = RouteAnalysisRequest(
        origin_latitude=15.3493,    # Goa / Mormugao Port
        origin_longitude=73.8000,
        destination_latitude=18.9220, # Mumbai Port
        destination_longitude=72.8347,
        vessel_speed_knots=10.0
    )
    result = await route_svc.analyze_route(req)

    # Goa to Mumbai is approx 410-430 km
    assert 380.0 <= result.total_distance_km <= 450.0
    assert 200.0 <= result.total_distance_nautical_miles <= 250.0
    
    # Transit time: ~22-25 hours (~1300-1500 minutes) at 10 knots
    assert 1200 <= result.estimated_transit_minutes <= 1600
    
    # LineString GeoJSON coordinates
    assert result.route_geometry.type == "LineString"
    assert len(result.route_geometry.coordinates) >= 4

    # Navigational limitation disclaimer
    assert "navigational_limitation" in result.model_dump()
    assert "nautical navigational charts" in result.navigational_limitation

# -----------------------------------------------------------------------------
# 8. Route Proximity to Active Warning Zones
# -----------------------------------------------------------------------------
@pytest.mark.anyio
async def test_route_warning_proximity():
    route_svc = RouteIntelligenceService()
    # Route starting directly inside active Goa warning area (radius 120 km)
    req = RouteAnalysisRequest(
        origin_latitude=15.3493,
        origin_longitude=73.8000,
        destination_latitude=15.7000,
        destination_longitude=73.6000,
        vessel_speed_knots=10.0
    )
    result = await route_svc.analyze_route(req)
    assert len(result.warnings_along_route) >= 1
    w_info = result.warnings_along_route[0]
    assert w_info["intersects_route"] is True
    # If intersecting warning, departure is flagged not safe
    assert result.is_safe_to_depart is False
    assert "UNSAFE" in result.departure_recommendation

# -----------------------------------------------------------------------------
# 9. Coastal Search & Coordinate Parsing
# -----------------------------------------------------------------------------
def test_coastal_search_and_coordinates():
    # Search by port name
    res_name = client.get("/api/map/search-locations", params={"query": "mumbai", "latitude": 15.3493, "longitude": 73.8000})
    assert res_name.status_code == 200
    items = res_name.json()
    assert len(items) >= 1
    assert "Mumbai" in items[0]["name"]
    assert items[0]["distance_from_query_km"] > 300.0

    # Search by coordinates
    res_coord = client.get("/api/map/search-locations", params={"query": "15.40, 73.80"})
    assert res_coord.status_code == 200
    items_coord = res_coord.json()
    assert len(items_coord) == 1
    assert items_coord[0]["latitude"] == 15.40
    assert items_coord[0]["longitude"] == 73.80

# -----------------------------------------------------------------------------
# 10. Multi-Agent Orchestrator Integration for Route Queries
# -----------------------------------------------------------------------------
def test_orchestrator_route_query():
    res = client.post(
        "/api/chat",
        json={"message": "Calculate nautical route to target coordinates", "latitude": 15.3493, "longitude": 73.8000}
    )
    assert res.status_code == 200
    data = res.json()
    assert "Route" in data["answer"] or "corridor" in data["answer"]
    assert len(data["map_actions"]) >= 1
