import pytest
import datetime
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.connectors.base_connector import BaseConnector, ConnectorResult
from app.connectors.open_meteo_weather import OpenMeteoWeatherConnector
from app.connectors.open_meteo_marine import OpenMeteoMarineConnector
from app.connectors.nasa_earthdata import NASAEarthdataConnector
from app.connectors.incois_connector import INCOISConnector
from app.connectors.imd_connector import IMDWarningsConnector
from app.services.marine_data_service import MarineDataService, KNOWN_COASTAL_LOCATIONS
from app.schemas.marine_normalized import (
    MarineQueryRequest,
    MarineTask,
    DataType,
    QualityStatus
)

client = TestClient(app)

# -------------------------------------------------------------
# 1. Valid API Responses Parsing & Normalization
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_weather_connector_normalization_valid():
    connector = OpenMeteoWeatherConnector()
    mock_data = {
        "current": {
            "time": "2026-10-10T12:00",
            "temperature_2m": 29.4,
            "relative_humidity_2m": 80,
            "wind_speed_10m": 18.5,
            "wind_direction_10m": 240,
            "wind_gusts_10m": 25.0,
            "precipitation": 0.0,
            "weather_code": 1,
            "surface_pressure": 1012.3
        }
    }
    raw_res = ConnectorResult(success=True, data=mock_data, source="Open-Meteo Weather")
    records, warnings = connector.normalize(raw_res, 15.30, 73.70, "Goa")

    # Verify normalization
    assert len(records) > 0
    temp_rec = next(r for r in records if r.variable_name == "air_temperature")
    assert temp_rec.value == 29.4
    assert temp_rec.unit == "°C"
    assert temp_rec.data_type == DataType.OBSERVATION
    assert temp_rec.quality_status == QualityStatus.VALID
    assert temp_rec.is_missing is False

    wind_rec = next(r for r in records if r.variable_name == "wind_speed")
    assert wind_rec.value == 18.5
    assert wind_rec.unit == "km/h"

    # Low wind shouldn't trigger gale warnings
    assert len(warnings) == 0

# -------------------------------------------------------------
# 2. Missing & Malformed Fields Handling (No 0 Substitution)
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_marine_connector_missing_fields_no_zero_fabrication():
    connector = OpenMeteoMarineConnector()
    # Missing wave_height (e.g. land grid cell)
    mock_data = {
        "current": {
            "time": "2026-10-10T12:00",
            "wave_height": None,
            "wave_direction": None,
            "wave_period": None,
            "ocean_current_velocity": None,
            "sea_surface_temperature": None
        }
    }
    raw_res = ConnectorResult(success=True, data=mock_data, source="Open-Meteo Marine")
    records, warnings = connector.normalize(raw_res, 15.29, 74.12, "Goa Mainland")

    wave_rec = next(r for r in records if r.variable_name == "significant_wave_height")
    # Must NOT fabricate 0.0
    assert wave_rec.value is None
    assert wave_rec.is_missing is True
    assert wave_rec.quality_status == QualityStatus.UNAVAILABLE

# -------------------------------------------------------------
# 3. Provider Timeouts Handling
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_base_connector_timeout_handling():
    connector = BaseConnector(name="TestTimeout", base_url="https://mock.service/api", default_timeout=0.1)
    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Timed out")):
        res = await connector.get_with_retry("https://mock.service/api")
        assert res.success is False
        assert "timed out" in res.error.lower()

# -------------------------------------------------------------
# 4. HTTP Errors & Rate Limiting (429) Handling
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_base_connector_rate_limiting():
    connector = BaseConnector(name="TestRateLimit", base_url="https://mock.service/api")
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Too Many Requests"

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        res = await connector.get_with_retry("https://mock.service/api")
        assert res.success is False
        assert res.is_rate_limited is True
        assert res.status_code == 429

# -------------------------------------------------------------
# 5. Missing Credentials Handling (Honest Unconfigured Status)
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_incois_and_imd_missing_credentials():
    with patch("app.core.config.settings.INCOIS_API_KEY", ""):
        with patch("app.core.config.settings.ENABLE_SIMULATION_FALLBACK", False):
            incois = INCOISConnector()
            res = await incois.fetch_advisory(15.30, 73.70)
            assert res.success is False
            assert "requires" in res.error.lower()
            records = incois.normalize(res, 15.30, 73.70)
            assert records[0].quality_status == QualityStatus.OFFLINE
            assert records[0].is_missing is True

    with patch("app.core.config.settings.IMD_API_KEY", ""):
        with patch("app.core.config.settings.ENABLE_SIMULATION_FALLBACK", False):
            imd = IMDWarningsConnector()
            res = await imd.fetch_warnings(15.30, 73.70)
            assert res.success is False
            warns = imd.normalize(res, 15.30, 73.70)
            assert any(w.warning_type == "OFFICIAL_FEED_UNAVAILABLE" for w in warns)

# -------------------------------------------------------------
# 6. Unit & Timestamp Normalization
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_unit_and_timestamp_normalization():
    connector = OpenMeteoMarineConnector()
    mock_data = {
        "current": {
            "time": "2026-10-10T14:30",
            "wave_height": 1.25,
            "wave_direction": 180,
            "wave_period": 9.5,
            "ocean_current_velocity": 1.8,
            "sea_surface_temperature": 29.8
        }
    }
    raw_res = ConnectorResult(success=True, data=mock_data, source="Open-Meteo Marine")
    records, _ = connector.normalize(raw_res, 15.30, 73.70)

    sst_rec = next(r for r in records if r.variable_name == "sea_surface_temperature")
    assert sst_rec.unit == "°C"
    assert sst_rec.value == 29.8
    assert "Z" in sst_rec.timestamp

# -------------------------------------------------------------
# 7. Geographic & Date-Range Resolution
# -------------------------------------------------------------
def test_location_resolution_preserves_explicit_hub():
    service = MarineDataService()
    # User is in Goa (15.29, 74.12) but specifically asks about "Mumbai"
    req = MarineQueryRequest(
        latitude=15.2993,
        longitude=74.1240,
        question="Mumbai mein waves kaisi hain?"
    )
    lat, lon, name, is_ambiguous = service.resolve_location(req)
    assert lat == KNOWN_COASTAL_LOCATIONS["mumbai"]["lat"]
    assert lon == KNOWN_COASTAL_LOCATIONS["mumbai"]["lon"]
    assert "Mumbai" in name
    assert is_ambiguous is True  # Ambiguous because user coordinates differ from asked location!

# -------------------------------------------------------------
# 8. Partial Failure Resilience (One Provider Unavailable)
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_partial_failure_service_resilience():
    service = MarineDataService()
    # Mock weather succeeding, but marine throwing error
    weather_ok = ConnectorResult(
        success=True,
        data={"current": {"temperature_2m": 30.0, "wind_speed_10m": 12.0, "weather_code": 0}},
        source="Open-Meteo Weather"
    )

    with patch.object(service.weather_connector, "fetch_weather", return_value=weather_ok):
        with patch.object(service.marine_connector, "fetch_marine", side_effect=httpx.ConnectError("Marine offline")):
            req = MarineQueryRequest(latitude=15.30, longitude=73.70)
            snapshot = await service.get_marine_snapshot(req)

            # Weather records are present despite marine outage
            assert len(snapshot.weather_records) > 0
            assert "marine" in snapshot.unavailable_sources
            assert snapshot.source_statuses["marine"].status == "ERROR"

# -------------------------------------------------------------
# 9. Correct Source Attribution
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_correct_source_attribution():
    service = MarineDataService()
    # Mock clean results
    weather_ok = ConnectorResult(
        success=True,
        data={"current": {"temperature_2m": 28.0, "wind_speed_10m": 15.0}},
        source="Open-Meteo Weather API"
    )
    marine_ok = ConnectorResult(
        success=True,
        data={"current": {"wave_height": 0.8, "sea_surface_temperature": 30.5}},
        source="Open-Meteo Marine (Copernicus/ECMWF)"
    )

    with patch.object(service.weather_connector, "fetch_weather", return_value=weather_ok):
        with patch.object(service.marine_connector, "fetch_marine", return_value=marine_ok):
            req = MarineQueryRequest(latitude=15.30, longitude=73.70)
            snapshot = await service.get_marine_snapshot(req)

            temp = next(r for r in snapshot.weather_records if r.variable_name == "air_temperature")
            wave = next(r for r in snapshot.ocean_records if r.variable_name == "significant_wave_height")

            assert "Open-Meteo Weather" in temp.source_provider
            assert "Open-Meteo Marine" in wave.source_provider

# -------------------------------------------------------------
# 10. FastAPI Route Integration Tests
# -------------------------------------------------------------
def test_marine_conditions_endpoint():
    resp = client.get("/api/marine/conditions?latitude=15.30&longitude=73.70")
    assert resp.status_code == 200
    data = resp.json()
    assert "weather_records" in data
    assert "ocean_records" in data
    assert "evidence_items" in data
    assert data["latitude"] == 15.30
    assert data["longitude"] == 73.70

def test_marine_warnings_endpoint():
    resp = client.get("/api/marine/warnings?latitude=15.30&longitude=73.70")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

def test_marine_query_endpoint():
    payload = {
        "question": "Kal fishing ke liye Goa coast par conditions check karo",
        "latitude": 15.30,
        "longitude": 73.70,
        "intended_task": "FISHING"
    }
    resp = client.post("/api/marine/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["evidence_items"]) > 0
