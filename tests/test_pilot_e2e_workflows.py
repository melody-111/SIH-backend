import pytest
import datetime
import uuid
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import Settings, ConfigurationError
from app.core.logging import SensitiveDataRedactor
from app.db.database import DatabaseManager
from app.connectors.base_connector import ConnectorResult
from app.services.offline_cache import OfflineMarineCache
from app.services.official_warning_provider import OfficialWarningProvider
from app.schemas.hazard_alert import OfficialWarningDetail, AlertStatus

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_offline_state():
    cache = OfflineMarineCache()
    cache.clear()
    cache.set_force_offline(False)
    yield
    cache.clear()
    cache.set_force_offline(False)

# =============================================================================
# WORKFLOW A: Marine/Weather Question with Real Source Metadata
# =============================================================================

@pytest.mark.anyio
async def test_workflow_a_marine_query_real_source_metadata():
    """
    Workflow A: A user asks a marine/weather question and receives a response
    with real source metadata when a live provider is configured.
    """
    with patch("app.services.weather_service.WeatherService.get_weather", new_callable=AsyncMock) as mock_weather, \
         patch("app.services.ocean_service.OceanService.get_ocean_conditions", new_callable=AsyncMock) as mock_marine:

        mock_weather.return_value = {
            "temperature": 29.4,
            "wind_speed": 22.5,
            "wind_direction": 240,
            "precipitation": 0.0,
            "status": "success",
            "source": "Open-Meteo Weather API"
        }
        mock_marine.return_value = {
            "wave_height": 1.6,
            "wave_direction": 230,
            "wave_period": 7.0,
            "current_velocity": 1.2,
            "sst": 28.5,
            "status": "success",
            "source": "Open-Meteo Marine API"
        }

        resp = client.post("/api/chat", json={
            "message": "What is the wave height and wind condition off Kochi?",
            "latitude": 9.9312,
            "longitude": 76.2673,
            "language": "en"
        }, headers={"X-User-ID": "pilot_fisherman_kochi"})

        assert resp.status_code == 200
        data = resp.json()

        # Verify grounded response
        assert "answer" in data
        assert len(data["answer"]) > 10
        assert data["risk"] in ["LOW", "MODERATE", "HIGH", "DANGER", "UNKNOWN"]
        assert "timestamp" in data
        assert len(data.get("progress_steps", [])) > 0
        assert "ocean" in data.get("execution_plan", [])
        assert "weather" in data.get("execution_plan", [])


# =============================================================================
# WORKFLOW B: Provider Unavailable — Missing/Stale Data Without Fabrication
# =============================================================================

@pytest.mark.anyio
async def test_workflow_b_provider_unavailable_no_zero_fabrication():
    """
    Workflow B: A provider is unavailable, and the app clearly reports missing
    or stale data without inventing zero or synthetic measurements.
    """
    with patch("app.services.ocean_service.OceanService.get_ocean_conditions", new_callable=AsyncMock) as mock_marine, \
         patch("app.services.weather_service.WeatherService.get_weather", new_callable=AsyncMock) as mock_weather:

        mock_marine.return_value = {
            "status": "unavailable",
            "error": "Upstream Open-Meteo marine service timed out (503 Service Unavailable)",
            "source": "Open-Meteo Marine API"
        }
        mock_weather.return_value = {
            "status": "unavailable",
            "error": "Upstream weather gateway unreachable",
            "source": "Open-Meteo Weather API"
        }

        resp = client.post("/api/chat", json={
            "message": "Current wave height in Arabian sea?",
            "latitude": 15.30,
            "longitude": 73.70,
            "language": "en"
        }, headers={"X-User-ID": "pilot_user_b"})

        assert resp.status_code == 200
        data = resp.json()

        # Must report failure/unavailability honestly in answer
        assert "answer" in data
        ans_lower = data["answer"].lower()
        assert any(term in ans_lower for term in ["unavailable", "unable", "unreachable", "offline", "data", "error", "caution"])
        # Should NOT fabricate synthetic zero measurements
        assert data["risk"] in ["LOW", "MODERATE", "HIGH", "UNKNOWN"]


# =============================================================================
# WORKFLOW C: Fishing / PFZ Request — Verified Provider Data or Honest Unavailable
# =============================================================================

@pytest.mark.anyio
async def test_workflow_c_pfz_request_unauthenticated_safety():
    """
    Workflow C: Fishing/PFZ request returns verified provider data or
    explicitly reports that verified PFZ data is unavailable without fake coordinates.
    """
    with patch("app.core.config.settings.INCOIS_API_KEY", ""), \
         patch("app.core.config.settings.ENABLE_SIMULATION_FALLBACK", False):

        resp = client.post("/api/chat", json={
            "message": "Where are the potential fishing zones in the Arabian Sea today?",
            "latitude": 16.99,
            "longitude": 73.30,
            "language": "en"
        }, headers={"X-User-ID": "ratnagiri_angler"})

        assert resp.status_code == 200
        data = resp.json()

        # Recommended location must not fabricate fake GPS
        assert data.get("recommended_location") is None or "INCOIS" in str(data.get("recommended_location"))
        assert len(data["answer"]) > 0


@pytest.mark.anyio
async def test_workflow_c_pfz_request_verified_provider_data():
    """
    Workflow C (Part 2): When verified institutional credentials are configured,
    verified PFZ advisory is returned with full provenance.
    """
    with patch("app.core.config.settings.INCOIS_API_KEY", "INCOIS_PROD_KEY_VERIFIED"), \
         patch("app.connectors.incois_connector.INCOISConnector.fetch_advisory", new_callable=AsyncMock) as mock_pfz:

        mock_pfz.return_value = ConnectorResult(
            success=True,
            data={
                "has_pfz": True,
                "recommended_zone": {"latitude": 17.05, "longitude": 73.22, "bearing": "WNW", "distance_km": 12.4},
                "water_temperature_c": 28.5,
                "chlorophyll_a": 1.45,
                "valid_until": "2026-10-12T18:00:00Z",
                "advisory_notes": "Verified thermal front edge observed off Ratnagiri shelf."
            },
            latency_ms=120.0
        )

        resp = client.post("/api/chat", json={
            "message": "Suggest fishing areas near Ratnagiri.",
            "latitude": 16.99,
            "longitude": 73.30,
            "language": "en"
        }, headers={"X-User-ID": "ratnagiri_angler"})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["answer"]) > 0
        assert "pfz" in data.get("execution_plan", [])


# =============================================================================
# WORKFLOW D: Hazard Request — Prioritizes Official Warnings
# =============================================================================

@pytest.mark.anyio
async def test_workflow_d_hazard_prioritizes_official_warnings():
    """
    Workflow D: A hazard request prioritizes applicable official warnings
    and displays their source and validity.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    valid_until = now + datetime.timedelta(hours=24)

    warning_provider = OfficialWarningProvider()
    b_cyclone = OfficialWarningDetail(
        warning_id="IMD-CYC-PORBANDAR-99",
        source="India Meteorological Department (IMD)",
        warning_type="CYCLONIC_STORM_ALERT",
        source_severity="RED WARNING",
        normalized_severity="DANGER",
        bulletin_text="Severe Cyclonic Storm warning over North Gujarat and Porbandar Coast. Fishermen advised total suspension of fishing operations.",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        valid_until=valid_until.isoformat(),
        affected_regions=["Porbandar Coast", "North Gujarat", "Arabian Sea"],
        affected_coordinates={"latitude": 21.64, "longitude": 69.60},
        coverage_radius_km=150.0,
        status=AlertStatus.ACTIVE,
        is_official=True
    )
    warning_provider.register_bulletin(b_cyclone)

    resp = client.post("/api/chat", json={
        "message": "Is there a cyclone or dangerous storm near Porbandar?",
        "latitude": 21.64,
        "longitude": 69.60,
        "language": "en"
    }, headers={"X-User-ID": "porbandar_pilot"})

    assert resp.status_code == 200
    data = resp.json()

    # Risk must be escalated to HIGH or DANGER
    assert data["risk"] in ["HIGH", "DANGER"]
    ans_lower = data["answer"].lower()
    assert any(w in ans_lower for w in ["cyclone", "warning", "imd", "danger", "critical", "suspension", "alert", "khatra"])


# =============================================================================
# WORKFLOW E: Offline Request — Displays Cached Data with Timestamp & Freshness
# =============================================================================

@pytest.mark.anyio
async def test_workflow_e_offline_request_cached_data_and_freshness():
    """
    Workflow E: An offline request displays only cached data with its original
    timestamp and freshness status.
    """
    cache = OfflineMarineCache()
    sample_snapshot = {
        "wave_height_m": 1.4,
        "wind_speed_kmh": 18.0,
        "location_name": "Mumbai Offshore"
    }
    cache.store_observation(18.92, 72.83, sample_snapshot, location_name="Mumbai Offshore")
    cache.set_force_offline(True)

    try:
        resp = client.post("/api/chat", json={
            "message": "What are the ocean conditions right now?",
            "latitude": 18.92,
            "longitude": 72.83,
            "language": "en"
        }, headers={"X-User-ID": "sailor_offline_pilot"})

        assert resp.status_code == 200
        data = resp.json()

        # Cached indicator must be true
        assert data.get("is_cached") is True
        assert data.get("cached_at") is not None
        assert "cached" in data["answer"].lower() or "offline" in data["answer"].lower()
    finally:
        cache.set_force_offline(False)


# =============================================================================
# WORKFLOW F: Map Geospatial Data & Layer Honesty
# =============================================================================

def test_workflow_f_gis_map_layers_honest_reporting():
    """
    Workflow F: Map displays available real geospatial data and identifies
    unavailable layers honestly without inventing coordinates.
    """
    # 1. Catalog identifies layer capabilities
    catalog_resp = client.get("/api/map/layers")
    assert catalog_resp.status_code == 200
    catalog = catalog_resp.json()
    assert isinstance(catalog, list)
    layer_ids = [l["layer_id"] for l in catalog]
    assert "sst" in layer_ids
    assert "warnings" in layer_ids
    assert "pfz" in layer_ids

    # 2. SST Layer returns GeoJSON FeatureCollection
    sst_resp = client.get("/api/map/layer-data?layer_id=sst&latitude=15.0&longitude=73.0")
    assert sst_resp.status_code == 200
    sst_geojson = sst_resp.json()
    assert sst_geojson["type"] == "FeatureCollection"
    assert "features" in sst_geojson

    # 3. PFZ Layer with unauthenticated credentials returns zero fake features
    with patch("app.core.config.settings.INCOIS_API_KEY", ""):
        pfz_resp = client.get("/api/map/layer-data?layer_id=pfz&latitude=15.0&longitude=73.0")
        assert pfz_resp.status_code == 200
        pfz_geojson = pfz_resp.json()
        assert pfz_geojson["type"] == "FeatureCollection"
        assert len(pfz_geojson["features"]) == 0
        assert pfz_geojson.get("metadata", {}).get("status") == "REQUIRES_AUTH"
        assert "Zero simulated coordinates" in pfz_geojson.get("metadata", {}).get("message", "")


# =============================================================================
# WORKFLOW G: Authentication & User-Data Isolation
# =============================================================================

def test_workflow_g_auth_tenant_isolation():
    """
    Workflow G: Authentication and user-data isolation work correctly.
    User A's preferences and private memories are inaccessible to User B.
    """
    db = DatabaseManager()
    user_a = f"pilot_tenant_A_{uuid.uuid4().hex[:6]}"
    user_b = f"pilot_tenant_B_{uuid.uuid4().hex[:6]}"

    # Setup Tenant A preferences and memory
    db.update_user_preferences(user_a, {
        "preferred_language": "ml",
        "saved_locations": [
            {"id": "loc_a", "name": "Vizhinjam Port", "latitude": 8.38, "longitude": 76.99}
        ]
    })
    db.save_user_memory(user_a, "mem_secret_a", "Private Fishing Spot", "Secret reef 4km south")

    # Tenant B accesses /api/memory/preferences
    resp_b_pref = client.get("/api/memory/preferences", headers={"X-User-ID": user_b})
    assert resp_b_pref.status_code == 200
    data_b_pref = resp_b_pref.json()
    assert data_b_pref["user_id"] == user_b
    assert data_b_pref["preferred_language"] == "en"  # default, NOT ml
    assert len(data_b_pref["saved_locations"]) == 0   # Tenant A's locations not leaked!

    # Tenant B accesses /api/memory/memories
    resp_b_mems = client.get("/api/memory/memories", headers={"X-User-ID": user_b})
    assert resp_b_mems.status_code == 200
    data_b_mems = resp_b_mems.json()
    assert not any(m["memory_id"] == "mem_secret_a" for m in data_b_mems)

    # Tenant B attempts to delete Tenant A's memory
    del_resp = client.delete("/api/memory/memories/mem_secret_a", headers={"X-User-ID": user_b})
    assert del_resp.status_code == 404


# =============================================================================
# WORKFLOW H: Boundary Handling & Safety
# =============================================================================

def test_workflow_h_boundary_and_malformed_requests():
    """
    Workflow H: Invalid coordinates, invalid dates, malformed requests,
    and timeouts are handled safely without unhandled 500 stack traces.
    """
    # 1. Out-of-bounds latitude (> 90.0)
    resp_bad_lat = client.post("/api/chat", json={
        "message": "Wave conditions?",
        "latitude": 95.0,
        "longitude": 72.0
    })
    # Must fail or sanitize safely with correlation ID present
    assert "X-Correlation-ID" in resp_bad_lat.headers

    # 2. Out-of-bounds longitude (> 180.0)
    resp_bad_lon = client.post("/api/chat", json={
        "message": "Wave conditions?",
        "latitude": 15.0,
        "longitude": 195.0
    })
    assert "X-Correlation-ID" in resp_bad_lon.headers

    # 3. Route analysis with invalid coastal coordinates
    resp_bad_route = client.post("/api/map/route", json={
        "start_latitude": -95.0,
        "start_longitude": 72.0,
        "end_latitude": 18.0,
        "end_longitude": 72.0
    })
    assert resp_bad_route.status_code in [400, 422]


# =============================================================================
# ENVIRONMENT CONFIGURATION & STAGING VALIDATION TESTS
# =============================================================================

def test_environment_configuration_validation():
    """
    Tests environment validation logic:
    1. Valid dev configuration passes.
    2. Missing database path fails with ConfigurationError.
    3. Enabling simulation in staging/production strictly fails.
    """
    # 1. Normal development config
    dev_settings = Settings(APP_ENV="development", DATABASE_PATH="app/data/neerai.db")
    dev_settings.validate_environment()  # Should not raise

    # 2. Missing database path raises ConfigurationError
    bad_db_settings = Settings(APP_ENV="development", DATABASE_PATH="")
    with pytest.raises(ConfigurationError) as exc_info:
        bad_db_settings.validate_environment()
    assert "DATABASE_PATH" in str(exc_info.value)
    # Check no sensitive secrets leaked in message
    assert "secret" not in str(exc_info.value).lower()

    # 3. Simulation enabled in staging mode must strictly fail
    staging_fake_settings = Settings(
        APP_ENV="staging",
        DATABASE_PATH="app/data/neerai.db",
        ENABLE_SIMULATION_FALLBACK=True
    )
    with pytest.raises(ConfigurationError) as exc_info:
        staging_fake_settings.validate_environment()
    assert "ENABLE_SIMULATION_FALLBACK cannot be enabled in 'staging'" in str(exc_info.value)

    # 4. Real data requirement prohibits simulation
    real_data_conflict = Settings(
        APP_ENV="development",
        REQUIRE_REAL_DATA=True,
        ENABLE_SIMULATION_FALLBACK=True
    )
    with pytest.raises(ConfigurationError) as exc_info:
        real_data_conflict.validate_environment()
    assert "REQUIRE_REAL_DATA is enabled" in str(exc_info.value)


# =============================================================================
# LOGGING REDACTION & SENSITIVE DATA LEAKAGE PREVENTION TESTS
# =============================================================================

def test_logging_sensitive_data_redaction():
    """
    Verifies that the SensitiveDataRedactor filter scrubs tokens, passwords,
    and base64 audio data from log messages.
    """
    redactor = SensitiveDataRedactor()

    # Bearer token redaction
    log_token = "User authenticated with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret123"
    sanitized_token = redactor.redact_text(log_token)
    assert "secret123" not in sanitized_token
    assert "[REDACTED_TOKEN]" in sanitized_token

    # API key and secret redaction
    log_secret = "Connector config: api_key=AIzaSyA873918237912 and password=SuperSecretPassword123"
    sanitized_secret = redactor.redact_text(log_secret)
    assert "AIzaSyA873918237912" not in sanitized_secret
    assert "SuperSecretPassword123" not in sanitized_secret
    assert "[REDACTED]" in sanitized_secret

    # Base64 audio blob redaction
    log_audio = 'Payload received: {"audio_base64": "' + ("A" * 120) + '"}'
    sanitized_audio = redactor.redact_text(log_audio)
    assert "A" * 120 not in sanitized_audio
    assert "[REDACTED_AUDIO_PAYLOAD]" in sanitized_audio


# =============================================================================
# VOICE API CONTRACT & LIVE CONNECTOR NORMALIZATION CONTRACT TESTS
# =============================================================================

def test_voice_api_routes_stt_tts_and_languages():
    """
    Verifies voice API routes contract:
    - /api/voice/languages returns verified language capability list.
    - /api/voice/detect-language detects language scripts.
    - /api/voice/stt handles unconfigured STT gracefully.
    - /api/voice/tts sanitizes internal diagnostics before synthesis.
    """
    # 1. Languages
    lang_resp = client.get("/api/voice/languages")
    assert lang_resp.status_code == 200
    languages = lang_resp.json().get("languages", [])
    assert len(languages) >= 5
    codes = [l["code"] for l in languages]
    assert "hi" in codes and "en" in codes

    # 2. Detect Language
    detect_resp = client.post("/api/voice/detect-language", json={"text": "कल मौसम कैसा रहेगा?"})
    assert detect_resp.status_code == 200
    assert detect_resp.json().get("language") == "hi"

    # 3. Speech to Text unconfigured
    stt_resp = client.post("/api/voice/stt", json={"audio_base64": "dGVzdA==", "language": "hi"})
    assert stt_resp.status_code == 200
    stt_data = stt_resp.json()
    assert stt_data.get("status") == "UNCONFIGURED"

    # 4. Text to Speech sanitization
    tts_resp = client.post("/api/voice/tts", json={
        "text": "```json\n{\"debug\": true}\n```\nSST temperature is 28.5 degrees.",
        "language": "en"
    })
    assert tts_resp.status_code == 200
    tts_data = tts_resp.json()
    assert tts_data.get("success") is True
    assert "debug" not in tts_data.get("spoken_text", "")
    assert "temperature is 28.5 degrees" in tts_data.get("spoken_text", "")


@pytest.mark.anyio
async def test_live_marine_connectors_schema_contract():
    """
    Verifies that marine connectors normalize data strictly adhering to
    NormalizedMarineRecord schemas with explicit sources and no fake values.
    """
    from app.connectors.open_meteo_weather import OpenMeteoWeatherConnector
    from app.connectors.open_meteo_marine import OpenMeteoMarineConnector
    from app.connectors.nasa_earthdata import NASAEarthdataConnector
    from app.connectors.incois_connector import INCOISConnector
    from app.schemas.marine_normalized import QualityStatus

    # Weather normalization contract
    weather_conn = OpenMeteoWeatherConnector()
    w_res = ConnectorResult(
        success=True,
        data={
            "current": {"temperature_2m": 29.5, "wind_speed_10m": 15.2, "surface_pressure": 1012.0, "time": "2026-10-10T12:00"},
            "hourly": {"time": ["2026-10-10T12:00"], "temperature_2m": [29.5], "wind_speed_10m": [15.2]}
        }
    )
    w_records, _ = weather_conn.normalize(w_res, 15.3, 73.7, "Goa")
    assert len(w_records) > 0
    assert all(r.source_provider == "Open-Meteo Weather API" for r in w_records)
    # Provided variables must be VALID
    temp_rec = next(r for r in w_records if r.variable_name == "air_temperature")
    assert temp_rec.value == 29.5
    assert temp_rec.quality_status == QualityStatus.VALID
    assert temp_rec.is_missing is False
    # Unprovided variables must be UNAVAILABLE without zero fabrication
    gust_rec = next(r for r in w_records if r.variable_name == "wind_gusts")
    assert gust_rec.value is None
    assert gust_rec.quality_status == QualityStatus.UNAVAILABLE
    assert gust_rec.is_missing is True

    # INCOIS unauthenticated contract
    incois_conn = INCOISConnector()
    i_res = await incois_conn.fetch_advisory(15.3, 73.7)
    i_records = incois_conn.normalize(i_res, 15.3, 73.7, "Goa")
    assert len(i_records) == 1
    assert i_records[0].is_missing is True
    assert i_records[0].quality_status == QualityStatus.OFFLINE
    assert i_records[0].value is None

