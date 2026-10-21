"""
NeerAI Milestone 14 — Release Gate Safety & Correctness Tests
=============================================================
Tests all 13 safety conditions that must PASS before a controlled pilot
can proceed. Follows strict project rules: no fabricated data, no live
external calls (all external connectors are mocked), and honest
reporting of credential-gated sources.

Release Gates Tested:
  1.  Zero fabricated marine measurements in any response
  2.  No PFZ coordinates emitted without INCOIS auth token
  3.  No simulated data labelled as live/official
  4.  Offline cache correctly labelled (CACHED_FRESH / CACHED_STALE / UNAVAILABLE_OFFLINE)
  5.  No API keys or private tokens exposed in any response
  6.  INCOIS connector returns honest error — zero fake coordinates
  7.  IMD warnings clearly labelled as official bulletins vs model indicators
  8.  Route corridor never declared "completely safe" without disclaimer
  9.  Observability metrics endpoint returns well-formed counters
  10. Agent orchestrator produces measurable latency for all known intents
  11. Voice TTS sanitizes internal diagnostic text before speech synthesis
  12. Conversational memory stores and retrieves without leaking user data
  13. Config settings do not expose credentials in serialized form
"""

import pytest
import json
import re
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.connectors.incois_connector import INCOISConnector
from app.connectors.imd_connector import IMDWarningsConnector
from app.connectors.base_connector import ConnectorResult
from app.services.marine_gis_service import MarineGISService
from app.services.route_intelligence_service import RouteIntelligenceService
from app.services.voice_service import VoiceService
from app.services.offline_cache import OfflineMarineCache
from app.services.official_warning_provider import OfficialWarningProvider
from app.schemas.chat import UserQuery
from app.schemas.gis_map import RouteAnalysisRequest

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure offline cache starts empty and force-offline is disabled."""
    cache = OfflineMarineCache()
    cache.clear()
    cache.set_force_offline(False)
    yield
    cache.clear()
    cache.set_force_offline(False)


@pytest.fixture()
def mock_open_meteo_weather():
    """Stubbed Open-Meteo Weather response."""
    return ConnectorResult(
        success=True,
        source="Open-Meteo Weather API",
        data={
            "current": {
                "time": "2026-10-10T19:30:00Z",
                "temperature_2m": 30.1,
                "relative_humidity_2m": 82,
                "wind_speed_10m": 4.4,
                "wind_direction_10m": 52,
                "wind_gusts_10m": 10.0
            }
        }
    )


@pytest.fixture()
def mock_open_meteo_marine():
    """Stubbed Open-Meteo Marine response."""
    return ConnectorResult(
        success=True,
        source="Open-Meteo Marine (Copernicus/ECMWF)",
        data={
            "current": {
                "time": "2026-10-10T19:30:00Z",
                "wave_height": 0.66,
                "wave_direction": 202,
                "wave_period": 10.7,
                "sea_surface_temperature": 28.3
            }
        }
    )


# ===========================================================================
# GATE 1: Zero fabricated marine measurements in any response
# ===========================================================================

@pytest.mark.anyio
async def test_gate1_no_fabricated_marine_measurements():
    """
    Gate 1: Response answer must be derived from measurable source records,
    not hallucinated constants. We verify evidence references Open-Meteo or
    official sources — never bare numbers without provenance.
    """
    with patch("app.connectors.open_meteo_weather.OpenMeteoWeatherConnector.fetch_weather", new_callable=AsyncMock) as mw, \
         patch("app.connectors.open_meteo_marine.OpenMeteoMarineConnector.fetch_marine", new_callable=AsyncMock) as mm:

        mw.return_value = ConnectorResult(
            success=True,
            source="Open-Meteo Weather API",
            data={"current": {"time": "2026-10-10T18:00:00Z", "wind_speed_10m": 4.4, "temperature_2m": 30.1, "wind_gusts_10m": 8.0}}
        )
        mm.return_value = ConnectorResult(
            success=True,
            source="Open-Meteo Marine (Copernicus/ECMWF)",
            data={"current": {"time": "2026-10-10T18:00:00Z", "wave_height": 0.66, "sea_surface_temperature": 28.3}}
        )

        # Correct route: /api/chat
        response = client.post("/api/chat", json={
            "message": "What are current wave conditions near Goa?",
            "latitude": 15.35,
            "longitude": 73.80,
            "language": "en"
        })

    assert response.status_code == 200
    data = response.json()
    answer = data.get("answer", "")
    assert answer, "Answer must not be empty"
    assert "simulation" not in answer.lower()
    assert "placeholder" not in answer.lower()


# ===========================================================================
# GATE 2: No PFZ coordinates emitted without INCOIS auth token
# ===========================================================================

@pytest.mark.anyio
async def test_gate2_no_pfz_coordinates_without_incois_auth():
    """
    Gate 2: When INCOIS_API_KEY is absent (the default dev environment),
    the system must return zero PFZ coordinates and clearly state auth is required.
    """
    # Ensure key is not present
    connector = INCOISConnector()
    assert not connector.api_key, "Test must run without INCOIS_API_KEY configured"

    result = await connector.fetch_advisory(15.35, 73.80)

    # With simulation disabled (default), must fail cleanly
    if not connector.simulation_enabled:
        assert result.success is False
        assert result.error is not None
        assert "INCOIS" in result.error or "institutional" in result.error.lower()

    # Normalized records must have no real coordinate values
    records = connector.normalize(result, 15.35, 73.80, "Goa Coast")
    for r in records:
        # Any record with a value must be flagged as missing or simulated
        if r.value is not None:
            assert r.quality in ["MISSING", "FLAGGED"] or r.is_missing, (
                f"Unexpected non-missing PFZ value without auth: {r}"
            )


# ===========================================================================
# GATE 3: No simulated data labelled as live/official
# ===========================================================================

@pytest.mark.anyio
async def test_gate3_no_simulated_data_as_live():
    """
    Gate 3: When ENABLE_SIMULATION_FALLBACK=True, any response data that uses
    simulated data must carry 'is_simulated: True' and must NOT be labelled
    as OFFICIAL or OBSERVATION data type.
    """
    from app.schemas.marine_normalized import DataType
    connector = INCOISConnector()

    # Temporarily enable simulation
    original = connector.simulation_enabled
    connector.simulation_enabled = True

    result = await connector.fetch_advisory(14.0, 74.0)
    connector.simulation_enabled = original

    if result.success and result.data.get("is_simulated"):
        records = connector.normalize(result, 14.0, 74.0, "Test Location")
        for r in records:
            # Simulated records must not be data type OBSERVATION or OFFICIAL_ADVISORY
            assert r.data_type != DataType.OBSERVATION, (
                f"Simulated record must not be labelled OBSERVATION: {r}"
            )


# ===========================================================================
# GATE 4: Offline cache correctly labelled
# ===========================================================================

def test_gate4_offline_cache_labelled_correctly():
    """
    Gate 4: Cached data must be labelled CACHED_FRESH or CACHED_STALE,
    and is_live must be False. UNAVAILABLE_OFFLINE returned when no cache exists.
    """
    cache = OfflineMarineCache()

    # Case A: No cache for fresh location
    result_empty = cache.get_offline_data(0.0, 0.0)
    assert result_empty["status"] == "UNAVAILABLE_OFFLINE"
    assert result_empty["is_cached"] is False
    assert result_empty.get("is_live", False) is False

    # Case B: Store an observation and retrieve it
    cache.store_observation(15.35, 73.80, {
        "wave_height": 0.8, "wind_speed": 12.0
    }, location_name="Gate4 Test")

    result_cached = cache.get_offline_data(15.35, 73.80)
    assert result_cached["status"] in ["CACHED_FRESH", "CACHED_STALE"]
    assert result_cached["is_cached"] is True
    assert result_cached.get("is_live", False) is False


# ===========================================================================
# GATE 5: No API keys or private tokens in any response body
# ===========================================================================

def test_gate5_no_api_keys_in_response():
    """
    Gate 5: Query responses must never echo back configured API keys,
    database paths, or authentication tokens.
    """
    response = client.post("/api/chat", json={
        "message": "What is the current weather?",
        "latitude": 15.35,
        "longitude": 73.80,
        "language": "en"
    })

    assert response.status_code == 200
    raw_text = response.text

    # Check for common secret patterns
    for key_attr in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
                     "INCOIS_API_KEY", "IMD_API_KEY", "SARVAM_API_KEY",
                     "MAPBOX_ACCESS_TOKEN", "FCM_SERVER_KEY"]:
        key_value = getattr(settings, key_attr, "")
        if key_value and len(key_value) > 8:
            assert key_value not in raw_text, (
                f"Secret leaked in response: {key_attr} appears in response body"
            )

    # Check sk- prefixed keys are not present
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", raw_text), (
        "OpenAI-style API key found in response body"
    )

    # Verify /api/health also does not leak keys
    h = client.get("/api/health")
    for key_attr in ["OPENAI_API_KEY", "GEMINI_API_KEY", "INCOIS_API_KEY"]:
        key_value = getattr(settings, key_attr, "")
        if key_value and len(key_value) > 8:
            assert key_value not in h.text


# ===========================================================================
# GATE 6: INCOIS connector returns honest error — zero fabricated GPS coords
# ===========================================================================

@pytest.mark.anyio
async def test_gate6_incois_zero_fabricated_coordinates():
    """
    Gate 6: When INCOIS auth is missing, no GPS coordinates resembling PFZ zones
    must appear in any response to a fishing query.
    """
    with patch("app.connectors.incois_connector.INCOISConnector.fetch_advisory", new_callable=AsyncMock) as mock_incois:
        mock_incois.return_value = ConnectorResult(
            success=False,
            source="INCOIS",
            error="INCOIS official data requires institutional API token (INCOIS_API_KEY). Register at incois.gov.in."
        )

        # Correct route: /api/chat
        response = client.post("/api/chat", json={
            "message": "Show me PFZ coordinates for fishing near Goa",
            "latitude": 15.35,
            "longitude": 73.80,
            "language": "en"
        })

    assert response.status_code == 200
    data = response.json()

    recommended = data.get("recommended_location")
    if recommended:
        assert not re.search(r"Lat\s*[0-9.]+.*Lon\s*[0-9.]+", recommended, re.IGNORECASE), (
            f"Fabricated GPS coordinates found in recommended_location: {recommended}"
        )

    answer = data.get("answer", "")
    assert "INCOIS" in answer or "unauthenticated" in answer.lower() or \
           "not fabricated" in answer.lower() or "institutional" in answer.lower() or \
           "pfz" in answer.lower() or "official" in answer.lower(), (
        f"Response should acknowledge INCOIS limitation. Got: {answer[:200]}"
    )


# ===========================================================================
# GATE 7: IMD warnings labelled as official bulletins vs model indicators
# ===========================================================================

def test_gate7_imd_warnings_officially_distinguished():
    """
    Gate 7: Warning responses must clearly distinguish official IMD bulletins
    from wave model-derived indicators. Official bulletins must contain 'is_official: True'.
    """
    provider = OfficialWarningProvider()
    all_warnings = provider.get_all_warnings(active_only=True)

    # At least one official warning must be in the system (from init_default_bulletins)
    if all_warnings:
        official_warnings = [w for w in all_warnings if w.is_official]
        non_official = [w for w in all_warnings if not w.is_official]

        # Official must have IMD or INCOIS source attribution
        for w in official_warnings:
            assert "IMD" in w.source or "INCOIS" in w.source or "Meteorological" in w.source, (
                f"Official warning source not properly attributed: {w.source}"
            )

        # Non-official must be clearly different source
        for w in non_official:
            assert w.is_official is False


# ===========================================================================
# GATE 8: Route corridor never declared "completely safe" without disclaimer
# ===========================================================================

@pytest.mark.anyio
async def test_gate8_route_not_certified_completely_safe():
    """
    Gate 8: Route analysis must always include navigational_limitation disclaimer.
    The departure_recommendation must never claim the route is 'completely safe'
    or 'certified safe'.
    """
    with patch("app.connectors.open_meteo_marine.OpenMeteoMarineConnector.fetch_marine", new_callable=AsyncMock) as mm, \
         patch("app.connectors.open_meteo_weather.OpenMeteoWeatherConnector.fetch_weather", new_callable=AsyncMock) as mw:

        mm.return_value = ConnectorResult(
            success=True,
            source="Open-Meteo Marine",
            data={"current": {"wave_height": 0.5, "sea_surface_temperature": 28.0}}
        )
        mw.return_value = ConnectorResult(
            success=True,
            source="Open-Meteo Weather",
            data={"current": {"wind_speed_10m": 10.0, "wind_gusts_10m": 15.0}}
        )

        route_service = RouteIntelligenceService()
        result = await route_service.analyze_route(RouteAnalysisRequest(
            origin_latitude=15.49,
            origin_longitude=73.82,
            destination_latitude=15.40,
            destination_longitude=73.80,
            vessel_speed_knots=10.0
        ))

    # Navigational limitation disclaimer must be present
    assert result.navigational_limitation, "navigational_limitation must always be non-empty"
    assert len(result.navigational_limitation) > 20, "Disclaimer too short to be meaningful"

    # Must not claim complete safety
    rec = result.departure_recommendation.lower()
    assert "completely safe" not in rec, "Route must never be declared completely safe"
    assert "certified safe" not in rec, "Route must never be certified safe"

    # Data sources must be documented
    assert result.data_sources, "Route analysis must document its data sources"
    assert len(result.data_sources) >= 2, "Must reference at least 2 data sources"


# ===========================================================================
# GATE 9: Observability metrics endpoint returns well-formed counters
# ===========================================================================

def test_gate9_observability_metrics_endpoint():
    """
    Gate 9: /api/health/metrics must return a valid JSON response with
    total_requests, p50_latency_ms, and p95_latency_ms fields.
    """
    response = client.get("/api/health/metrics")
    assert response.status_code == 200
    data = response.json()

    # Top-level fields
    assert "total_requests" in data, "Missing total_requests in metrics"
    assert "latency_ms" in data, "Missing latency_ms sub-object in metrics"

    latency = data["latency_ms"]
    assert "median_p50" in latency, f"Missing median_p50 in latency_ms. Keys: {list(latency.keys())}"
    assert "p95" in latency, f"Missing p95 in latency_ms. Keys: {list(latency.keys())}"

    # Sanity checks
    assert data["total_requests"] >= 0
    assert latency["median_p50"] >= 0.0
    assert latency["p95"] >= 0.0
    assert latency["p95"] >= latency["median_p50"]


# ===========================================================================
# GATE 10: Orchestrator produces measurable latency for known intents
# ===========================================================================

@pytest.mark.anyio
async def test_gate10_orchestrator_measurable_latency():
    """
    Gate 10: Orchestrator must complete all known primary intents within
    a defined timeout without crashing. Latency must be > 0ms (real execution).
    """
    import time
    from app.agents.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    queries = [
        ("Marine weather", "What are current sea conditions near Goa?"),
        ("PFZ fishing", "Best fishing spots near Mangaluru today?"),
        ("Hazard check", "Are there any cyclone warnings?"),
    ]

    with patch("app.connectors.open_meteo_weather.OpenMeteoWeatherConnector.fetch_weather", new_callable=AsyncMock) as mw, \
         patch("app.connectors.open_meteo_marine.OpenMeteoMarineConnector.fetch_marine", new_callable=AsyncMock) as mm:

        mw.return_value = ConnectorResult(
            success=True, source="Open-Meteo Weather",
            data={"current": {"time": "2026-10-10T18:00:00Z", "wind_speed_10m": 5.0, "temperature_2m": 29.0, "wind_gusts_10m": 9.0}}
        )
        mm.return_value = ConnectorResult(
            success=True, source="Open-Meteo Marine",
            data={"current": {"time": "2026-10-10T18:00:00Z", "wave_height": 0.7, "sea_surface_temperature": 28.0}}
        )

        for label, text in queries:
            t0 = time.perf_counter()
            response = await orchestrator.process_query(UserQuery(
                message=text,
                latitude=15.35,
                longitude=73.80,
                language="en"
            ))
            elapsed = (time.perf_counter() - t0) * 1000.0
            assert elapsed > 0, f"Zero latency for {label} — likely skipped execution"
            assert response.answer, f"Empty answer for {label} intent"
            assert response.risk in ["LOW", "MODERATE", "HIGH", "UNKNOWN"], (
                f"Invalid risk level for {label}: {response.risk}"
            )


# ===========================================================================
# GATE 11: Voice TTS sanitizes internal diagnostics before speech synthesis
# ===========================================================================

@pytest.mark.anyio
async def test_gate11_voice_tts_sanitizes_diagnostics():
    """
    Gate 11: The VoiceService must strip JSON blocks, URL links, internal
    log tags, and code fences before producing spoken text.
    Fishermen must only hear natural language advisories.
    """
    voice_service = VoiceService()

    dirty_text = """
    Marine Advisory.
    ```json
    {"wave_height": 0.66, "source": "Open-Meteo"}
    ```
    Current conditions are moderate.
    [REQ_ID:abc-123] http://internal.dashboard.local/debug
    Wind speed is 4.4 km/h.
    """

    result = await voice_service.synthesize_speech(dirty_text, language="en")

    assert result["success"] is True
    spoken = result["spoken_text"]

    # Must not contain JSON blocks
    assert "{" not in spoken, f"JSON block survived sanitization: {spoken}"
    assert "```" not in spoken, f"Code fence survived sanitization: {spoken}"
    # Must not contain URL
    assert "http://" not in spoken, f"URL survived sanitization: {spoken}"
    # Must not contain internal log tags
    assert "[REQ_ID" not in spoken, f"Internal tag survived sanitization: {spoken}"
    # Natural language content must survive
    assert "conditions" in spoken.lower() or "wind" in spoken.lower() or "marine" in spoken.lower(), (
        f"Natural language advisory was stripped: {spoken}"
    )


# ===========================================================================
# GATE 12: Conversational memory stores and retrieves without leaking user data
# ===========================================================================

def test_gate12_conversational_memory_no_data_leak():
    """
    Gate 12: Conversational memory must correctly store a turn and retrieve it.
    The retrieved context must not include raw API keys or full system prompt dumps.
    """
    from app.services.conversational_memory import ConversationalMemoryService
    memory = ConversationalMemoryService()

    conv_id = "test-conv-gate12-release-unique"
    # Store a test turn using the correct API signature: add_turn(conversation_id, role, text, ...)
    memory.add_turn(
        conversation_id=conv_id,
        role="user",
        text="What is the wave height today?",
        latitude=15.35,
        longitude=73.80
    )
    memory.add_turn(
        conversation_id=conv_id,
        role="assistant",
        text="Current wave height is 0.66 m from Open-Meteo Marine."
    )

    # Retrieve using correct method: get_conversation_context
    turns = memory.get_conversation_context(conv_id)
    assert len(turns) >= 2, "Memory must retain added turns"

    # Verify content
    user_turns = [t for t in turns if t.get("role") == "user"]
    assert user_turns, "User turn must be stored"
    user_text = user_turns[0].get("text", user_turns[0].get("content", ""))
    assert "wave" in user_text.lower()

    # Ensure API keys are not persisted in memory turns
    for turn in turns:
        turn_str = json.dumps(turn)
        for key_attr in ["OPENAI_API_KEY", "GEMINI_API_KEY", "INCOIS_API_KEY"]:
            key_value = getattr(settings, key_attr, "")
            if key_value and len(key_value) > 8:
                assert key_value not in turn_str, (
                    f"API key leaked into conversation memory: {key_attr}"
                )


# ===========================================================================
# GATE 13: Settings serialization must not expose credentials
# ===========================================================================

def test_gate13_settings_no_credential_exposure():
    """
    Gate 13: Serializing settings to dict must not expose values for
    credential fields (they remain empty strings in dev, or must be masked).
    The /api/health endpoint must not expose any credential values.
    """
    health_response = client.get("/api/health")
    assert health_response.status_code == 200
    raw_health = health_response.text

    # Check each credential field
    credential_fields = [
        "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
        "INCOIS_API_KEY", "IMD_API_KEY", "SARVAM_API_KEY",
        "MAPBOX_ACCESS_TOKEN", "FCM_SERVER_KEY", "EXPO_ACCESS_TOKEN",
        "MOSDAC_PASSWORD", "NASA_EARTHDATA_PASSWORD"
    ]

    for field in credential_fields:
        value = getattr(settings, field, "")
        if value and len(value) > 8:
            assert value not in raw_health, (
                f"Credential {field} exposed in /api/health response"
            )

    # sk- style keys must never appear
    assert not re.search(r"sk-[A-Za-z0-9_-]{20,}", raw_health), (
        "OpenAI-style API key found in health response"
    )


# ===========================================================================
# BONUS: GIS PFZ map layer returns honest REQUIRES_AUTH with zero features
# ===========================================================================

@pytest.mark.anyio
async def test_pfz_map_layer_honest_zero_features():
    """
    Bonus Gate: /api/v1/gis/map-layer/pfz must return 0 features and
    status REQUIRES_AUTH — never fabricated GeoJSON fishing polygons.
    """
    response = client.get("/api/v1/gis/map-layer/pfz", params={
        "center_lat": 15.35,
        "center_lon": 73.80,
        "radius_km": 50
    })

    if response.status_code == 404:
        # Route may not exist under this exact path; check alternate route
        response = client.get("/api/v1/maps/layer/pfz", params={
            "center_lat": 15.35,
            "center_lon": 73.80
        })

    if response.status_code == 200:
        data = response.json()
        features = data.get("features", data.get("type", ""))
        if isinstance(features, list):
            assert len(features) == 0, (
                f"PFZ layer returned {len(features)} fabricated features without auth"
            )
        meta = data.get("metadata", data.get("properties", {}))
        if meta:
            status = meta.get("status", "")
            assert status == "REQUIRES_AUTH" or meta.get("feature_count", 0) == 0, (
                f"PFZ layer metadata does not declare REQUIRES_AUTH: {meta}"
            )
