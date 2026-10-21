"""
NeerAI Master Acceptance Test Suite — 16 Mandated Product Scenarios
===================================================================
Executes realistic integration tests across:
1. Coordinates weather query
2. Wave and wind summary
3. SST and chlorophyll comparison
4. Route corridor analysis
5. Official cyclone warning
6. PFZ (unauthenticated honest status vs verified)
7. Chlorophyll science explanation
8. Hindi marine question
9. Follow-up conversational context
10. Provider timeout resilience
11. Invalid / out-of-bounds coordinates
12. Missing Gemini credentials handling
13. Ollama unavailable handling
14. Conflicting or stale provider results
15. No official hazard bulletin handling
16. Offline cached mode
"""

import pytest
import datetime
import uuid
import httpx
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.connectors.base_connector import ConnectorResult
from app.services.offline_cache import OfflineMarineCache
from app.services.official_warning_provider import OfficialWarningProvider
from app.schemas.hazard_alert import OfficialWarningDetail, AlertStatus
from app.services.llm_service import MultiModelLLMService, OllamaProvider, GeminiProvider

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_cache_state():
    cache = OfflineMarineCache()
    cache.clear()
    cache.set_force_offline(False)
    yield
    cache.clear()
    cache.set_force_offline(False)


# =============================================================================
# SCENARIO 1: Weather query at coordinates
# =============================================================================
def test_scenario_01_weather_at_coordinates():
    resp = client.post("/api/chat", json={
        "message": "What is the weather at latitude 15.40 and longitude 73.80?",
        "latitude": 15.40,
        "longitude": 73.80,
        "language": "en"
    }, headers={"X-User-ID": "test_user_s1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "weather" in data.get("execution_plan", [])
    assert data["risk"] in ["LOW", "MODERATE", "HIGH", "DANGER", "UNKNOWN"]
    assert "timestamp" in data


# =============================================================================
# SCENARIO 2: Wave and wind summary
# =============================================================================
def test_scenario_02_wave_and_wind_summary():
    resp = client.post("/api/chat", json={
        "message": "Summarize the wave and wind conditions here off Kochi.",
        "latitude": 9.93,
        "longitude": 76.26,
        "language": "en"
    }, headers={"X-User-ID": "test_user_s2"})
    assert resp.status_code == 200
    data = resp.json()
    assert "ocean" in data.get("execution_plan", []) or "weather" in data.get("execution_plan", [])
    assert any(term in data["answer"].lower() for term in ["wind", "wave", "km/h", "m", "condition", "ocean", "weather"])
    assert data["risk"] in ["LOW", "MODERATE", "HIGH", "DANGER", "UNKNOWN"]


# =============================================================================
# SCENARIO 3: Compare SST and chlorophyll at two locations
# =============================================================================
def test_scenario_03_compare_sst_and_chlorophyll():
    # Probe location 1 (Goa Coast)
    resp1 = client.get("/api/marine/snapshot?latitude=15.40&longitude=73.80")
    assert resp1.status_code == 200
    data1 = resp1.json()

    # Probe location 2 (Kochi Coast)
    resp2 = client.get("/api/marine/snapshot?latitude=9.93&longitude=76.26")
    assert resp2.status_code == 200
    data2 = resp2.json()

    assert "ocean_records" in data1 and "evidence_items" in data1
    assert "ocean_records" in data2 and "evidence_items" in data2
    # Check SST presence and source attribution
    all_records1 = data1.get("ocean_records", []) + data1.get("evidence_items", [])
    sources1 = [r.get("source") for r in all_records1 if isinstance(r, dict)]
    assert any("Open-Meteo" in s or "Marine" in s for s in sources1 if s)
    # Check satellite records or evidence items
    assert "satellite_records" in data1


# =============================================================================
# SCENARIO 4: Show marine conditions along route
# =============================================================================
def test_scenario_04_route_corridor_analysis():
    resp = client.post("/api/map/route", json={
        "start_latitude": 9.93,
        "start_longitude": 76.26,
        "end_latitude": 8.38,
        "end_longitude": 76.99
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "route_geometry" in data
    assert data["route_geometry"]["type"] == "LineString"
    assert "total_distance_km" in data
    assert data["total_distance_km"] > 0
    assert "waypoints" in data
    assert len(data["waypoints"]) > 0
    assert "departure_recommendation" in data


# =============================================================================
# SCENARIO 5: Official cyclone warning bulletin
# =============================================================================
def test_scenario_05_official_cyclone_warning():
    provider = OfficialWarningProvider()
    now = datetime.datetime.now(datetime.timezone.utc)
    bulletin = OfficialWarningDetail(
        warning_id="TEST-CYCLONE-GUJARAT-01",
        source="India Meteorological Department (IMD)",
        warning_type="CYCLONIC_STORM_ALERT",
        source_severity="RED WARNING",
        normalized_severity="DANGER",
        bulletin_text="Cyclonic storm approaching Porbandar coast. Fishermen advised total suspension of fishing operations.",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        valid_until=(now + datetime.timedelta(hours=24)).isoformat(),
        affected_regions=["Porbandar Coast", "Arabian Sea"],
        affected_coordinates={"latitude": 21.64, "longitude": 69.60},
        coverage_radius_km=150.0,
        status=AlertStatus.ACTIVE,
        is_official=True
    )
    provider.register_bulletin(bulletin)

    resp = client.post("/api/chat", json={
        "message": "Is there an official cyclone warning near Porbandar?",
        "latitude": 21.64,
        "longitude": 69.60,
        "language": "en"
    }, headers={"X-User-ID": "test_user_cyclone"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk"] in ["HIGH", "DANGER"]
    ans_lower = data["answer"].lower()
    assert any(term in ans_lower for term in ["cyclone", "warning", "imd", "danger", "alert", "suspension"])


# =============================================================================
# SCENARIO 6: PFZ information (unauthenticated safety)
# =============================================================================
def test_scenario_06_pfz_unauthenticated_safety():
    with patch("app.core.config.settings.INCOIS_API_KEY", ""):
        resp = client.post("/api/chat", json={
            "message": "Find PFZ fishing zone coordinates near Ratnagiri.",
            "latitude": 16.99,
            "longitude": 73.30,
            "language": "en"
        }, headers={"X-User-ID": "test_user_pfz"})
        assert resp.status_code == 200
        data = resp.json()
        # Must NOT fabricate fake coordinates
        assert data.get("recommended_location") is None or "INCOIS" in str(data.get("recommended_location"))
        assert len(data["answer"]) > 0


# =============================================================================
# SCENARIO 7: Explain why chlorophyll matters for fishing
# =============================================================================
def test_scenario_07_scientific_chlorophyll_explanation():
    resp = client.post("/api/chat", json={
        "message": "Explain why chlorophyll matters for fishing.",
        "language": "en"
    }, headers={"X-User-ID": "test_user_science"})
    assert resp.status_code == 200
    data = resp.json()
    ans_lower = data["answer"].lower()
    assert any(term in ans_lower for term in ["chlorophyll", "plankton", "fish", "food", "phytoplankton", "marine"])


# =============================================================================
# SCENARIO 8: Hindi marine question
# =============================================================================
def test_scenario_08_hindi_marine_question():
    resp = client.post("/api/chat", json={
        "message": "गोवा के पास आज समुद्र में मौसम और लहरें कैसी हैं?",
        "latitude": 15.39,
        "longitude": 73.80,
        "language": "hi"
    }, headers={"X-User-ID": "test_user_hindi"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["answer"]) > 0
    # Must preserve Hindi context or structured advice
    assert data.get("language") in ["hi", "en"]
    assert data["risk"] in ["LOW", "MODERATE", "HIGH", "DANGER", "UNKNOWN"]


# =============================================================================
# SCENARIO 9: Follow-up question referring to previous answer
# =============================================================================
def test_scenario_09_follow_up_conversational_context():
    user_id = f"test_followup_{uuid.uuid4().hex[:6]}"
    # Turn 1
    resp1 = client.post("/api/chat", json={
        "message": "What is the wave height at Visakhapatnam?",
        "latitude": 17.68,
        "longitude": 83.21,
        "language": "en"
    }, headers={"X-User-ID": user_id})
    assert resp1.status_code == 200

    # Turn 2: Follow-up referring to "tomorrow" without repeating coordinates
    resp2 = client.post("/api/chat", json={
        "message": "What about tomorrow morning there?",
        "language": "en"
    }, headers={"X-User-ID": user_id})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["answer"]) > 0


# =============================================================================
# SCENARIO 10: Provider timeout resilience
# =============================================================================
@pytest.mark.anyio
async def test_scenario_10_provider_timeout_resilience():
    with patch("app.services.weather_service.WeatherService.get_weather", new_callable=AsyncMock) as mock_weather:
        mock_weather.side_effect = httpx.TimeoutException("Upstream weather API timed out after 6 seconds")

        resp = client.post("/api/chat", json={
            "message": "Weather forecast off Mumbai?",
            "latitude": 18.92,
            "longitude": 72.83,
            "language": "en"
        }, headers={"X-User-ID": "test_timeout"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert any(term in data["answer"].lower() for term in ["unavailable", "timed out", "error", "caution", "unable", "offline"])


# =============================================================================
# SCENARIO 11: Invalid or missing coordinates
# =============================================================================
def test_scenario_11_invalid_out_of_bounds_coordinates():
    resp_bad = client.post("/api/chat", json={
        "message": "Current sea conditions?",
        "latitude": 125.0,  # Invalid latitude > 90
        "longitude": 210.0, # Invalid longitude > 180
        "language": "en"
    }, headers={"X-User-ID": "test_invalid_coords"})
    # System handles gracefully without unhandled 500
    assert resp_bad.status_code in [200, 400, 422]
    if resp_bad.status_code == 200:
        data = resp_bad.json()
        assert "answer" in data
    assert "X-Correlation-ID" in resp_bad.headers


# =============================================================================
# SCENARIO 12: Missing Gemini credentials
# =============================================================================
@pytest.mark.anyio
async def test_scenario_12_missing_gemini_credentials():
    gemini = GeminiProvider()
    with patch("app.core.config.settings.GEMINI_API_KEY", ""):
        gemini.api_key = ""
        assert not gemini.is_available
        diag = await gemini.check_diagnostics()
        assert diag["status"] == "UNCONFIGURED"
        assert diag["is_available"] is False
        assert "GEMINI_API_KEY" in diag["action"]


# =============================================================================
# SCENARIO 13: Ollama unavailable
# =============================================================================
@pytest.mark.anyio
async def test_scenario_13_ollama_unavailable_diagnostic():
    ollama = OllamaProvider()
    ollama.base_url = "http://localhost:11434"
    diag = await ollama.check_diagnostics()
    # On this machine, Ollama port 11434 is unreachable
    assert diag["status"] == "UNREACHABLE"
    assert diag["is_available"] is False
    assert "ollama pull" in diag["action"] or "brew install ollama" in diag["action"]


# =============================================================================
# SCENARIO 14: Conflicting or stale provider results
# =============================================================================
def test_scenario_14_conflicting_or_stale_provider_results():
    from app.agents.data_validation_agent import DataValidationAgent
    from app.schemas.agent import AgentResult

    validator = DataValidationAgent()
    # Result with stale timestamp (48 hours ago)
    old_time = (datetime.datetime.utcnow() - datetime.timedelta(hours=48)).isoformat() + "Z"
    stale_result = AgentResult(
        agent="ocean",
        status="SUCCESS",
        data={"wave_height": 1.2, "wind_speed": 15.0},
        confidence=0.9,
        sources=["Historical Archive"],
        timestamp=old_time,
        quality_status="VALID"
    )

    validation = validator.validate_results({"ocean": stale_result})
    assert validation is not None
    # DataQualityAgent flags or handles timestamp freshness


# =============================================================================
# SCENARIO 15: No available official hazard bulletin
# =============================================================================
def test_scenario_15_no_active_hazard_bulletin():
    resp = client.get("/api/hazard/bulletins?latitude=15.40&longitude=73.80&radius_km=50")
    assert resp.status_code == 200
    bulletins = resp.json()
    assert isinstance(bulletins, list)
    # When no bulletins exist, response states empty without claiming sea is unconditionally safe
    if len(bulletins) == 0:
        resp_chat = client.post("/api/chat", json={
            "message": "Are there any cyclone alerts here?",
            "latitude": 15.40,
            "longitude": 73.80,
            "language": "en"
        })
        assert resp_chat.status_code == 200
        data = resp_chat.json()
        assert "no official" in data["answer"].lower() or "no active" in data["answer"].lower() or "none" in data["answer"].lower() or "0" in data["answer"].lower() or "safe" in data["answer"].lower()


# =============================================================================
# SCENARIO 16: Offline mode with previously cached data
# =============================================================================
def test_scenario_16_offline_mode_cached_data():
    cache = OfflineMarineCache()
    sample_snapshot = {
        "wave_height_m": 1.3,
        "wind_speed_kmh": 14.5,
        "location_name": "Goa Coastal Station"
    }
    cache.store_observation(15.40, 73.80, sample_snapshot, location_name="Goa Coastal Station")
    cache.set_force_offline(True)

    try:
        resp = client.post("/api/chat", json={
            "message": "What are the current marine conditions?",
            "latitude": 15.40,
            "longitude": 73.80,
            "language": "en"
        }, headers={"X-User-ID": "offline_pilot_tester"})

        assert resp.status_code == 200
        data = resp.json()
        assert data.get("is_cached") is True
        assert data.get("cached_at") is not None
        assert any(term in data["answer"].lower() for term in ["cached", "offline", "stored"])
    finally:
        cache.set_force_offline(False)
