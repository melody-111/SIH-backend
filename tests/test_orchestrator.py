import pytest
import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.agents.orchestrator import Orchestrator
from app.schemas.chat import UserQuery
from app.schemas.agent import AgentResult
from app.services.offline_cache import OfflineMarineCache

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_cache():
    cache = OfflineMarineCache()
    cache.clear()
    cache.set_force_offline(False)
    yield
    cache.clear()
    cache.set_force_offline(False)

# -------------------------------------------------------------
# 1. Routing & Agent Selection: Example A (Fishing Assistance)
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_example_a_fishing_assistance_routing():
    orchestrator = Orchestrator()
    query = UserQuery(
        message="Suggest fishing areas near my location for tomorrow.",
        latitude=15.30,
        longitude=73.70,
        language="en"
    )

    response = await orchestrator.process_query(query)

    # Verifications
    assert response.risk in ["LOW", "MODERATE", "HIGH"]
    assert "weather" in response.execution_plan
    assert "ocean" in response.execution_plan
    assert "pfz" in response.execution_plan
    # Verify no fabricated PFZ coordinates if official INCOIS is unauthenticated
    if "Official INCOIS PFZ data is unauthenticated" in response.answer:
        assert response.recommended_location is None
    # Progress steps must reflect actual execution
    step_names = [s.step for s in response.progress_steps]
    assert "Understanding query" in step_names
    assert "Fetching ocean data" in step_names
    assert "Validating evidence" in step_names

# -------------------------------------------------------------
# 2. Routing & Agent Selection: Example B (Marine Hazard)
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_example_b_marine_hazard_routing():
    orchestrator = Orchestrator()
    query = UserQuery(
        message="Is there a cyclone or dangerous sea condition near this coastal area?",
        latitude=15.30,
        longitude=73.70,
        language="en"
    )

    response = await orchestrator.process_query(query)

    assert "hazard" in response.execution_plan
    assert "weather" in response.execution_plan
    step_names = [s.step for s in response.progress_steps]
    assert "Checking official warnings" in step_names

# -------------------------------------------------------------
# 3. Routing & Agent Selection: Example C (Ocean Information / SST)
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_example_c_ocean_info_sst_routing():
    orchestrator = Orchestrator()
    query = UserQuery(
        message="What is the sea-surface temperature in this area?",
        latitude=15.30,
        longitude=73.70,
        language="en"
    )

    response = await orchestrator.process_query(query)

    # Selective planning: ocean agent alone should be planned
    assert response.execution_plan == ["ocean"]
    assert "ocean" in response.agents_used
    assert "weather" not in response.agents_used  # Weather should NOT be unnecessarily called!
    assert "Open-Meteo Marine" in response.answer or "SST" in response.answer

# -------------------------------------------------------------
# 4. Routing & Agent Selection: Example D (General Explanatory)
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_example_d_general_explanatory_question():
    orchestrator = Orchestrator()
    query = UserQuery(
        message="Explain what chlorophyll concentration tells us about ocean productivity.",
        latitude=15.30,
        longitude=73.70,
        language="en"
    )

    response = await orchestrator.process_query(query)

    # Must NOT call live sensor agents or pretend data was retrieved
    assert response.execution_plan == []
    assert len(response.agents_used) == 0
    assert "phytoplankton" in response.answer.lower() or "productivity" in response.answer.lower()

# -------------------------------------------------------------
# 5. Invalid Coordinates Handling
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_invalid_coordinates_validation():
    orchestrator = Orchestrator()
    query = UserQuery(
        message="Check weather",
        latitude=120.0,  # Invalid latitude (> 90)
        longitude=73.70
    )

    response = await orchestrator.process_query(query)
    assert response.risk == "UNKNOWN"
    assert "Invalid coordinates" in response.answer
    assert any("Invalid" in w for w in response.warnings)

# -------------------------------------------------------------
# 6. Partial Failure Resilience (One Specialist Fails)
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_partial_specialist_failure_resilience():
    orchestrator = Orchestrator()
    query = UserQuery(
        message="Suggest fishing areas near my location",
        latitude=15.30,
        longitude=73.70
    )

    # Simulate weather agent failing completely
    with patch.object(orchestrator.weather, "execute", side_effect=Exception("Weather provider connection down")):
        response = await orchestrator.process_query(query)
        # Should not crash! Must return partial results from other agents
        assert response.answer != ""
        assert "ocean" in response.agents_used

# -------------------------------------------------------------
# 7. Safety Rule: Official Warnings Take Precedence
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_official_warning_takes_precedence():
    orchestrator = Orchestrator()
    query = UserQuery(
        message="Is it safe to go out today?",
        latitude=15.30,
        longitude=73.70
    )

    # Inject an active official cyclone alert
    mock_hazard_result = AgentResult(
        agent="hazard",
        status="SUCCESS",
        data={
            "has_hazards": True,
            "official_warnings": [{
                "active_cyclone": True,
                "bulletin": "Severe Cyclonic Storm Warning issued by IMD for Goa-Konkan coast."
            }],
            "is_official_feed_active": True
        },
        confidence=1.0,
        sources=["IMD Official"],
        timestamp=datetime.datetime.utcnow().isoformat() + "Z"
    )

    with patch.object(orchestrator.hazard, "execute", return_value=mock_hazard_result):
        response = await orchestrator.process_query(query)
        assert response.risk == "HIGH"
        assert any("IMD Cyclone Warning" in w for w in response.warnings)

# -------------------------------------------------------------
# 8. Offline Cache Behavior
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_offline_cache_behavior():
    cache = OfflineMarineCache()
    orchestrator = Orchestrator()

    # Pre-populate cache at 15.30, 73.70
    sample_snapshot = {
        "weather": {"data": {"air_temperature": 29.0, "wind_speed": 12.0}},
        "ocean": {"data": {"wave_height": 0.8, "sst": 30.5}}
    }
    cache.store_observation(15.30, 73.70, sample_snapshot, location_name="Goa Coast")

    # Enable offline mode
    cache.set_force_offline(True)

    query = UserQuery(
        message="Kya aaj samundar mein jaana theek hai?",
        latitude=15.30,
        longitude=73.70,
        language="hi"
    )
    response = await orchestrator.process_query(query)

    assert response.is_cached is True
    assert response.cached_at is not None
    assert "OFFLINE CACHE" in response.answer

    # Test offline when NO cache exists for distant coordinates (e.g. Lakshadweep)
    query_distant = UserQuery(
        message="Weather check",
        latitude=10.56,
        longitude=72.64,
        language="en"
    )
    resp_no_cache = await orchestrator.process_query(query_distant)
    assert resp_no_cache.risk == "UNKNOWN"
    assert "no cached marine observations" in resp_no_cache.answer.lower()

# -------------------------------------------------------------
# 9. API Route Compatibility with Existing Mobile Client
# -------------------------------------------------------------
def test_chat_api_mobile_compatibility():
    payload = {
        "message": "Kal fishing kaha karein?",
        "latitude": 15.2993,
        "longitude": 74.1240,
        "language": "hi",
        "conversation_id": "session_test_mobile"
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Mobile expectations
    assert "answer" in data
    assert "risk" in data
    assert "warnings" in data
    assert "reasons" in data
    assert "evidence" in data
    assert "map_actions" in data
    assert "progress_steps" in data
    assert len(data["progress_steps"]) > 0

# -------------------------------------------------------------
# 10. Real-Time Streaming Endpoint (/api/chat/stream)
# -------------------------------------------------------------
def test_chat_stream_endpoint():
    payload = {
        "message": "What is the wave height near Goa?",
        "latitude": 15.30,
        "longitude": 73.70,
        "language": "en"
    }
    resp = client.post("/api/chat/stream", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "data: " in text
    assert '"type": "progress"' in text
    assert '"type": "final"' in text
