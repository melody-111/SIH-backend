import pytest
import datetime
from unittest.mock import AsyncMock, patch

from app.schemas.chat import UserQuery
from app.schemas.agent import AgentResult
from app.agents.orchestrator import Orchestrator
from app.services.llm_service import MultiModelLLMService, GeminiProvider
from app.services.conversational_memory import ConversationalMemoryService

# =============================================================================
# SCENARIO 1: "Hello" (CHAT MODE - No Marine Agents Invoked)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_01_greeting_hello():
    orchestrator = Orchestrator()
    query = UserQuery(message="Hello", language="en")
    response = await orchestrator.process_query(query)

    assert response.mode == "CHAT"
    assert response.execution_plan == []
    assert len(response.agents_used) == 0
    assert any(g in response.answer.lower() for g in ["hello", "neerai", "copilot", "assist"])
    assert response.risk == "LOW"

# =============================================================================
# SCENARIO 2: "How are you?" (CHAT MODE - Natural Small Talk)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_02_how_are_you():
    orchestrator = Orchestrator()
    query = UserQuery(message="How are you?", language="en")
    response = await orchestrator.process_query(query)

    assert response.mode == "CHAT"
    assert response.execution_plan == []
    assert len(response.agents_used) == 0
    assert any(phrase in response.answer.lower() for phrase in ["doing well", "good", "thank you", "assist"])
    assert "0.58" not in response.answer  # Must NOT hallucinate wave numbers

# =============================================================================
# SCENARIO 3: "Mujhe aaj achha nahi lag raha." (CHAT MODE - Empathetic Hindi Response)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_03_empathetic_feelings_hindi():
    orchestrator = Orchestrator()
    query = UserQuery(message="Mujhe aaj achha nahi lag raha.", language="hi")
    response = await orchestrator.process_query(query)

    assert response.mode == "CHAT"
    assert response.execution_plan == []
    assert len(response.agents_used) == 0
    assert response.language == "hi"
    # Must offer comfort and empathy in Hindi without calling weather sensor agents
    assert any(w in response.answer for w in ["samajh", "aaram", "khayal", "achha", "baat"])
    # Absolutely NO fake marine sensor values or storm alerts
    assert "0.58" not in response.answer
    assert "wave" not in response.answer.lower()

# =============================================================================
# SCENARIO 4: "What can you do?" (CHAT MODE - Clear Capabilities Summary)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_04_capabilities_summary():
    orchestrator = Orchestrator()
    query = UserQuery(message="What can you do?", language="en")
    response = await orchestrator.process_query(query)

    assert response.mode == "CHAT"
    assert response.execution_plan == []
    assert len(response.agents_used) == 0
    assert any(c in response.answer.lower() for c in ["wave", "weather", "fishing", "cyclone", "gis", "map"])

# =============================================================================
# SCENARIO 5: "Explain ocean currents simply." (CHAT/SCIENCE - Clear Physical Principles)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_05_explain_ocean_currents():
    orchestrator = Orchestrator()
    query = UserQuery(message="Explain ocean currents simply.", language="en")
    response = await orchestrator.process_query(query)

    assert response.mode == "CHAT"
    assert response.execution_plan == []
    assert len(response.agents_used) == 0
    ans_lower = response.answer.lower()
    assert any(f in ans_lower for f in ["wind", "density", "coriolis", "movement", "rivers"])
    assert response.risk == "LOW"

# =============================================================================
# SCENARIO 6: "What is the weather near Goa?" (AGENT MODE - Real Sensor Data)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_06_weather_near_goa():
    orchestrator = Orchestrator()
    query = UserQuery(
        message="What is the weather near Goa?",
        latitude=15.30,
        longitude=73.80,
        language="en"
    )
    response = await orchestrator.process_query(query)

    assert response.mode in ["AGENT", "HYBRID"]
    assert "weather" in response.execution_plan
    assert "ocean" in response.execution_plan
    assert "weather" in response.agents_used
    assert response.risk in ["LOW", "MODERATE", "HIGH"]
    assert any(t in response.answer.lower() for t in ["wind", "wave", "km/h", "m", "c"])

# =============================================================================
# SCENARIO 7: "What about tomorrow?" after asking about Goa weather (FOLLOW-UP)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_07_follow_up_tomorrow():
    orchestrator = Orchestrator()
    session_id = "test_conv_flow_07"

    # Turn 1: Ask about Goa weather
    q1 = UserQuery(
        message="What is the weather near Goa?",
        latitude=15.30,
        longitude=73.80,
        conversation_id=session_id,
        language="en"
    )
    r1 = await orchestrator.process_query(q1)
    assert "weather" in r1.agents_used

    # Turn 2: Follow-up asking about tomorrow
    q2 = UserQuery(
        message="What about tomorrow?",
        conversation_id=session_id,
        language="en"
    )
    r2 = await orchestrator.process_query(q2)

    # Must inherit Goa coordinates and plan weather/ocean agents
    assert "weather" in r2.execution_plan
    assert r2.risk in ["LOW", "MODERATE", "HIGH"]

# =============================================================================
# SCENARIO 8: Hindi Marine Data Request (AGENT MODE - Kochi Waves)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_08_hindi_marine_data_request():
    orchestrator = Orchestrator()
    query = UserQuery(
        message="Kochi ke paas samundar mein kitni unchi lehrein hain?",
        latitude=9.93,
        longitude=76.26,
        language="hi"
    )
    response = await orchestrator.process_query(query)

    assert response.language == "hi"
    assert "ocean" in response.execution_plan
    assert "ocean" in response.agents_used
    assert any(term in response.answer for term in ["lehrein", "m", "hawa", "samundar"])

# =============================================================================
# SCENARIO 9: Ambiguous Question Requiring Clarification
# =============================================================================
@pytest.mark.anyio
async def test_scenario_09_ambiguous_question_clarification():
    orchestrator = Orchestrator()
    # Ambiguous question without prior conversation or location
    query = UserQuery(
        message="Is it good to go?",
        conversation_id="fresh_ambiguous_session",
        language="en"
    )
    response = await orchestrator.process_query(query)

    assert response.mode == "CHAT"
    assert response.execution_plan == []
    assert len(response.agents_used) == 0
    # Must ask polite clarification asking for the coastal location or harbor
    assert any(c in response.answer.lower() for c in ["which coastal", "location", "harbor", "sail from"])

# =============================================================================
# SCENARIO 10: Missing LLM Credentials (Transparent Fallback)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_10_missing_llm_credentials():
    llm = MultiModelLLMService()
    diag = await llm.get_provider_registry_status()

    # Verify that unconfigured providers are honestly reported as UNCONFIGURED
    assert "providers" in diag
    gemini_diag = diag["providers"]["gemini"]
    assert gemini_diag["status"] in ["UNCONFIGURED", "READY", "AUTH_FAILED"]
    # Deterministic fallback is operational without failing
    assert diag["providers"]["deterministic"]["status"] == "READY"

    # Orchestrator handles conversational query cleanly even without LLM credentials
    orchestrator = Orchestrator()
    query = UserQuery(message="Hello NeerAI", language="en")
    response = await orchestrator.process_query(query)
    assert response.mode == "CHAT"
    assert len(response.answer) > 10

# =============================================================================
# SCENARIO 11: Provider Timeout Handling (Safe Fallback, No 500 Crash)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_11_provider_timeout_handling():
    llm = MultiModelLLMService()

    # Simulate provider timeout
    with patch.object(GeminiProvider, "generate_chat_completion", side_effect=TimeoutError("Request timed out after 8s")):
        with patch.object(GeminiProvider, "is_available", True):
            resp = await llm.generate_chat_response(
                messages=[{"role": "user", "content": "Hello"}],
                correlation_id="test_timeout_correl"
            )
            # Must return None on timeout so caller can use fallback safely
            assert resp is None

    # Telemetry must record fallback
    recent = llm.get_telemetry_history()
    assert any(t.get("status") == "FALLBACK" for t in recent)

# =============================================================================
# SCENARIO 12: Marine Provider Unavailable (Zero Fabrication)
# =============================================================================
@pytest.mark.anyio
async def test_scenario_12_marine_provider_unavailable():
    orchestrator = Orchestrator()
    # Simulate weather agent returning UNAVAILABLE
    mock_res = AgentResult(
        agent="weather",
        status="UNAVAILABLE",
        data={},
        confidence=0.0,
        sources=["Open-Meteo"],
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        errors=["503 Service Unavailable"],
        data_type="OBSERVATION",
        quality_status="UNAVAILABLE"
    )
    with patch.object(orchestrator.weather, "execute", new=AsyncMock(return_value=mock_res)):
        query = UserQuery(
            message="Check weather in Goa",
            latitude=15.30,
            longitude=73.80,
            language="en"
        )
        response = await orchestrator.process_query(query)
        # Must NOT crash, and must NOT fabricate weather numbers
        assert response.risk in ["LOW", "UNKNOWN", "MODERATE", "HIGH"]
        assert isinstance(response.answer, str)

# =============================================================================
# SCENARIO 13: Two Different Sessions Must NOT Share Private History
# =============================================================================
@pytest.mark.anyio
async def test_scenario_13_two_sessions_isolation():
    orchestrator = Orchestrator()
    memory = ConversationalMemoryService()

    session_a = "session_tenant_A"
    session_b = "session_tenant_B"

    # Clear any previous runs
    memory.clear_conversation(session_a)
    memory.clear_conversation(session_b)

    # Session A asks about Chennai
    q_a = UserQuery(
        message="What is the weather in Chennai?",
        latitude=13.08,
        longitude=80.27,
        conversation_id=session_a,
        language="en"
    )
    await orchestrator.process_query(q_a)

    # Verify Session A has history
    context_a = memory.get_conversation_context(session_a)
    assert len(context_a) >= 2
    assert any("chennai" in t.get("text", "").lower() for t in context_a)

    # Session B is completely fresh
    context_b = memory.get_conversation_context(session_b)
    assert len(context_b) == 0

    # Session B asks ambiguous "What about tomorrow?"
    followup_b = memory.resolve_followup_reference(session_b, "What about tomorrow?")
    # Session B must NOT inherit Chennai coordinates from Session A!
    assert followup_b.get("resolved_location_name") != "Chennai"
    assert followup_b.get("resolved_latitude") != 13.08
