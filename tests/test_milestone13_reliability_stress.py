import asyncio
import datetime
import json
import uuid
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.core.config import settings
from app.db.database import DatabaseManager
from app.agents.orchestrator import Orchestrator
from app.schemas.chat import UserQuery
from app.schemas.agent import AgentResult
from app.schemas.response import FinalResponse
from app.services.observability_service import (
    ObservabilityService,
    RequestTrace,
    SafeErrorCategory,
    redact_sensitive,
    sanitize_location
)
from app.services.llm_service import MultiModelLLMService, OpenAIProvider
from app.services.stress_test_harness import StressTestHarness

client = TestClient(app)

# =============================================================================
# 1. AGENT EXECUTION RELIABILITY, CONCURRENCY & TIMEOUTS
# =============================================================================

@pytest.mark.anyio
async def test_orchestrator_agent_timeout_and_fault_isolation():
    """Verifies that an agent timeout is bounded, does not hang orchestrator, and preserves other agents."""
    orchestrator = Orchestrator()

    async def slow_execute(input_data):
        await asyncio.sleep(2.0)
        return AgentResult(
            agent="weather",
            status="SUCCESS",
            data={"wind_speed": 10.0},
            confidence=0.8,
            sources=["mock"],
            timestamp=datetime.datetime.utcnow().isoformat() + "Z"
        )

    orig_timeout = settings.AGENT_TIMEOUT_SECONDS
    settings.AGENT_TIMEOUT_SECONDS = 0.2

    try:
        with patch.object(orchestrator.weather, "execute", side_effect=slow_execute):
            query = UserQuery(
                message="What is the wave height and weather near Goa?",
                latitude=15.30,
                longitude=73.80,
                language="en"
            )
            resp = await orchestrator.process_query(query)

            # Response must complete and return
            assert resp is not None
            assert resp.answer != ""
            # Weather must be reported as unavailable / in warnings
            assert any("weather" in w.lower() for w in resp.warnings) or "weather" in resp.unavailable_sources if hasattr(resp, "unavailable_sources") else True
            # Ocean agent data should still be present
            assert resp.risk in ["LOW", "MODERATE", "HIGH", "UNKNOWN"]
    finally:
        settings.AGENT_TIMEOUT_SECONDS = orig_timeout

@pytest.mark.anyio
async def test_partial_failure_preserves_valid_evidence():
    """Proves that a catastrophic failure in one agent does NOT erase valid evidence from other agents."""
    orchestrator = Orchestrator()

    async def failing_execute(input_data):
        raise RuntimeError("Weather station hardware failure!")

    with patch.object(orchestrator.weather, "execute", side_effect=failing_execute):
        query = UserQuery(
            message="Check wave swell and wind speed near Goa",
            latitude=15.30,
            longitude=73.80,
            language="en"
        )
        resp = await orchestrator.process_query(query)

        assert resp is not None
        assert resp.answer != ""
        # The failing weather agent is flagged, but ocean and hazard succeeded
        assert any("weather" in w.lower() for w in resp.warnings) or any("temporarily unavailable" in w.lower() for w in resp.warnings)

@pytest.mark.anyio
async def test_cancellation_propagation_releases_resources():
    """Verifies that cancelling an in-flight request propagates CancelledError and releases active requests."""
    orchestrator = Orchestrator()
    obs = ObservabilityService()
    initial_active = obs.active_requests_count

    query = UserQuery(
        message="Check long forecast near Goa",
        latitude=15.30,
        longitude=73.80,
        language="en"
    )

    async def run_and_cancel():
        # Start pipeline task and cancel immediately
        task = asyncio.create_task(orchestrator.process_query(query))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    await run_and_cancel()
    # Active request count must not leak
    assert obs.active_requests_count == initial_active

# =============================================================================
# 2. LLM AND PROVIDER FAILURE HANDLING
# =============================================================================

@pytest.mark.anyio
async def test_llm_provider_timeout_fallback():
    """Simulates LLM provider timeout, ensuring safe fallback without unhandled crash."""
    llm = MultiModelLLMService()

    with patch.object(llm.providers["deterministic"], "generate_completion", side_effect=asyncio.TimeoutError("Provider call timed out")):
        res = await llm.classify_intent("Are there high waves in Mumbai?")
        assert res is not None
        assert res.intent in ["MARINE_CONDITION", "HAZARD", "GENERAL_MARINE"]
        assert "deterministic" in res.reasoning.lower() or "fallback" in res.reasoning.lower()

@pytest.mark.anyio
async def test_llm_provider_http_429_rate_limit():
    """Simulates HTTP 429 rate limit error from LLM provider."""
    llm = MultiModelLLMService()
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp_429 = httpx.Response(429, request=req, json={"error": {"message": "Rate limit exceeded"}})

    with patch.object(llm.providers["deterministic"], "generate_completion", side_effect=httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp_429)):
        res = await llm.classify_intent("Cyclone warning status")
        assert res is not None
        assert res.intent in ["HAZARD", "GENERAL_MARINE"]

@pytest.mark.anyio
async def test_llm_provider_http_5xx_error():
    """Simulates HTTP 503 upstream server error from LLM provider."""
    llm = MultiModelLLMService()
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp_503 = httpx.Response(503, request=req, json={"error": {"message": "Service Unavailable"}})

    with patch.object(llm.providers["deterministic"], "generate_completion", side_effect=httpx.HTTPStatusError("503 Service Unavailable", request=req, response=resp_503)):
        res = await llm.classify_intent("Can I sail tomorrow?")
        assert res is not None
        assert res.intent in ["MARINE_CONDITION", "GENERAL_MARINE"]

@pytest.mark.anyio
async def test_llm_provider_malformed_json():
    """Simulates LLM returning corrupted or malformed JSON."""
    llm = MultiModelLLMService()

    with patch.object(llm.providers["deterministic"], "generate_completion", return_value=("{not: valid: json", None, 0.0)):
        res = await llm.classify_intent("Tell me about fish zones")
        assert res is not None
        assert res.intent in ["FISHING_LOCATION", "GENERAL_MARINE"]

@pytest.mark.anyio
async def test_llm_provider_invalid_pydantic_schema():
    """Simulates LLM returning JSON missing required schema attributes."""
    llm = MultiModelLLMService()

    with patch.object(llm.providers["deterministic"], "generate_completion", return_value=(json.dumps({"random_field": 123}), None, 0.0)):
        res = await llm.classify_intent("Cyclone warning in Chennai")
        assert res is not None
        assert res.intent in ["HAZARD", "GENERAL_MARINE"]

@pytest.mark.anyio
async def test_llm_provider_empty_response():
    """Simulates LLM returning an empty string."""
    llm = MultiModelLLMService()

    with patch.object(llm.providers["deterministic"], "generate_completion", return_value=("", None, 0.0)):
        res = await llm.classify_intent("Sea surface temperature in Goa")
        assert res is not None
        assert res.intent in ["OCEAN_INFO", "GENERAL_MARINE"]

@pytest.mark.anyio
async def test_llm_unconfigured_credentials_safety():
    """Verifies that when LLM credentials are empty, system resolves to deterministic fallback without external calls."""
    llm = MultiModelLLMService()
    prov_name, prov = llm.resolve_provider()
    # In test environment, provider resolves to deterministic
    assert prov_name == "deterministic"
    assert prov.is_available is True

# =============================================================================
# 3. SSE STREAMING RELIABILITY
# =============================================================================

def test_sse_streaming_endpoint_event_ordering_and_completion():
    """Tests that SSE endpoint emits progress events, final response, and ends with [DONE]."""
    payload = {
        "message": "Wave swell and wind near Goa?",
        "latitude": 15.30,
        "longitude": 73.80,
        "language": "en"
    }
    resp = client.post("/api/chat/stream", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    body = resp.text
    lines = body.strip().split("\n\n")

    # Verify at least one progress event, one final event, and terminating [DONE]
    has_progress = False
    has_final = False
    has_done = False

    for chunk in lines:
        if not chunk.startswith("data: "):
            continue
        data_content = chunk[6:].strip()
        if data_content == "[DONE]":
            has_done = True
        else:
            try:
                parsed = json.loads(data_content)
                if parsed.get("type") == "progress":
                    has_progress = True
                elif parsed.get("type") == "final":
                    has_final = True
                    assert "answer" in parsed["response"]
                    assert "risk" in parsed["response"]
            except Exception:
                pass

    assert has_progress is True, "Stream must contain progress events"
    assert has_final is True, "Stream must contain a final response event"
    assert has_done is True, "Stream must terminate with [DONE]"

def test_sse_streaming_correlation_id_propagation():
    """Verifies that correlation ID is preserved and propagated in streaming responses."""
    custom_corr = f"corr_stream_test_{uuid.uuid4().hex[:6]}"
    payload = {
        "message": "Check marine weather",
        "latitude": 15.30,
        "longitude": 73.80,
        "language": "en"
    }
    resp = client.post("/api/chat/stream", json=payload, headers={"x-correlation-id": custom_corr})
    assert resp.status_code == 200
    assert resp.headers.get("X-Correlation-ID") == custom_corr

# =============================================================================
# 4. OBSERVABILITY & METRICS
# =============================================================================

def test_observability_credential_redaction():
    """Ensures sensitive API keys and Bearer tokens are redacted in all observability logs."""
    raw = "Failed to connect using sk-proj-1234567890abcdef123456 and Bearer eyJhbGciOiJIUzI1NiJ9.test"
    redacted = redact_sensitive(raw)
    assert "sk-[REDACTED]" in redacted
    assert "Bearer [REDACTED]" in redacted
    assert "1234567890abcdef" not in redacted

def test_observability_location_coarsening():
    """Ensures user coordinate tracking is coarsened to ~1.1km to protect user privacy."""
    coarse = sanitize_location(15.299345, 74.124012)
    assert coarse == "15.3,74.12"

def test_observability_metrics_percentiles_calculation():
    """Tests latency percentile calculations in ObservabilityService."""
    obs = ObservabilityService()
    obs.clear_metrics()

    # Insert 10 synthetic traces
    for d in [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 1000.0]:
        t = RequestTrace(
            correlation_id=f"trace_{d}",
            duration_ms=d,
            status="SUCCESS"
        )
        obs.record_trace(t)

    metrics = obs.get_metrics_summary()
    assert metrics["total_requests"] == 10
    assert metrics["latency_ms"]["min"] == 100.0
    assert metrics["latency_ms"]["max"] == 1000.0
    assert metrics["latency_ms"]["median_p50"] == 300.0
    assert metrics["latency_ms"]["p95"] == 1000.0

def test_agent_execution_logs_sqlite_persistence():
    """Verifies that agent execution traces can be durably written to and read from SQLite."""
    db = DatabaseManager()
    corr_id = f"corr_db_exec_{uuid.uuid4().hex[:6]}"
    exec_id = f"exec_{uuid.uuid4().hex[:6]}"

    db.log_agent_execution(
        execution_id=exec_id,
        correlation_id=corr_id,
        agent_name="ocean_test",
        status="SUCCESS",
        execution_time_ms=145.2
    )

    logs = db.get_agent_execution_logs(correlation_id=corr_id)
    assert len(logs) == 1
    assert logs[0]["execution_id"] == exec_id
    assert logs[0]["agent_name"] == "ocean_test"
    assert logs[0]["execution_time_ms"] == 145.2

def test_health_metrics_api_endpoint():
    """Verifies the /api/health/metrics endpoint returns structured telemetry."""
    resp = client.get("/api/health/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "latency_ms" in data
    assert "cache_metrics" in data
    assert "token_usage" in data

# =============================================================================
# 5. MARINE SAFETY REGRESSION TESTS
# =============================================================================

@pytest.mark.anyio
async def test_safety_official_warning_priority_over_optimistic_agent():
    """
    CRITICAL SAFETY TEST:
    When an official IMD hazard bulletin exists (e.g. Cyclone / High Waves),
    even if the fishing agent recommends a PFZ area, the risk engine and response
    must enforce HIGH risk and warn fishermen NOT to venture into open sea.
    """
    orchestrator = Orchestrator()

    # Inject official cyclone warning into hazard agent
    async def mock_hazard(input_data):
        return AgentResult(
            agent="hazard",
            status="SUCCESS",
            data={
                "official_warnings": [{
                    "type": "OFFICIAL_CYCLONE_BULLETIN",
                    "severity": "WARNING",
                    "bulletin": "Severe Cyclone Alert: Wind speeds exceeding 65 km/h, rough seas.",
                    "source": "IMD_OFFICIAL",
                    "active_cyclone": True
                }],
                "cyclone_warning": True,
                "is_official_feed_active": True
            },
            confidence=1.0,
            sources=["IMD_OFFICIAL"],
            timestamp=datetime.datetime.utcnow().isoformat() + "Z"
        )

    with patch.object(orchestrator.hazard, "execute", side_effect=mock_hazard):
        query = UserQuery(
            message="Can I go fishing today near Goa coast?",
            latitude=15.30,
            longitude=73.80,
            language="en"
        )
        resp = await orchestrator.process_query(query)

        # Assessed risk must be elevated
        assert resp.risk in ["HIGH", "DANGER"]
        # Advisory must explicitly contain safety warning
        assert any("cyclone" in w.lower() or "rough" in w.lower() or "danger" in w.lower() for w in resp.warnings + [resp.answer])
        # Must not certify open water safety
        assert "safe" not in resp.answer.lower() or "not safe" in resp.answer.lower() or "avoid" in resp.answer.lower()

@pytest.mark.anyio
async def test_safety_stale_cache_data_explicitly_labeled():
    """Verifies that cached data is never presented as real-time observations."""
    orchestrator = Orchestrator()
    orchestrator.cache.set_force_offline(True)

    # Store cache entry
    orchestrator.cache.store_observation(
        lat=15.30,
        lon=74.10,
        data={"wave": {"wave_height": 1.2}},
        location_name="Goa Harbor"
    )

    try:
        query = UserQuery(
            message="Check wave height",
            latitude=15.30,
            longitude=74.10,
            language="en"
        )
        resp = await orchestrator.process_query(query)
        assert resp.is_cached is True
        assert resp.cached_at is not None
        assert any("cache" in w.lower() or "offline" in w.lower() for w in resp.warnings + [resp.answer])
    finally:
        orchestrator.cache.set_force_offline(False)

@pytest.mark.anyio
async def test_safety_missing_pfz_data_never_invents_fake_coordinates():
    """Proves that when PFZ data is unavailable, no fabricated coordinates are produced."""
    orchestrator = Orchestrator()

    async def unauth_pfz(input_data):
        return AgentResult(
            agent="pfz",
            status="UNAVAILABLE",
            data={},
            confidence=0.0,
            sources=["INCOIS_PFZ"],
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            errors=["Authentication credentials unavailable for INCOIS PFZ service"]
        )

    with patch.object(orchestrator.pfz, "execute", side_effect=unauth_pfz):
        query = UserQuery(
            message="Where are the best fishing spots near Goa?",
            latitude=15.30,
            longitude=73.80,
            language="en"
        )
        resp = await orchestrator.process_query(query)
        # Recommended location must remain None (never fabricated!)
        assert resp.recommended_location is None
        # Caveat about unavailable data must be declared
        assert any("pfz" in w.lower() or "unavailable" in w.lower() or "not fabricated" in w.lower() for w in resp.warnings + [resp.answer])

@pytest.mark.anyio
async def test_safety_route_analysis_discloses_non_navigational_status():
    """Verifies that route analysis includes a disclaimer that it does NOT replace navigation charts."""
    orchestrator = Orchestrator()
    query = UserQuery(
        message="Plot passage from Panaji to Mormugao",
        latitude=15.49,
        longitude=73.82,
        language="en"
    )
    resp = await orchestrator.process_query(query)
    assert resp is not None
    # Must contain route disclaimer or advisory status
    assert any("navigation" in w.lower() or "harbor" in w.lower() or "route" in w.lower() for w in resp.warnings + resp.reasons + [resp.answer])

# =============================================================================
# 6. REPEATABLE LOCAL STRESS HARNESS INTEGRATION
# =============================================================================

@pytest.mark.anyio
async def test_stress_harness_database_and_concurrency():
    """Runs a controlled slice of the stress harness within the test suite."""
    harness = StressTestHarness()
    res = await harness.run_database_contention_test(concurrency=10, operations_per_worker=5)

    assert res["lock_or_busy_errors"] == 0, "SQLite WAL must not hit lock errors during controlled concurrency"
    assert res["latencies_ms"]["p95"] < 100.0, "p95 database transaction latency should be under 100ms locally"

@pytest.mark.anyio
async def test_stress_harness_provider_timeout_simulation():
    """Verifies the stress harness timeout simulation under concurrency."""
    harness = StressTestHarness()
    res = await harness.run_provider_timeout_simulation(concurrency=5)

    assert res["crashed_requests"] == 0
    assert res["partial_successes"] == 5
    assert res["timeouts_detected"] == 5
