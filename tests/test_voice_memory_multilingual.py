import pytest
import datetime
from fastapi.testclient import TestClient
from app.main import app
from app.services.language_service import LanguageService
from app.services.voice_service import VoiceService
from app.services.conversational_memory import ConversationalMemoryService

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_memory():
    """Reset conversational memory stores before each test."""
    ConversationalMemoryService().clear_all_for_test()
    yield
    ConversationalMemoryService().clear_all_for_test()

# -----------------------------------------------------------------------------
# 1. Language Detection & Capability Tests
# -----------------------------------------------------------------------------
def test_language_capabilities():
    response = client.get("/api/voice/languages")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    languages = data["languages"]
    codes = [l["code"] for l in languages]
    assert "en" in codes
    assert "hi" in codes
    assert "ta" in codes
    assert "te" in codes
    assert "bn" in codes

def test_language_detection():
    # Devanagari Hindi
    res_hi = client.post("/api/voice/detect-language", json={"text": "कल समुद्र में लहरें कैसी रहेंगी?"})
    assert res_hi.status_code == 200
    assert res_hi.json()["language"] == "hi"

    # Romanized Hindi keywords
    res_rom = client.post("/api/voice/detect-language", json={"text": "wahan lehrein aur hawa kaisa hai"})
    assert res_rom.status_code == 200
    assert res_rom.json()["language"] == "hi"

    # English
    res_en = client.post("/api/voice/detect-language", json={"text": "What are the marine wave heights near Goa?"})
    assert res_en.status_code == 200
    assert res_en.json()["language"] == "en"

# -----------------------------------------------------------------------------
# 2. Text-to-Speech Sanitization (Preventing Diagnostic Dumps from Being Spoken)
# -----------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tts_sanitization():
    voice_svc = VoiceService()
    text_with_diagnostics = """
    Waves are 1.5 meters along Goa coast.
    ```json
    {"tool": "fetch_wave_data", "status": "200 OK"}
    ```
    [INTERNAL_LOG: req_12345] [CORRELATION_ID: abc]
    http://internal-telemetry.orca/debug
    Safe for small artisanal vessels.
    """
    clean_result = await voice_svc.synthesize_speech(text_with_diagnostics, language="en")
    assert clean_result["success"] is True
    spoken = clean_result["spoken_text"]
    assert "fetch_wave_data" not in spoken
    assert "INTERNAL_LOG" not in spoken
    assert "http://" not in spoken
    assert "Waves are 1.5 meters" in spoken
    assert "Safe for small artisanal vessels." in spoken

# -----------------------------------------------------------------------------
# 3. Speech-to-Text Reporting Unconfigured Provider Cleanly
# -----------------------------------------------------------------------------
def test_stt_unconfigured_reporting():
    response = client.post("/api/voice/stt", json={"audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA", "language": "hi"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UNCONFIGURED"
    assert "SARVAM_API_KEY missing" in data["message"]

# -----------------------------------------------------------------------------
# 4. Conversational Memory: Follow-Up Resolution (Temporal & Spatial)
# -----------------------------------------------------------------------------
def test_conversational_followup_temporal_and_spatial():
    memory = ConversationalMemoryService()
    conv_id = "test_conv_goa_01"

    # Turn 1: User asks about Goa
    memory.add_turn(
        conversation_id=conv_id,
        role="user",
        text="What are the wave conditions in Goa?",
        intent="MARINE_CONDITION",
        latitude=15.3493,
        longitude=73.8000,
        location_name="Goa Coast"
    )
    memory.add_turn(
        conversation_id=conv_id,
        role="assistant",
        text="Waves are currently 1.4 meters in Goa with light winds.",
        intent="MARINE_CONDITION",
        latitude=15.3493,
        longitude=73.8000
    )

    # Turn 2: Follow-up temporal question: "What about tomorrow?"
    res_followup = memory.resolve_followup_reference(
        conversation_id=conv_id,
        query_text="What about tomorrow?",
        current_lat=15.3493,
        current_lon=73.8000
    )
    assert res_followup["is_followup"] is True
    assert res_followup["needs_clarification"] is False
    assert res_followup["resolved_latitude"] == 15.3493
    assert res_followup["resolved_longitude"] == 73.8000
    # Tomorrow is verified
    res_dt = datetime.datetime.fromisoformat(res_followup["resolved_time_iso"])
    now = datetime.datetime.now(datetime.timezone.utc)
    assert (res_dt - now).total_seconds() > 3600 * 20  # ~24 hours ahead

    # Turn 3: Spatial follow-up: "wahan hawa kaisi hai?"
    res_spatial = memory.resolve_followup_reference(
        conversation_id=conv_id,
        query_text="wahan hawa kaisi hai?",
        current_lat=0.0,
        current_lon=0.0
    )
    assert res_spatial["is_followup"] is True
    assert res_spatial["resolved_latitude"] == 15.3493
    assert res_spatial["resolved_longitude"] == 73.8000

def test_ambiguous_followup_clarification():
    memory = ConversationalMemoryService()
    conv_id = "test_empty_session"

    # Relative query with no prior context or location
    res = memory.resolve_followup_reference(
        conversation_id=conv_id,
        query_text="What about there?",
        current_lat=None,
        current_lon=None
    )
    assert res["is_followup"] is True
    assert res["needs_clarification"] is True
    assert "Which coastal location" in res["clarification_prompt"]

# -----------------------------------------------------------------------------
# 5. User Preferences Persistence
# -----------------------------------------------------------------------------
def test_user_preferences_crud():
    uid = "sailor_mormugao"

    # Default lookup
    res_def = client.get("/api/memory/preferences", headers={"X-User-ID": uid})
    assert res_def.status_code == 200
    assert res_def.json()["preferred_language"] == "en"

    # Update preferences
    res_up = client.put(
        "/api/memory/preferences",
        headers={"X-User-ID": uid},
        json={
            "preferred_language": "hi",
            "saved_locations": [{"id": "loc_1", "name": "Mormugao Port", "latitude": 15.40, "longitude": 73.80}],
            "enable_spoken_responses": True
        }
    )
    assert res_up.status_code == 200
    assert res_up.json()["preferred_language"] == "hi"

    # Verify persistence
    res_get = client.get("/api/memory/preferences", headers={"X-User-ID": uid})
    assert res_get.status_code == 200
    assert res_get.json()["preferred_language"] == "hi"
    assert len(res_get.json()["saved_locations"]) == 1

# -----------------------------------------------------------------------------
# 6. User-Approved Long-Term Memory CRUD & Privacy Guard
# -----------------------------------------------------------------------------
def test_explicit_user_memory_crud_and_privacy_guard():
    uid = "sailor_veraval"

    # Create memory
    res_create = client.post(
        "/api/memory/memories",
        headers={"X-User-ID": uid},
        json={"title": "Primary Boat Info", "content": "Operating 12m wooden mechanized trawler out of Veraval.", "category": "VESSEL"}
    )
    assert res_create.status_code == 200
    mem_id = res_create.json()["memory_id"]
    assert mem_id.startswith("mem_")

    # List memories
    res_list = client.get("/api/memory/memories", headers={"X-User-ID": uid})
    assert res_list.status_code == 200
    assert len(res_list.json()) == 1

    # Update memory
    res_up = client.put(
        f"/api/memory/memories/{mem_id}",
        headers={"X-User-ID": uid},
        json={"title": "Primary Boat Info (Updated)", "content": "Operating 14m trawler."}
    )
    assert res_up.status_code == 200
    assert res_up.json()["title"] == "Primary Boat Info (Updated)"

    # Privacy Policy Guard: Prohibit saving passwords or API keys
    res_bad = client.post(
        "/api/memory/memories",
        headers={"X-User-ID": uid},
        json={"title": "My Secret", "content": "Secret API_KEY=sk_test_123456", "category": "SECRET"}
    )
    assert res_bad.status_code == 400
    assert "Privacy Policy Violation" in res_bad.json()["detail"]

    # Delete memory
    res_del = client.delete(f"/api/memory/memories/{mem_id}", headers={"X-User-ID": uid})
    assert res_del.status_code == 200
    assert len(client.get("/api/memory/memories", headers={"X-User-ID": uid}).json()) == 0

def test_memory_tenant_isolation():
    # User 1 adds memory
    client.post(
        "/api/memory/memories",
        headers={"X-User-ID": "user_one"},
        json={"title": "User One Note", "content": "Note 1"}
    )

    # User 2 queries memories
    res_u2 = client.get("/api/memory/memories", headers={"X-User-ID": "user_two"})
    assert res_u2.status_code == 200
    assert len(res_u2.json()) == 0

# -----------------------------------------------------------------------------
# 7. Feedback & Governance
# -----------------------------------------------------------------------------
def test_feedback_governance():
    res_fb = client.post(
        "/api/memory/feedback",
        headers={"X-User-ID": "user_gov"},
        json={"message_id": "msg_001", "rating": 5, "feedback_notes": "Accurate wave prediction.", "consent_given": True}
    )
    assert res_fb.status_code == 200
    assert res_fb.json()["status"] == "RECORDED"

# -----------------------------------------------------------------------------
# 8. Real SSE Streaming & Turn Recording
# -----------------------------------------------------------------------------
def test_real_sse_streaming_and_turn_recording():
    conv_id = "test_streaming_session"
    res = client.post(
        "/api/chat/stream",
        json={"message": "Is there any hazard in Goa today?", "latitude": 15.3493, "longitude": 73.8000, "conversation_id": conv_id, "language": "en"}
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    text_content = res.text
    assert "data: " in text_content
    assert "Understanding query" in text_content
    assert "Evaluating maritime safety" in text_content

    # Check that turn was recorded in conversational memory
    ctx_res = client.get(f"/api/memory/context/{conv_id}")
    assert ctx_res.status_code == 200
    assert ctx_res.json()["count"] >= 2
