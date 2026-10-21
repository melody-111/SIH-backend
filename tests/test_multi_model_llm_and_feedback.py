import pytest
import os
import uuid
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.db.database import DatabaseManager
from app.services.llm_service import MultiModelLLMService, redact_secrets, OpenAIProvider, AnthropicProvider, GeminiProvider
from app.services.conversational_memory import ConversationalMemoryService
from app.services.feedback_governance import ControlledQualityImprovementEngine
from app.schemas.llm import TaskType, IntentClassificationOutput, SynthesizedAdvisoryOutput

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_memory():
    """Reset DB stores before each test for deterministic isolation."""
    mem = ConversationalMemoryService()
    mem.clear_all_for_test()

# =============================================================================
# 1. Multi-Model LLM Provider Abstraction & Telemetry Tests
# =============================================================================

def test_llm_secret_redaction():
    text_with_keys = "Error with key sk-proj-1234567890abcdef123456 and token AIzaSyD9876543210zyxwvutsrq"
    redacted = redact_secrets(text_with_keys)
    assert "sk-[REDACTED]" in redacted
    assert "AIza[REDACTED]" in redacted
    assert "sk-proj-1234567890abcdef" not in redacted

def test_provider_availability_and_fallback():
    service = MultiModelLLMService()
    prov_name, prov = service.resolve_provider()
    # In test environment without external API keys, must resolve safely to deterministic fallback
    assert prov_name in ["deterministic", "openai", "anthropic", "gemini"]
    assert prov.is_available is True

def test_missing_credentials_handling():
    # Instantiating providers without keys must not crash
    openai_prov = OpenAIProvider()
    anthropic_prov = AnthropicProvider()
    gemini_prov = GeminiProvider()

    assert isinstance(openai_prov.is_available, bool)
    assert isinstance(anthropic_prov.is_available, bool)
    assert isinstance(gemini_prov.is_available, bool)

@pytest.mark.anyio
async def test_llm_task_routing_and_telemetry():
    service = MultiModelLLMService()
    correlation_id = f"test_corr_{uuid.uuid4().hex[:6]}"

    # Task 1: Intent classification
    res = await service.classify_intent("Are there cyclones near Goa?", correlation_id=correlation_id)
    assert isinstance(res, IntentClassificationOutput)
    assert res.intent in ["HAZARD", "MARINE_CONDITION"]

    # Verify diagnostic telemetry record was captured
    telemetry = service.get_telemetry_history()
    assert len(telemetry) > 0
    recent = telemetry[-1]
    assert recent["correlation_id"] == correlation_id
    assert recent["task"] == TaskType.INTENT_CLASSIFICATION.value
    assert recent["status"] in ["SUCCESS", "FALLBACK"]
    assert "duration_ms" in recent

# =============================================================================
# 2. Database Migration 002 & Persistent User Feedback Tests
# =============================================================================

def test_database_migration_002_health():
    db = DatabaseManager()
    health = db.check_db_health()
    assert health["status"] == "healthy"
    assert health["migrations_applied"] >= 2
    assert health["integrity"] == "ok"

def test_feedback_persistence_and_retrieval():
    user_id = f"user_{uuid.uuid4().hex[:6]}"
    message_id = f"msg_{uuid.uuid4().hex[:6]}"
    corr_id = f"corr_{uuid.uuid4().hex[:6]}"

    # Submit feedback
    resp_post = client.post(
        "/api/memory/feedback",
        headers={"X-User-ID": user_id},
        json={
            "message_id": message_id,
            "correlation_id": corr_id,
            "rating": 5,
            "category": "ACCURACY",
            "feedback_notes": "Wave forecast matched exactly what I saw at the harbor.",
            "consent_given": True
        }
    )
    assert resp_post.status_code == 200
    data = resp_post.json()
    assert data["status"] == "RECORDED"
    fb_id = data["feedback"]["feedback_id"]

    # List user feedback (tenant isolation)
    resp_get = client.get(
        "/api/memory/feedback",
        headers={"X-User-ID": user_id}
    )
    assert resp_get.status_code == 200
    feedbacks = resp_get.json()["feedback"]
    assert len(feedbacks) == 1
    assert feedbacks[0]["feedback_id"] == fb_id
    assert feedbacks[0]["rating"] == 5
    assert feedbacks[0]["correlation_id"] == corr_id

    # Another user cannot see this feedback
    other_user = f"user_other_{uuid.uuid4().hex[:6]}"
    resp_other = client.get(
        "/api/memory/feedback",
        headers={"X-User-ID": other_user}
    )
    assert resp_other.status_code == 200
    assert len(resp_other.json()["feedback"]) == 0

def test_feedback_privacy_guard_rejects_credentials():
    user_id = "user_sec_test"
    resp = client.post(
        "/api/memory/feedback",
        headers={"X-User-ID": user_id},
        json={
            "message_id": "msg_sec_01",
            "rating": 4,
            "feedback_notes": "My api_key is sk-123456789 and password is secret",
            "consent_given": True
        }
    )
    assert resp.status_code == 400
    assert "Privacy Policy Violation" in resp.json()["detail"]

def test_feedback_rating_bounds_validation():
    user_id = "user_bounds_test"
    # Rating 6 is out of bounds (1-5 allowed)
    resp = client.post(
        "/api/memory/feedback",
        headers={"X-User-ID": user_id},
        json={"message_id": "msg_001", "rating": 6, "consent_given": True}
    )
    assert resp.status_code == 422

    # Rating 0 is out of bounds
    resp_zero = client.post(
        "/api/memory/feedback",
        headers={"X-User-ID": user_id},
        json={"message_id": "msg_001", "rating": 0, "consent_given": True}
    )
    assert resp_zero.status_code == 422

def test_feedback_deletion_and_tenant_isolation():
    user_a = "user_owner"
    user_b = "user_intruder"

    # User A creates feedback
    res = client.post(
        "/api/memory/feedback",
        headers={"X-User-ID": user_a},
        json={"message_id": "msg_del_01", "rating": 4, "feedback_notes": "Good", "consent_given": True}
    )
    fb_id = res.json()["feedback"]["feedback_id"]

    # User B attempts to delete User A's feedback
    del_b = client.delete(f"/api/memory/feedback/{fb_id}", headers={"X-User-ID": user_b})
    assert del_b.status_code == 404

    # User A successfully deletes their own feedback
    del_a = client.delete(f"/api/memory/feedback/{fb_id}", headers={"X-User-ID": user_a})
    assert del_a.status_code == 200
    assert del_a.json()["status"] == "success"

# =============================================================================
# 3. Controlled Quality Improvement Loop & Anonymized Export Tests
# =============================================================================

def test_controlled_governance_and_export():
    gov = ControlledQualityImprovementEngine()
    policy = gov.get_governance_policy()
    assert policy["automated_prompt_mutation"] == "DISABLED"
    assert policy["automated_weight_fine_tuning"] == "DISABLED"
    assert policy["private_conversation_training"] == "PROHIBITED"

    # Seed 2 feedback submissions: 1 with consent, 1 without consent
    client.post(
        "/api/memory/feedback",
        headers={"X-User-ID": "user_consent"},
        json={"message_id": "msg_c1", "rating": 2, "feedback_notes": "Wave height was off by 0.5m", "consent_given": True}
    )
    client.post(
        "/api/memory/feedback",
        headers={"X-User-ID": "user_no_consent"},
        json={"message_id": "msg_c2", "rating": 1, "feedback_notes": "No consent given", "consent_given": False}
    )

    # Export for evaluation benchmark
    resp_exp = client.get("/api/memory/feedback/export")
    assert resp_exp.status_code == 200
    export_data = resp_exp.json()
    assert export_data["status"] == "success"
    samples = export_data["evaluation_samples"]
    # Only the consented submission can be exported
    assert len(samples) == 1
    assert "user_consent" not in str(samples[0])  # User ID must NOT be leaked
    assert samples[0]["rating"] == 2
