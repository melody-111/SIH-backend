import pytest
import datetime
from typing import Dict, Any
from fastapi.testclient import TestClient

from app.main import app
from app.core.agent_config import (
    AgentConfigManager,
    AgentSystemConfig,
    AgentMetadataConfig,
    ModelSelectionConfig,
    PromptTemplateConfig,
    ConfigurationError
)
from app.reasoning.context_resolver import ContextResolver
from app.services.conversational_memory import ConversationalMemoryService
from app.services.feedback_analyzer import FeedbackQualityAnalyzer
from app.services.model_adaptation_assessor import ModelAdaptationAssessor
from evals.candidate_evaluator import CandidateEvaluationManager
from evals.evaluator import OfflineBenchmarkEvaluator

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_config_and_memory():
    mgr = AgentConfigManager()
    mgr.reset_to_baseline()
    mem = ConversationalMemoryService()
    mem.clear_all_for_test()
    yield
    mgr.reset_to_baseline()
    mem.clear_all_for_test()

# =============================================================================
# 1. Versioned Agent Configuration System Tests
# =============================================================================

def test_known_good_baseline_config_loading():
    mgr = AgentConfigManager()
    active = mgr.get_active_config()
    assert active.version == "1.0.0"
    assert "HAZARD" in active.routing_policies
    assert "hazard" in active.routing_policies["HAZARD"]
    assert len(active.routing_policies["GENERAL_EXPLANATORY"]) == 0
    assert active.safety_invariants["forbid_fabricated_pfz"] is True

def test_invalid_configuration_rejection():
    mgr = AgentConfigManager()

    # Case A: Negative timeout
    with pytest.raises(Exception):
        ModelSelectionConfig(timeout_seconds=-5.0)

    # Case B: Disabling forbid_fabricated_pfz safety invariant
    with pytest.raises(Exception):
        AgentSystemConfig(
            version="2.0.0-unsafe",
            description="Unsafe config disabling PFZ safety",
            agents=mgr.get_active_config().agents,
            routing_policies=mgr.get_active_config().routing_policies,
            model_selection=ModelSelectionConfig(),
            prompt_templates=mgr.get_active_config().prompt_templates,
            safety_invariants={"forbid_fabricated_pfz": False, "official_warnings_highest_priority": True}
        )

    # Case C: Omitting 'hazard' agent from HAZARD routing policy
    with pytest.raises(Exception):
        bad_routing = dict(mgr.get_active_config().routing_policies)
        bad_routing["HAZARD"] = ["weather", "ocean"]  # Missing hazard agent
        AgentSystemConfig(
            version="2.0.0-unsafe",
            description="Missing hazard agent",
            agents=mgr.get_active_config().agents,
            routing_policies=bad_routing,
            model_selection=ModelSelectionConfig(),
            prompt_templates=mgr.get_active_config().prompt_templates
        )

def test_configuration_activation_and_rollback():
    mgr = AgentConfigManager()
    assert mgr.get_active_version() == "1.0.0"

    # Switch to candidate adaptive version
    mgr.activate_version("1.1.0-adaptive")
    assert mgr.get_active_version() == "1.1.0-adaptive"
    assert "gis" not in mgr.get_active_config().routing_policies["FISHING_LOCATION"]

    # Rollback to baseline
    rolled_back = mgr.rollback()
    assert rolled_back.version == "1.0.0"
    assert mgr.get_active_version() == "1.0.0"

def test_config_export_sanitization():
    mgr = AgentConfigManager()
    exported = mgr.export_config("1.0.0")
    assert exported["version"] == "1.0.0"
    # Ensure no API keys or private variables exist in configuration export
    dump_str = str(exported).lower()
    assert "api_key" not in dump_str
    assert "password" not in dump_str
    assert "bearer" not in dump_str

# =============================================================================
# 2. Controlled Candidate Evaluation Tests
# =============================================================================

@pytest.mark.anyio
async def test_candidate_evaluation_workflow():
    eval_mgr = CandidateEvaluationManager()

    # Evaluate candidate 1.1.0-adaptive
    result = await eval_mgr.evaluate_candidate(
        candidate_version="1.1.0-adaptive",
        defect_description="Redundant GIS calls on general fishing queries",
        hypothesis="Pruning GIS from fishing location reduces agent overhead without compromising safety"
    )

    assert result["candidate_version"] == "1.1.0-adaptive"
    assert result["decision"] == "ACCEPTED"
    assert result["safety_critical_regressions"] == 0
    assert result["zero_fabrication_maintained"] is True
    assert result["official_warning_precedence_maintained"] is True
    assert result["candidate_score_pct"] == 100.0

@pytest.mark.anyio
async def test_candidate_evaluation_rejection_on_regression():
    eval_mgr = CandidateEvaluationManager()
    mgr = AgentConfigManager()

    # Create deliberately degraded candidate that drops the weather agent from marine conditions
    degraded_routing = dict(mgr.get_active_config().routing_policies)
    degraded_routing["MARINE_CONDITION"] = ["ocean"]  # Dropped weather & hazard

    degraded_config = AgentSystemConfig(
        version="1.9.9-degraded",
        description="Degraded candidate for rejection verification",
        agents=mgr.get_active_config().agents,
        routing_policies=degraded_routing,
        model_selection=ModelSelectionConfig(),
        prompt_templates=mgr.get_active_config().prompt_templates
    )
    mgr.register_config(degraded_config, activate=False)

    # Candidate evaluation should fail/reject degraded candidate
    result = await eval_mgr.evaluate_candidate(
        candidate_version="1.9.9-degraded",
        defect_description="Synthetic degradation test",
        hypothesis="Should be rejected"
    )

    # Rejection verification
    assert result["candidate_version"] == "1.9.9-degraded"
    assert result["requires_human_architect_signoff"] is True

# =============================================================================
# 3. Context Resolution & Coastal Port Geocoding Tests
# =============================================================================

def test_coastal_port_geocoding_resolution():
    resolver = ContextResolver()

    # Query mentioning Kochi
    ctx_kochi = resolver.resolve_context("sess_1", "What are the waves off Kochi harbor?", "MARINE_CONDITION", 15.2993, 74.1240)
    assert abs(ctx_kochi["latitude"] - 9.9312) < 0.01
    assert abs(ctx_kochi["longitude"] - 76.2673) < 0.01
    assert ctx_kochi["port_name"] == "Kochi"

    # Query mentioning Veraval
    ctx_veraval = resolver.resolve_context("sess_2", "Check wind speed near Veraval", "MARINE_CONDITION", 15.2993, 74.1240)
    assert abs(ctx_veraval["latitude"] - 20.9000) < 0.01
    assert abs(ctx_veraval["longitude"] - 70.3700) < 0.01

    # Query without explicit port retains passed GPS
    ctx_default = resolver.resolve_context("sess_3", "How are conditions here?", "MARINE_CONDITION", 15.3493, 73.8000)
    assert ctx_default["latitude"] == 15.3493
    assert ctx_default["port_name"] is None

# =============================================================================
# 4. Safe Context & Memory Improvements Tests
# =============================================================================

def test_preference_schema_validation():
    mem = ConversationalMemoryService()

    # Valid preferences update
    valid_up = mem.update_preferences("user_val_01", {
        "preferred_language": "hi",
        "preferred_units": {"wind_speed": "km/h", "wave_height": "m", "temperature": "celsius"}
    })
    assert valid_up["preferred_language"] == "hi"

    # Invalid language code rejected
    with pytest.raises(ValueError) as exc:
        mem.update_preferences("user_val_01", {"preferred_language": "xx-invalid"})
    assert "Unsupported language code" in str(exc.value)

    # Invalid wind_speed unit rejected
    with pytest.raises(ValueError) as exc_unit:
        mem.update_preferences("user_val_01", {"preferred_units": {"wind_speed": "gallons_per_hour"}})
    assert "Invalid wind_speed unit" in str(exc_unit.value)

def test_conversational_freshness_boundary():
    mem = ConversationalMemoryService()
    conv_id = "test_freshness_session"

    # Add a turn 3 hours ago
    old_iso = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=3)).isoformat()
    mem.add_turn(conv_id, "user", "How is the sea in Panaji?", intent="MARINE_CONDITION", latitude=15.4989, longitude=73.8278, location_name="Panaji")
    mem._conversations[conv_id][-1]["timestamp"] = old_iso

    # Follow-up "what about there?" after 3 hours
    res = mem.resolve_followup_reference(conv_id, "what about there?", current_lat=15.2993, current_lon=74.1240)
    assert res["is_followup"] is True
    assert res["is_stale_session"] is True
    assert res["needs_clarification"] is True
    assert "over 2 hours old" in res["clarification_prompt"]

# =============================================================================
# 5. Feedback-Driven Quality Analysis & Model Adaptation Assessment
# =============================================================================

def test_feedback_analyzer_zero_fabrication():
    analyzer = FeedbackQualityAnalyzer()
    # With empty feedback database
    report = analyzer.analyze_feedback_corpus(min_sample_size=5)
    assert report["status"] == "NO_DATA"
    assert report["sample_size"] == 0
    assert "Zero synthetic records fabricated" in report["message"]

def test_feedback_analyzer_with_real_submissions():
    # Submit 5 test feedback records
    client.post("/api/memory/feedback", headers={"X-User-ID": "u1"}, json={"message_id": "m1", "rating": 5, "consent_given": True})
    client.post("/api/memory/feedback", headers={"X-User-ID": "u2"}, json={"message_id": "m2", "rating": 4, "consent_given": True})
    client.post("/api/memory/feedback", headers={"X-User-ID": "u3"}, json={"message_id": "m3", "rating": 1, "feedback_notes": "Wrong language used", "consent_given": True})
    client.post("/api/memory/feedback", headers={"X-User-ID": "u4"}, json={"message_id": "m4", "rating": 2, "feedback_notes": "Hindi translation was inaccurate", "consent_given": True})
    client.post("/api/memory/feedback", headers={"X-User-ID": "u5"}, json={"message_id": "m5", "rating": 5, "consent_given": True})

    analyzer = FeedbackQualityAnalyzer()
    report = analyzer.analyze_feedback_corpus(min_sample_size=5)
    assert report["sample_size"] == 5
    assert report["low_rating_count"] == 2
    # Verify signal categorization
    assert report["problem_category_distribution"]["WRONG_LANGUAGE"] >= 1

def test_model_adaptation_assessment_governance():
    assessor = ModelAdaptationAssessor()
    assessment = assessor.generate_adaptation_assessment()
    assert assessment["model_weight_training_conducted"] is False
    assert assessment["strategies_evaluated"]["supervised_model_finetuning"]["recommendation"] == "NOT_RECOMMENDED_AT_CURRENT_STAGE"
    assert assessment["strategies_evaluated"]["versioned_routing_optimization"]["cost_usd"] == 0.0
