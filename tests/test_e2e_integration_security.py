import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.hazard_alert import AlertSubscriptionRequest
from app.services.offline_cache import OfflineMarineCache
from app.services.conversational_memory import ConversationalMemoryService
from app.services.language_service import LanguageService
from app.services.voice_service import VoiceService

client = TestClient(app)

# =============================================================================
# 1. SECURITY & TRACING: CORRELATION ID & REQUEST HEADERS
# =============================================================================

def test_correlation_id_auto_generation():
    """Verify that requests without correlation ID header get an auto-generated X-Correlation-ID."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "x-correlation-id" in response.headers
    assert len(response.headers["x-correlation-id"]) > 0

def test_correlation_id_propagation():
    """Verify that a client-provided X-Correlation-ID is preserved and returned."""
    custom_cid = "test-client-trace-12345"
    response = client.get("/api/health", headers={"X-Correlation-ID": custom_cid})
    assert response.status_code == 200
    assert response.headers.get("x-correlation-id") == custom_cid

def test_cors_headers_and_preflight():
    """Verify CORS origin handling and exposed tracing headers."""
    response = client.get(
        "/api/health",
        headers={
            "Origin": "http://localhost:19006"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
    assert "x-correlation-id" in response.headers.get("access-control-expose-headers", "").lower()

# =============================================================================
# 2. SECURITY: SAFE ERROR HANDLING & NO STACK TRACE LEAKAGE
# =============================================================================

def test_unhandled_error_sanitization_no_stack_traces():
    """
    Ensure 500/422 errors are cleanly formatted and NEVER leak internal Python
    stack traces, file paths, or credentials to the client.
    """
    # Trigger bad latitude query validation error (422) vs bad request
    response = client.get("/api/marine?latitude=999.0&longitude=74.0")
    assert response.status_code == 422
    assert "Traceback" not in response.text

    # Route with invalid coordinate bounds
    route_resp = client.post("/api/map/route", json={
        "origin_latitude": 999.0,
        "origin_longitude": 74.0,
        "destination_latitude": 15.0,
        "destination_longitude": 74.0
    })
    assert route_resp.status_code == 422
    assert "Traceback" not in route_resp.text

# =============================================================================
# 3. SECURITY: TENANT DATA ISOLATION & USER AUTHORIZATION
# =============================================================================

def test_memory_tenant_isolation_and_authorization():
    """
    Verify strict user-level data isolation:
    User A cannot view, modify, or delete User B's memories.
    """
    user_a = "user_alpha_1"
    user_b = "user_beta_2"

    # 1. Unauthenticated request rejected
    unauth_resp = client.get("/api/memory/memories")
    assert unauth_resp.status_code == 401

    # 2. User A creates memory
    create_resp = client.post(
        "/api/memory/memories",
        headers={"X-User-ID": user_a},
        json={
            "title": "Preferred Dock",
            "content": "Mormugao harbor berth 4",
            "category": "PREFERENCE"
        }
    )
    assert create_resp.status_code == 200
    mem_id = create_resp.json()["memory_id"]

    # 3. User B cannot see User A's memory
    user_b_list = client.get("/api/memory/memories", headers={"X-User-ID": user_b})
    assert user_b_list.status_code == 200
    assert not any(m["memory_id"] == mem_id for m in user_b_list.json())

    # 4. User B cannot delete User A's memory
    del_resp = client.delete(f"/api/memory/memories/{mem_id}", headers={"X-User-ID": user_b})
    assert del_resp.status_code == 404

    # 5. User A can delete their own memory
    del_a_resp = client.delete(f"/api/memory/memories/{mem_id}", headers={"X-User-ID": user_a})
    assert del_a_resp.status_code == 200

def test_alert_subscription_tenant_isolation():
    """
    Verify that alert subscriptions cannot be subscribed or deleted across tenants.
    """
    user_a = "fisherman_10"
    user_b = "fisherman_20"

    # 1. Cannot subscribe on behalf of another user
    mismatch_resp = client.post(
        "/api/alerts/subscribe",
        headers={"X-User-ID": user_b},
        json={
            "user_id": user_a,
            "region_name": "Goa Coast",
            "latitude": 15.29,
            "longitude": 74.12
        }
    )
    assert mismatch_resp.status_code == 403

    # 2. Valid subscription for user A
    sub_resp = client.post(
        "/api/alerts/subscribe",
        headers={"X-User-ID": user_a},
        json={
            "user_id": user_a,
            "region_name": "Goa Coast",
            "latitude": 15.29,
            "longitude": 74.12
        }
    )
    assert sub_resp.status_code == 200
    sub_id = sub_resp.json()["subscription_id"]

    # 3. User B cannot delete User A's subscription
    del_b = client.delete(f"/api/alerts/subscriptions/{sub_id}", headers={"X-User-ID": user_b})
    assert del_b.status_code == 404

    # 4. User A deletes their subscription
    del_a = client.delete(f"/api/alerts/subscriptions/{sub_id}", headers={"X-User-ID": user_a})
    assert del_a.status_code == 200

def test_privacy_guard_blocks_credentials_in_memory():
    """Verify that memories containing secret keys or passwords are strictly rejected."""
    user = "fisherman_privacy_check"
    resp = client.post(
        "/api/memory/memories",
        headers={"X-User-ID": user},
        json={
            "title": "Private Note",
            "content": "My secret password is my_secret_token_12345",
            "category": "NOTE"
        }
    )
    assert resp.status_code == 400
    assert "privacy policy violation" in resp.json()["detail"].lower()

# =============================================================================
# 4. WORKFLOW A: MARINE INFORMATION QUERY END-TO-END
# =============================================================================

@pytest.mark.anyio
async def test_workflow_a_marine_query_grounding():
    """
    Workflow A: Mobile query -> orchestrator -> marine data -> validation -> grounded response.
    Verify source attribution, timestamps, and absence of fabricated values.
    """
    response = client.post("/api/chat", json={
        "message": "What is the wave height and sea conditions near Goa?",
        "latitude": 15.2993,
        "longitude": 74.1240,
        "language": "en"
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert data["risk"] in ["LOW", "MODERATE", "HIGH", "DANGER", "UNKNOWN"]
    assert "timestamp" in data
    assert len(data.get("progress_steps", [])) > 0
    # Verify execution plan contains planned specialists
    assert len(data.get("execution_plan", [])) > 0

# =============================================================================
# 5. WORKFLOW B: FISHING / PFZ QUERY WITH UNCONFIGURED SAFETY
# =============================================================================

@pytest.mark.anyio
async def test_workflow_b_fishing_pfz_unauthenticated_safety():
    """
    Workflow B: Fishing query handled safely when INCOIS key is not configured.
    Must never fabricate fish abundance or bogus GPS coordinates.
    """
    response = client.post("/api/chat", json={
        "message": "Where are the potential fishing zones in the Arabian Sea today?",
        "latitude": 15.2993,
        "longitude": 74.1240,
        "language": "en"
    })
    assert response.status_code == 200
    data = response.json()
    # Answer should be transparent about data status
    assert len(data["answer"]) > 0
    assert "timestamp" in data

    # Check the GIS PFZ map layer endpoint directly
    pfz_layer = client.get("/api/map/layer-data?layer_id=pfz&latitude=15.2993&longitude=74.1240")
    assert pfz_layer.status_code == 200
    pfz_json = pfz_layer.json()
    assert pfz_json["type"] == "FeatureCollection"
    assert pfz_json["metadata"]["status"] == "REQUIRES_AUTH"
    assert len(pfz_json["features"]) == 0

# =============================================================================
# 6. WORKFLOW C: HAZARD WARNING DEDUPLICATION & ALERT DELIVERY
# =============================================================================

@pytest.mark.anyio
async def test_workflow_c_hazard_warning_and_deduplication():
    """
    Workflow C: Ingest official warnings, match geographic area,
    apply deduplication, and simulate delivery record in sandbox mode.
    """
    # 1. Fetch active warnings for Arabian Sea
    resp = client.get("/api/hazard/warnings?latitude=15.0&longitude=72.0&region_name=Arabian Sea")
    assert resp.status_code == 200
    warnings = resp.json()
    assert isinstance(warnings, list)

    # 2. Sandbox delivery simulation for user
    test_user = "pilot_fisherman_goa"
    sub_resp = client.post(
        "/api/alerts/subscribe",
        headers={"X-User-ID": test_user},
        json={
            "user_id": test_user,
            "region_name": "Arabian Sea",
            "latitude": 15.0,
            "longitude": 72.0,
            "minimum_severity": "MODERATE"
        }
    )
    assert sub_resp.status_code == 200

    # Test delivery
    delivery_resp = client.post(
        "/api/alerts/test-deliver",
        headers={"X-User-ID": test_user},
        json={"user_id": test_user}
    )
    assert delivery_resp.status_code == 200
    first_records = delivery_resp.json()

    # Second delivery within window should be suppressed by deduplication
    dedup_resp = client.post(
        "/api/alerts/test-deliver",
        headers={"X-User-ID": test_user},
        json={"user_id": test_user}
    )
    assert dedup_resp.status_code == 200
    second_records = dedup_resp.json()
    assert len(second_records) == 0  # Deduplicated

# =============================================================================
# 7. WORKFLOW D: OFFLINE CACHED QUERY & STALENESS
# =============================================================================

@pytest.mark.anyio
async def test_workflow_d_offline_cached_query():
    """
    Workflow D: Cache sync -> toggle offline -> query returns labeled cached response.
    """
    cache = OfflineMarineCache()
    # 1. Sync cache window
    sync_res = client.post("/api/cache/sync", json={
        "latitude": 15.2993,
        "longitude": 74.1240,
        "location_name": "Goa Port",
        "past_days": 2,
        "forecast_days": 2
    })
    assert sync_res.status_code == 200

    # 2. Force offline mode
    client.post("/api/cache/toggle-offline", json={"enabled": True})

    try:
        # 3. Query orchestrator in offline state
        offline_query = client.post("/api/chat", json={
            "message": "What are the sea conditions?",
            "latitude": 15.2993,
            "longitude": 74.1240,
            "language": "en"
        })
        assert offline_query.status_code == 200
        result = offline_query.json()
        assert result.get("is_cached") is True
        assert "answer" in result
    finally:
        # Restore online mode
        client.post("/api/cache/toggle-offline", json={"enabled": False})

# =============================================================================
# 8. WORKFLOW E: MULTILINGUAL VOICE & SPEECH SANITIZATION
# =============================================================================

def test_workflow_e_multilingual_voice_and_sanitization():
    """
    Workflow E: Language detection, unconfigured STT reporting, and TTS sanitization.
    """
    # 1. Hindi language detection
    lang_resp = client.post("/api/voice/detect-language", json={"text": "कल समुद्र में लहरें कैसी रहेंगी?"})
    assert lang_resp.status_code == 200
    assert lang_resp.json()["language"] == "hi"

    # 2. English language detection
    lang_en = client.post("/api/voice/detect-language", json={"text": "What is the wave height tomorrow?"})
    assert lang_en.status_code == 200
    assert lang_en.json()["language"] == "en"

    # 3. STT reports unconfigured provider cleanly without fake transcription
    stt_resp = client.post("/api/voice/stt", json={"audio_base64": "dummy_audio_bytes", "language": "hi"})
    assert stt_resp.status_code == 200
    assert stt_resp.json()["status"] == "UNCONFIGURED"

    # 4. TTS sanitization strips code blocks, JSON dicts, and internal diagnostic tokens
    tts_resp = client.post("/api/voice/tts", json={
        "text": "```json {'secret': 'key'} ``` Ocean conditions are safe. [AGENT:hazard]",
        "language": "en"
    })
    assert tts_resp.status_code == 200
    sanitized = tts_resp.json()["spoken_text"]
    assert "secret" not in sanitized
    assert "[AGENT:hazard]" not in sanitized
    assert "Ocean conditions are safe" in sanitized

# =============================================================================
# 9. WORKFLOW F: GIS MARINE MAP & GEODESIC ROUTE SAFETY
# =============================================================================

@pytest.mark.anyio
async def test_workflow_f_gis_map_and_route_intelligence():
    """
    Workflow F: GeoJSON coordinate order [lon, lat], route distance,
    warning proximity, and mandatory nautical navigation disclaimer.
    """
    # 1. GeoJSON layers catalog
    layers_resp = client.get("/api/map/layers")
    assert layers_resp.status_code == 200
    layers = layers_resp.json()
    assert len(layers) >= 5

    # 2. Live SST GeoJSON layer preserves [longitude, latitude]
    sst_resp = client.get("/api/map/layer-data?layer_id=sst&latitude=15.2993&longitude=74.1240")
    assert sst_resp.status_code == 200
    sst_geojson = sst_resp.json()
    assert sst_geojson["type"] == "FeatureCollection"
    for feature in sst_geojson["features"]:
        coords = feature["geometry"]["coordinates"]
        # In WGS84 GeoJSON, coords[0] is longitude (approx 74.0), coords[1] is latitude (approx 15.3)
        assert 70.0 <= coords[0] <= 78.0
        assert 12.0 <= coords[1] <= 18.0

    # 3. Geodesic Maritime Route calculation
    route_resp = client.post("/api/map/route", json={
        "origin_latitude": 15.2993,
        "origin_longitude": 74.1240,
        "destination_latitude": 18.9220,
        "destination_longitude": 72.8347,
        "vessel_speed_knots": 10.0
    })
    assert route_resp.status_code == 200
    route_data = route_resp.json()
    assert route_data["total_distance_km"] > 400.0
    assert route_data["total_distance_nautical_miles"] > 200.0
    assert route_data["estimated_transit_minutes"] > 0
    assert len(route_data["waypoints"]) >= 5
    # Mandatory nautical safety limitation
    assert "nautical" in route_data["navigational_limitation"].lower()
    assert "bathymetric" in route_data["navigational_limitation"].lower()
