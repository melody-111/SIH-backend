import pytest
import datetime
from fastapi.testclient import TestClient
from app.main import app
from app.services.official_warning_provider import OfficialWarningProvider
from app.services.hazard_assessment_service import HazardAssessmentService
from app.services.alert_rules_engine import AlertRulesEngine
from app.services.notification_service import NotificationDeliveryService
from app.services.offline_cache import OfflineMarineCache
from app.schemas.hazard_alert import (
    OfficialWarningDetail,
    AlertStatus,
    DeliveryStatus,
    AlertSubscriptionRequest
)
from app.core.config import settings

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singletons before each test to ensure test isolation."""
    OfficialWarningProvider().clear()
    NotificationDeliveryService().clear()
    AlertRulesEngine().clear()
    OfflineMarineCache().clear()
    OfflineMarineCache().set_force_offline(False)
    # Re-init default test bulletins
    OfficialWarningProvider()._init_default_bulletins()
    yield
    OfficialWarningProvider().clear()
    NotificationDeliveryService().clear()
    AlertRulesEngine().clear()
    OfflineMarineCache().clear()

# -------------------------------------------------------------
# 1. Coordinate Validation Tests
# -------------------------------------------------------------
def test_valid_coordinates():
    response = client.get("/api/hazard/assessment", params={"latitude": 15.3493, "longitude": 73.8000})
    assert response.status_code == 200
    data = response.json()
    assert data["latitude"] == 15.3493
    assert data["longitude"] == 73.8000
    assert "overall_risk_level" in data
    assert isinstance(data["is_safe_to_sail"], bool)

def test_invalid_latitude():
    response = client.get("/api/hazard/assessment", params={"latitude": 95.0, "longitude": 73.8000})
    assert response.status_code == 400
    assert "Invalid latitude" in response.json()["detail"]

def test_invalid_longitude():
    response = client.get("/api/hazard/assessment", params={"latitude": 15.3493, "longitude": -195.0})
    assert response.status_code == 400
    assert "Invalid longitude" in response.json()["detail"]

# -------------------------------------------------------------
# 2. Warning Geography & Time Window Matching
# -------------------------------------------------------------
def test_warning_geography_matching():
    # Warning exists for Goa (lat: 15.3493, lon: 73.8, radius: 120km)
    goa_res = client.get("/api/hazard/warnings", params={"latitude": 15.3493, "longitude": 73.8000})
    assert goa_res.status_code == 200
    goa_warnings = goa_res.json()
    assert len(goa_warnings) >= 1
    assert any("Goa" in w["bulletin_text"] or "Goa" in str(w["affected_regions"]) for w in goa_warnings)

    # Visakhapatnam is ~1,000 km away on East coast - should not match Goa warning
    vizag_res = client.get("/api/hazard/warnings", params={"latitude": 17.6868, "longitude": 83.2185})
    assert vizag_res.status_code == 200
    vizag_warnings = vizag_res.json()
    assert len(vizag_warnings) == 0

def test_warning_time_window_and_expired():
    provider = OfficialWarningProvider()
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Register an expired warning
    expired_bulletin = OfficialWarningDetail(
        warning_id="TEST-EXP-01",
        source="IMD",
        warning_type="SQUALL_ALERT",
        source_severity="YELLOW WATCH",
        normalized_severity="ADVISORY",
        bulletin_text="Old squall bulletin that has already passed.",
        issue_timestamp=(now - datetime.timedelta(days=2)).isoformat(),
        valid_from=(now - datetime.timedelta(days=2)).isoformat(),
        valid_until=(now - datetime.timedelta(hours=1)).isoformat(),  # 1 hour ago
        affected_coordinates={"latitude": 15.3493, "longitude": 73.8000},
        coverage_radius_km=100.0,
        status=AlertStatus.ACTIVE
    )
    provider.register_bulletin(expired_bulletin)

    # Default query excludes expired
    active_warnings = provider.get_warnings_for_location(15.3493, 73.8000, include_expired=False)
    assert not any(w.warning_id == "TEST-EXP-01" for w in active_warnings)

    # Query with include_expired=True returns it tagged as EXPIRED
    all_warnings = provider.get_warnings_for_location(15.3493, 73.8000, include_expired=True)
    exp = next((w for w in all_warnings if w.warning_id == "TEST-EXP-01"), None)
    assert exp is not None
    assert exp.status == AlertStatus.EXPIRED

# -------------------------------------------------------------
# 3. Escalated, Updated & Cancelled Warnings
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_escalated_and_cancelled_warnings():
    provider = OfficialWarningProvider()
    rules = AlertRulesEngine()
    now = datetime.datetime.now(datetime.timezone.utc)

    # Subscribe user
    sub = rules.add_subscription(
        AlertSubscriptionRequest(
            user_id="user_sailor_01",
            device_token="device_token_abc",
            latitude=15.3493,
            longitude=73.8000,
            radius_km=50.0,
            min_severity="ADVISORY",
            enable_push=True,
            enable_in_app=True
        )
    )

    # Initial warning
    warning = OfficialWarningDetail(
        warning_id="TEST-CYC-01",
        source="IMD Cyclone Warning Division",
        warning_type="CYCLONIC_STORM",
        source_severity="ORANGE WARNING",
        normalized_severity="WARNING",
        bulletin_text="Cyclonic storm approaching coast.",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        valid_until=(now + datetime.timedelta(hours=24)).isoformat(),
        affected_coordinates={"latitude": 15.3493, "longitude": 73.8000},
        coverage_radius_km=100.0,
        status=AlertStatus.ACTIVE
    )
    provider.register_bulletin(warning)

    # First dispatch
    records_1 = await rules.process_warning_dispatch(warning, target_user_id="user_sailor_01")
    assert any(r.delivery_status == DeliveryStatus.DELIVERED for r in records_1)

    # Escalate warning to RED DANGER
    escalated = provider.update_bulletin_status(
        warning_id="TEST-CYC-01",
        new_status=AlertStatus.ESCALATED,
        new_source_severity="RED DANGER",
        updated_bulletin_text="ESCALATION: Cyclone upgraded to Extremely Severe. Imminent landfall."
    )
    assert escalated is not None
    assert escalated.status == AlertStatus.ESCALATED
    assert escalated.normalized_severity == "DANGER"

    # Escalated warning must be dispatched immediately despite deduplication window
    records_2 = await rules.process_warning_dispatch(escalated, target_user_id="user_sailor_01")
    escalated_deliveries = [r for r in records_2 if r.delivery_status == DeliveryStatus.DELIVERED]
    assert len(escalated_deliveries) > 0
    assert any("ESCALATED" in r.message for r in escalated_deliveries)

    # Cancel warning
    cancelled = provider.update_bulletin_status(
        warning_id="TEST-CYC-01",
        new_status=AlertStatus.CANCELLED,
        updated_bulletin_text="Cyclone moved inland and weakened. All coastal warnings cancelled."
    )
    assert cancelled is not None
    records_3 = await rules.process_warning_dispatch(cancelled, target_user_id="user_sailor_01")
    cancelled_deliveries = [r for r in records_3 if r.delivery_status == DeliveryStatus.DELIVERED]
    assert len(cancelled_deliveries) > 0
    assert any("CANCELLED" in r.message for r in cancelled_deliveries)

# -------------------------------------------------------------
# 4. Duplicate-Alert Suppression and Idempotency
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_duplicate_suppression_and_idempotency():
    rules = AlertRulesEngine()
    now = datetime.datetime.now(datetime.timezone.utc)

    rules.add_subscription(
        AlertSubscriptionRequest(
            user_id="user_test_dedup",
            device_token="device_token_xyz",
            latitude=15.3493,
            longitude=73.8000,
            radius_km=50.0,
            min_severity="ADVISORY"
        )
    )

    warning = OfficialWarningDetail(
        warning_id="TEST-WARN-DEDUP",
        source="INCOIS Ocean State",
        warning_type="HIGH_WAVE",
        source_severity="ORANGE WARNING",
        normalized_severity="WARNING",
        bulletin_text="High swell waves of 3.2 meters expected.",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        valid_until=(now + datetime.timedelta(hours=12)).isoformat(),
        affected_coordinates={"latitude": 15.3493, "longitude": 73.8000},
        coverage_radius_km=80.0,
        status=AlertStatus.ACTIVE
    )
    OfficialWarningProvider().register_bulletin(warning)

    # First attempt: Delivered
    res1 = await rules.process_warning_dispatch(warning, target_user_id="user_test_dedup")
    delivered = [r for r in res1 if r.delivery_status == DeliveryStatus.DELIVERED]
    assert len(delivered) > 0
    key1 = delivered[0].idempotency_key
    assert len(key1) > 0

    # Second immediate attempt: Suppressed Duplicate
    res2 = await rules.process_warning_dispatch(warning, target_user_id="user_test_dedup")
    suppressed = [r for r in res2 if r.delivery_status == DeliveryStatus.SUPPRESSED_DUPLICATE]
    assert len(suppressed) > 0
    assert suppressed[0].idempotency_key == key1  # Deterministic idempotency key

# -------------------------------------------------------------
# 5. Permission Denial & Failed Deliveries
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_notification_permission_denial():
    notif_service = NotificationDeliveryService()
    now = datetime.datetime.now(datetime.timezone.utc)
    warning = OfficialWarningDetail(
        warning_id="TEST-W1",
        source="IMD",
        warning_type="ROUGH_SEA",
        source_severity="WARNING",
        normalized_severity="WARNING",
        bulletin_text="Rough sea test",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        affected_coordinates={"latitude": 15.0, "longitude": 73.0}
    )

    # Set user permission denied
    notif_service.set_user_permission_denied("user_denied", denied=True)
    rec = await notif_service.deliver_push(
        user_id="user_denied",
        device_token="some_token",
        warning=warning,
        idempotency_key="idemp_123",
        message="Test alert"
    )
    assert rec.delivery_status == DeliveryStatus.PERMISSION_DENIED

@pytest.mark.anyio
async def test_failed_push_missing_token_and_unconfigured_sms():
    notif_service = NotificationDeliveryService()
    now = datetime.datetime.now(datetime.timezone.utc)
    warning = OfficialWarningDetail(
        warning_id="TEST-W2",
        source="IMD",
        warning_type="ROUGH_SEA",
        source_severity="WARNING",
        normalized_severity="WARNING",
        bulletin_text="Rough sea test",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        affected_coordinates={"latitude": 15.0, "longitude": 73.0}
    )

    # Missing push token
    push_rec = await notif_service.deliver_push(
        user_id="user_no_token",
        device_token=None,
        warning=warning,
        idempotency_key="idemp_456",
        message="Test alert"
    )
    assert push_rec.delivery_status == DeliveryStatus.FAILED
    assert "Missing user device registration token" in push_rec.provider_response

    # Unconfigured SMS gateway
    sms_rec = await notif_service.deliver_sms(
        user_id="user_sms",
        warning=warning,
        idempotency_key="idemp_789",
        message="Test alert"
    )
    assert sms_rec.delivery_status == DeliveryStatus.FAILED
    assert "SMS_API_KEY" in sms_rec.provider_response

# -------------------------------------------------------------
# 6. User Subscription Isolation & Auth
# -------------------------------------------------------------
def test_subscription_isolation_and_ownership():
    # User A creates a subscription
    sub_a_res = client.post(
        "/api/alerts/subscribe",
        headers={"X-User-ID": "user_alice"},
        json={
            "user_id": "user_alice",
            "device_token": "token_alice",
            "latitude": 15.3493,
            "longitude": 73.8000,
            "radius_km": 60.0,
            "min_severity": "WARNING"
        }
    )
    assert sub_a_res.status_code == 200
    sub_a = sub_a_res.json()
    sub_a_id = sub_a["subscription_id"]

    # User B queries subscriptions -> must be empty
    sub_b_res = client.get("/api/alerts/subscriptions", headers={"X-User-ID": "user_bob"})
    assert sub_b_res.status_code == 200
    assert len(sub_b_res.json()) == 0

    # User B tries to delete User A's subscription -> 404 / unauthorized
    del_res = client.delete(f"/api/alerts/subscriptions/{sub_a_id}", headers={"X-User-ID": "user_bob"})
    assert del_res.status_code == 404

    # User A deletes their own subscription -> success
    del_a_res = client.delete(f"/api/alerts/subscriptions/{sub_a_id}", headers={"X-User-ID": "user_alice"})
    assert del_a_res.status_code == 200

# -------------------------------------------------------------
# 7. Offline Cache: 2-Day Past + 2-Day Forecast Window & Stale Behavior
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_offline_cache_sync_and_incremental():
    cache = OfflineMarineCache()
    sync_res = await cache.sync_cache_window(
        lat=15.3493,
        lon=73.8000,
        location_name="Goa Coast",
        past_days=2,
        forecast_days=2
    )
    assert sync_res.status_code if hasattr(sync_res, "status_code") else True
    assert sync_res.cached_window_past_days == 2
    assert sync_res.cached_window_forecast_days == 2
    assert sync_res.records_cached_count > 0
    assert sync_res.status in ["FRESH", "PARTIAL"]

    # Incremental sync with same timestamp returns UP_TO_DATE
    inc_res = await cache.sync_cache_window(
        lat=15.3493,
        lon=73.8000,
        past_days=2,
        forecast_days=2,
        last_sync_timestamp=sync_res.sync_timestamp
    )
    assert inc_res.status == "UP_TO_DATE"

def test_stale_cache_and_offline_behavior():
    cache = OfflineMarineCache()
    # Query location with no cached data
    empty_res = cache.get_offline_data(lat=22.0, lon=89.0)
    assert empty_res["status"] == "UNAVAILABLE_OFFLINE"
    assert empty_res["is_cached"] is False

    # Store cache entry
    cache.store_observation(
        lat=15.3493,
        lon=73.8000,
        data={"wave_height": 1.2, "wind_speed": 18.0},
        location_name="Goa Port"
    )

    # Query online / offline
    cached_data = cache.get_offline_data(lat=15.35, lon=73.81)
    assert cached_data["is_cached"] is True
    assert cached_data["distance_km"] < 5.0
    assert cached_data["status"] in ["CACHED_FRESH", "CACHED_STALE"]

# -------------------------------------------------------------
# 8. Source Attribution, Units, and No Fabricated Data
# -------------------------------------------------------------
@pytest.mark.anyio
async def test_source_attribution_and_no_fabricated_alerts():
    hazard_svc = HazardAssessmentService()
    assessment = await hazard_svc.assess_hazard(lat=15.3493, lon=73.8000, location_name="Goa Coast")
    
    # Official warnings preserve original severity and source
    for w in assessment.official_warnings:
        assert w.is_official is True
        assert len(w.source_severity) > 0
        assert len(w.source) > 0
        assert len(w.bulletin_text) > 0

    # Model-derived indicators have documented units and thresholds
    for ind in assessment.derived_indicators:
        assert ind.unit in ["km/h", "m", "hPa"]
        assert ind.threshold > 0
        assert len(ind.source_model) > 0

    # Uncertainty notes explicitly present when government API key is unconfigured
    if not settings.IMD_API_KEY:
        assert any("IMD" in note or "key" in note for note in assessment.uncertainty_notes)
