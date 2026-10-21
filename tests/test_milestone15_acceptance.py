import pytest
import datetime
import uuid
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.official_warning_provider import OfficialWarningProvider
from app.services.hazard_assessment_service import HazardAssessmentService
from app.services.alert_rules_engine import AlertRulesEngine
from app.services.notification_service import NotificationDeliveryService
from app.services.offline_cache import OfflineMarineCache
from app.connectors.open_meteo_weather import OpenMeteoWeatherConnector
from app.connectors.open_meteo_marine import OpenMeteoMarineConnector
from app.connectors.nasa_earthdata import NASAEarthdataConnector
from app.connectors.imd_connector import IMDWarningsConnector
from app.connectors.incois_connector import INCOISConnector
from app.schemas.hazard_alert import (
    OfficialWarningDetail,
    AlertStatus,
    DeliveryStatus,
    AlertSubscriptionRequest
)
from app.schemas.marine_normalized import ConnectorStatus
from app.core.logging import SensitiveDataRedactor
from app.core.config import settings

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_singletons_and_cache():
    OfficialWarningProvider().clear()
    NotificationDeliveryService().clear()
    AlertRulesEngine().clear()
    OfflineMarineCache().clear()
    OfflineMarineCache().set_force_offline(False)
    # Re-initialize baseline bulletin
    OfficialWarningProvider()._init_default_bulletins()
    yield
    OfficialWarningProvider().clear()
    NotificationDeliveryService().clear()
    AlertRulesEngine().clear()
    OfflineMarineCache().clear()

# =============================================================================
# 1. LIVE PROVIDER VERIFICATION CONTRACTS
# =============================================================================

@pytest.mark.anyio
async def test_m15_live_provider_connector_contracts():
    """
    Verifies that all 5 connectors declare explicit status,
    provide honest attribution, and never hallucinate fake credentials.
    """
    lat, lon = 15.2993, 73.8000

    # 1. Open-Meteo Weather
    w_conn = OpenMeteoWeatherConnector()
    assert w_conn.base_url == "https://api.open-meteo.com/v1/forecast"

    # 2. Open-Meteo Marine
    m_conn = OpenMeteoMarineConnector()
    assert m_conn.base_url == "https://marine-api.open-meteo.com/v1/marine"

    # 3. NASA Earthdata CMR
    nasa_conn = NASAEarthdataConnector()
    assert nasa_conn.status == ConnectorStatus.METADATA_ONLY
    assert nasa_conn.base_url == "https://cmr.earthdata.nasa.gov/search/collections.json"

    # 4. IMD Connector unconfigured gate
    with patch("app.core.config.settings.IMD_API_KEY", ""), \
         patch("app.core.config.settings.ENABLE_SIMULATION_FALLBACK", False):
        imd_conn = IMDWarningsConnector()
        res = await imd_conn.fetch_warnings(lat, lon)
        assert res.connector_status == ConnectorStatus.REQUIRES_ACCESS.value
        assert res.success is False
        norm = imd_conn.normalize(res, lat, lon)
        assert len(norm) == 1
        assert norm[0].warning_type == "OFFICIAL_FEED_UNAVAILABLE"
        assert norm[0].is_official is False

    # 5. INCOIS Connector unconfigured gate
    with patch("app.core.config.settings.INCOIS_API_KEY", ""), \
         patch("app.core.config.settings.ENABLE_SIMULATION_FALLBACK", False):
        incois_conn = INCOISConnector()
        res = await incois_conn.fetch_advisory(lat, lon)
        assert res.connector_status == ConnectorStatus.REQUIRES_ACCESS.value
        assert res.success is False
        norm = incois_conn.normalize(res, lat, lon)
        assert len(norm) == 1
        assert norm[0].value is None
        assert norm[0].quality_status.value == "OFFLINE"

# =============================================================================
# 2. CYCLONE & MARINE ALERT WORKFLOW (GEOGRAPHY, VALIDITY, ESCALATION, DEDUP)
# =============================================================================

def test_m15_cyclone_alert_ingestion_and_geo_matching():
    """
    Verifies that an official cyclone warning can be ingested, validated,
    matched for coastal zones within its radius, and excluded for distant ports.
    """
    provider = OfficialWarningProvider()
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Official IMD Cyclone Alert fixture
    cyclone = OfficialWarningDetail(
        warning_id="IMD-CYC-2026-GUJ-01",
        source="India Meteorological Department (IMD)",
        warning_type="CYCLONIC_STORM_ALERT",
        source_severity="RED WARNING",
        normalized_severity="DANGER",
        bulletin_text="Severe Cyclonic Storm over Northeast Arabian Sea approaching Saurashtra coast. Total suspension of fishing operations advised.",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        valid_until=(now + datetime.timedelta(hours=48)).isoformat(),
        affected_regions=["Porbandar", "Dwarka", "Okha", "Saurashtra Coast"],
        affected_coordinates={"latitude": 21.6417, "longitude": 69.6083},
        coverage_radius_km=150.0,
        status=AlertStatus.ACTIVE,
        is_official=True
    )
    provider.register_bulletin(cyclone)

    # 1. Porbandar Harbor (~10 km away) -> Matches
    porbandar_warnings = provider.get_warnings_for_location(21.64, 69.61)
    assert any(w.warning_id == "IMD-CYC-2026-GUJ-01" for w in porbandar_warnings)
    match = next(w for w in porbandar_warnings if w.warning_id == "IMD-CYC-2026-GUJ-01")
    assert match.is_official is True
    assert match.source_severity == "RED WARNING"
    assert match.normalized_severity == "DANGER"

    # 2. Kochi Harbor (~1,500 km away) -> Excluded
    kochi_warnings = provider.get_warnings_for_location(9.9312, 76.2673)
    assert not any(w.warning_id == "IMD-CYC-2026-GUJ-01" for w in kochi_warnings)

def test_m15_cyclone_alert_validity_and_expiry():
    """
    Verifies that expired warnings are identified as expired and suppressed
    from default active queries.
    """
    provider = OfficialWarningProvider()
    now = datetime.datetime.now(datetime.timezone.utc)

    # Expired warning (valid_until was 2 hours ago)
    past_warning = OfficialWarningDetail(
        warning_id="IMD-EXP-2026-GOA-99",
        source="IMD",
        warning_type="ROUGH_SEA_ALERT",
        source_severity="YELLOW WATCH",
        normalized_severity="ADVISORY",
        bulletin_text="Past sea alert has concluded.",
        issue_timestamp=(now - datetime.timedelta(hours=26)).isoformat(),
        valid_from=(now - datetime.timedelta(hours=24)).isoformat(),
        valid_until=(now - datetime.timedelta(hours=2)).isoformat(),
        affected_coordinates={"latitude": 15.3493, "longitude": 73.8000},
        coverage_radius_km=100.0,
        status=AlertStatus.ACTIVE,
        is_official=True
    )
    provider.register_bulletin(past_warning)

    # Active query excludes expired
    active_list = provider.get_warnings_for_location(15.3493, 73.8000, include_expired=False)
    assert not any(w.warning_id == "IMD-EXP-2026-GOA-99" for w in active_list)

    # Query with include_expired=True returns it tagged EXPIRED
    all_list = provider.get_warnings_for_location(15.3493, 73.8000, include_expired=True)
    exp = next((w for w in all_list if w.warning_id == "IMD-EXP-2026-GOA-99"), None)
    assert exp is not None
    assert exp.status == AlertStatus.EXPIRED

@pytest.mark.anyio
async def test_m15_cyclone_alert_deduplication_and_escalation():
    """
    Verifies:
    1. Identical warning dispatched twice within window is deduplicated and suppressed.
    2. Escalated warning (version increment) bypasses deduplication and delivers immediately.
    """
    provider = OfficialWarningProvider()
    rules = AlertRulesEngine()
    notif_svc = NotificationDeliveryService()
    now = datetime.datetime.now(datetime.timezone.utc)

    sub = rules.add_subscription(
        AlertSubscriptionRequest(
            user_id="capt_fernandes_01",
            device_token="fcm_token_mormugao_77",
            latitude=15.3493,
            longitude=73.8000,
            radius_km=60.0,
            min_severity="ADVISORY",
            enable_push=True,
            enable_in_app=True
        )
    )

    bulletin = OfficialWarningDetail(
        warning_id="IMD-GOA-CYCLONE-44",
        source="IMD Cyclone Warning Division",
        warning_type="DEEP_DEPRESSION",
        source_severity="ORANGE WARNING",
        normalized_severity="WARNING",
        bulletin_text="Deep depression intensifies over East Central Arabian Sea.",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        valid_until=(now + datetime.timedelta(hours=24)).isoformat(),
        affected_coordinates={"latitude": 15.3493, "longitude": 73.8000},
        coverage_radius_km=100.0,
        status=AlertStatus.ACTIVE,
        is_official=True
    )
    provider.register_bulletin(bulletin, version=1)

    # First dispatch -> Successfully delivered
    records_1 = await rules.process_warning_dispatch(bulletin, target_user_id="capt_fernandes_01")
    delivered_1 = [r for r in records_1 if r.delivery_status == DeliveryStatus.DELIVERED]
    assert len(delivered_1) >= 1

    # Second immediate dispatch of same bulletin -> SUPPRESSED_DUPLICATE
    records_2 = await rules.process_warning_dispatch(bulletin, target_user_id="capt_fernandes_01")
    suppressed = [r for r in records_2 if r.delivery_status == DeliveryStatus.SUPPRESSED_DUPLICATE]
    assert len(suppressed) >= 1
    assert "Duplicate alert suppressed" in suppressed[0].message

    # Escalation to RED WARNING -> Bypasses deduplication and dispatches
    escalated = provider.update_bulletin_status(
        warning_id="IMD-GOA-CYCLONE-44",
        new_status=AlertStatus.ESCALATED,
        new_source_severity="RED WARNING",
        updated_bulletin_text="ESCALATED: Severe Cyclonic Storm approaching Mormugao. All crafts return to port."
    )
    assert escalated is not None
    assert provider.get_bulletin_version("IMD-GOA-CYCLONE-44") == 2

    records_3 = await rules.process_warning_dispatch(escalated, target_user_id="capt_fernandes_01")
    delivered_3 = [r for r in records_3 if r.delivery_status == DeliveryStatus.DELIVERED]
    assert len(delivered_3) >= 1
    assert any("ESCALATED" in r.message for r in delivered_3)

# =============================================================================
# 3. SAFETY: FAILED REFRESHES NOT INTERPRETED AS NO HAZARD
# =============================================================================

@pytest.mark.anyio
async def test_m15_failed_provider_refreshes_not_interpreted_as_no_hazard():
    """
    CRITICAL SAFETY REQUIREMENT:
    When upstream providers fail/timeout and no observations can be fetched,
    the system must NOT claim that risk is LOW or conditions are safe to sail.
    It must return UNKNOWN risk with safe guidance.
    """
    # Clear official bulletins so we test pure provider failure path
    OfficialWarningProvider().clear()
    hazard_svc = HazardAssessmentService()

    with patch("app.connectors.open_meteo_weather.OpenMeteoWeatherConnector.fetch_weather", new_callable=AsyncMock) as mock_w, \
         patch("app.connectors.open_meteo_marine.OpenMeteoMarineConnector.fetch_marine", new_callable=AsyncMock) as mock_m:

        from app.connectors.base_connector import ConnectorResult
        mock_w.return_value = ConnectorResult(success=False, error="Weather gateway unreachable (504 Gateway Timeout)")
        mock_m.return_value = ConnectorResult(success=False, error="Marine gateway unreachable (503 Service Unavailable)")

        assessment = await hazard_svc.assess_hazard(lat=15.3493, lon=73.8000, location_name="Panaji Offshore")

        # Must NOT claim LOW risk or safe to sail!
        assert assessment.overall_risk_level == "UNKNOWN"
        assert assessment.is_safe_to_sail is False
        assert "DATA UNAVAILABLE" in assessment.safety_guidance or "cannot be verified" in assessment.safety_guidance
        assert len(assessment.uncertainty_notes) >= 2

@pytest.mark.anyio
async def test_m15_failed_refreshes_fallback_to_offline_cache():
    """
    When live providers fail, if a valid cached snapshot is present in
    OfflineMarineCache, the system evaluates hazard using cached data.
    """
    OfficialWarningProvider().clear()
    hazard_svc = HazardAssessmentService()
    cache = OfflineMarineCache()

    # Pre-cache a high-swell observation
    cache.store_observation(
        lat=15.3493,
        lon=73.8000,
        data={
            "wave_height_m": 3.8,  # High swell
            "wind_speed_kmh": 45.0, # Gale wind
            "location_name": "Goa Coastal Grid"
        },
        location_name="Goa Coastal Grid"
    )

    with patch("app.connectors.open_meteo_weather.OpenMeteoWeatherConnector.fetch_weather", new_callable=AsyncMock) as mock_w, \
         patch("app.connectors.open_meteo_marine.OpenMeteoMarineConnector.fetch_marine", new_callable=AsyncMock) as mock_m:

        from app.connectors.base_connector import ConnectorResult
        mock_w.return_value = ConnectorResult(success=False, error="Connection timeout")
        mock_m.return_value = ConnectorResult(success=False, error="Connection timeout")

        assessment = await hazard_svc.assess_hazard(lat=15.3493, lon=73.8000, location_name="Goa Coastal Grid")

        # Successfully evaluated based on cached data
        assert assessment.overall_risk_level in ["HIGH", "DANGER", "MODERATE"]
        assert assessment.is_safe_to_sail is False
        assert any("cached" in note.lower() for note in assessment.uncertainty_notes)
        assert len(assessment.derived_indicators) >= 1

# =============================================================================
# 4. DELIVERY: RESTART PERSISTENCE & PERMISSION DENIED
# =============================================================================

@pytest.mark.anyio
async def test_m15_alert_persistence_across_restart():
    """
    Verifies that subscriptions and alert notification delivery history
    persist in SQLite across service restarts (in-memory clearing).
    """
    rules = AlertRulesEngine()
    notif_svc = NotificationDeliveryService()
    test_user = f"pilot_user_{uuid.uuid4().hex[:8]}"

    # 1. Register subscription
    sub = rules.add_subscription(
        AlertSubscriptionRequest(
            user_id=test_user,
            device_token="device_token_xyz_99",
            latitude=15.2993,
            longitude=74.1240,
            radius_km=40.0,
            min_severity="WARNING",
            enable_push=True,
            enable_in_app=True
        )
    )

    # 2. Record notification
    now = datetime.datetime.now(datetime.timezone.utc)
    b = OfficialWarningDetail(
        warning_id=f"WARN-{uuid.uuid4().hex[:6]}",
        source="IMD",
        warning_type="ROUGH_SEA",
        source_severity="ORANGE WARNING",
        normalized_severity="WARNING",
        bulletin_text="Squally weather testing persistence.",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        valid_until=(now + datetime.timedelta(hours=12)).isoformat(),
        affected_coordinates={"latitude": 15.2993, "longitude": 74.1240},
        coverage_radius_km=50.0,
        status=AlertStatus.ACTIVE,
        is_official=True
    )
    rec = await notif_svc.deliver_in_app(
        user_id=test_user,
        warning=b,
        idempotency_key=f"idemp_{uuid.uuid4().hex[:12]}",
        message="In-app notification test."
    )

    # 3. Simulate Server Restart by clearing in-memory caches
    rules._subscriptions.clear()
    rules._user_subscriptions.clear()
    notif_svc._notifications.clear()
    notif_svc._user_index.clear()

    # 4. Lookup subscriptions -> Successfully recovered from SQLite
    restored_subs = rules.get_user_subscriptions(test_user)
    assert len(restored_subs) >= 1
    assert restored_subs[0].user_id == test_user
    assert restored_subs[0].latitude == 15.2993

    # 5. Lookup history -> Successfully recovered from SQLite
    restored_history = notif_svc.get_user_history(test_user)
    assert len(restored_history) >= 1
    assert restored_history[0].user_id == test_user
    assert restored_history[0].delivery_channel == "IN_APP"

@pytest.mark.anyio
async def test_m15_permission_denied_delivery_status():
    """
    Verifies that when a user denies notification permissions,
    the delivery service returns PERMISSION_DENIED without failing silently.
    """
    notif_svc = NotificationDeliveryService()
    denied_user = "denied_sailor_001"
    notif_svc.set_user_permission_denied(denied_user, denied=True)

    now = datetime.datetime.now(datetime.timezone.utc)
    b = OfficialWarningDetail(
        warning_id="WARN-DENIED-01",
        source="IMD",
        warning_type="CYCLONE",
        source_severity="RED WARNING",
        normalized_severity="DANGER",
        bulletin_text="Cyclone alert test.",
        issue_timestamp=now.isoformat(),
        valid_from=now.isoformat(),
        valid_until=(now + datetime.timedelta(hours=6)).isoformat(),
        affected_coordinates={"latitude": 15.29, "longitude": 74.12},
        coverage_radius_km=50.0,
        status=AlertStatus.ACTIVE,
        is_official=True
    )

    rec = await notif_svc.deliver_push(
        user_id=denied_user,
        device_token="device_token_abc",
        warning=b,
        idempotency_key=f"idemp_{uuid.uuid4().hex[:8]}",
        message="Push test"
    )

    assert rec.delivery_status == DeliveryStatus.PERMISSION_DENIED
    assert "denied OS notification permission" in rec.provider_response

# =============================================================================
# 5. COORDINATE VALIDATION & LOG REDACTION
# =============================================================================

def test_m15_coordinate_validation_rejection():
    """
    Verifies that invalid latitude and longitude parameters are rejected
    with HTTP 400 on both /api/hazard/assessment and /api/hazard/warnings.
    """
    # 1. Invalid latitude > 90.0
    r1 = client.get("/api/hazard/assessment", params={"latitude": 98.5, "longitude": 73.8})
    assert r1.status_code == 400
    assert "Invalid latitude" in r1.json()["detail"]

    # 2. Invalid longitude < -180.0
    r2 = client.get("/api/hazard/assessment", params={"latitude": 15.3, "longitude": -195.0})
    assert r2.status_code == 400
    assert "Invalid longitude" in r2.json()["detail"]

    # 3. Invalid coordinates on /api/hazard/warnings
    r3 = client.get("/api/hazard/warnings", params={"latitude": -95.0, "longitude": 73.8})
    assert r3.status_code == 400
    assert "Invalid latitude" in r3.json()["detail"]

def test_m15_sensitive_data_log_redaction():
    """
    Verifies that SensitiveDataRedactor scrubs credentials, tokens,
    API keys and secrets from log messages.
    """
    redactor = SensitiveDataRedactor()
    raw_log = "Connection to IMD failed: api_key='SECRET_KEY_9988' and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz"
    scrubbed = redactor.redact_text(raw_log)

    assert "SECRET_KEY_9988" not in scrubbed
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in scrubbed
    assert "[REDACTED]" in scrubbed
    assert "[REDACTED_TOKEN]" in scrubbed
