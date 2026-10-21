from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import datetime

class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    UPDATED = "UPDATED"
    ESCALATED = "ESCALATED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

class DeliveryStatus(str, Enum):
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    SUPPRESSED_DUPLICATE = "SUPPRESSED_DUPLICATE"
    SUPPRESSED_EXPIRED = "SUPPRESSED_EXPIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"

class OfficialWarningDetail(BaseModel):
    warning_id: str
    source: str  # e.g., "IMD", "INCOIS", "Port Authority"
    warning_type: str  # e.g., "CYCLONE_ALERT", "HIGH_WAVE", "STORM_SURGE", "HEAVY_RAINFALL"
    source_severity: str  # Original term: "RED ALERT", "ORANGE ALERT", "WARNING"
    normalized_severity: str  # "INFO", "ADVISORY", "WATCH", "WARNING", "DANGER"
    bulletin_text: str
    issue_timestamp: str  # ISO 8601 UTC
    valid_from: str
    valid_until: Optional[str] = None
    affected_regions: List[str] = []
    affected_coordinates: Optional[Dict[str, float]] = None
    coverage_radius_km: float = 100.0
    status: AlertStatus = AlertStatus.ACTIVE
    is_official: bool = True

class DerivedHazardIndicator(BaseModel):
    indicator_type: str  # e.g., "GALE_WIND", "SQUALL_GUST", "HIGH_SWELL", "BAROMETRIC_DEPRESSION"
    measured_value: float
    threshold: float
    unit: str
    severity: str  # "ADVISORY", "WARNING", "DANGER"
    message: str
    source_model: str  # e.g., "Open-Meteo Observations / Copernicus ECMWF"
    observation_time: str

class PreventionMeasureItem(BaseModel):
    priority: str
    action: str
    details: str
    status: str
    icon: Optional[str] = "⚓"

class EpicenterCoordinates(BaseModel):
    latitude: float
    longitude: float
    name: Optional[str] = "Hazard Epicenter"
    distance_km: Optional[float] = None
    bearing_degrees: Optional[float] = None

class HazardAssessmentResponse(BaseModel):
    latitude: float
    longitude: float
    location_name: Optional[str] = None
    assessment_timestamp: str
    overall_risk_level: str  # "LOW", "MODERATE", "HIGH", "DANGER"
    is_safe_to_sail: bool
    official_warnings: List[OfficialWarningDetail] = []
    derived_indicators: List[DerivedHazardIndicator] = []
    active_warning_count: int = 0
    expired_warning_count: int = 0
    data_freshness_minutes: float = 0.0
    safety_guidance: str
    uncertainty_notes: List[str] = []
    disaster_probability: float = 0.0
    disaster_type: str = "NONE"
    epicenter_coordinates: Optional[EpicenterCoordinates] = None
    prevention_measures: List[PreventionMeasureItem] = []

class AlertSubscriptionRequest(BaseModel):
    user_id: str
    region_name: Optional[str] = None
    device_token: Optional[str] = None
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    radius_km: float = Field(50.0, ge=5.0, le=500.0)
    min_severity: str = "ADVISORY"
    enable_push: bool = True
    enable_in_app: bool = True

class AlertSubscription(BaseModel):
    subscription_id: str
    user_id: str
    device_token: Optional[str] = None
    latitude: float
    longitude: float
    radius_km: float
    min_severity: str
    enable_push: bool = True
    enable_in_app: bool = True
    created_at: str

class AlertNotificationRecord(BaseModel):
    notification_id: str
    warning_id: str
    user_id: str
    idempotency_key: str
    delivery_channel: str  # "PUSH", "IN_APP", "SMS"
    delivery_status: DeliveryStatus
    message: str
    sent_timestamp: str
    provider_response: Optional[str] = None

class CacheSyncRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    location_name: Optional[str] = None
    past_days: int = Field(2, ge=0, le=5)
    forecast_days: int = Field(2, ge=0, le=5)
    last_sync_timestamp: Optional[str] = None

class CacheSyncResponse(BaseModel):
    latitude: float
    longitude: float
    sync_timestamp: str
    cached_window_past_days: int
    cached_window_forecast_days: int
    records_cached_count: int
    status: str  # "FRESH", "UP_TO_DATE", "PARTIAL", "FAILED"
    is_cached: bool = True
    freshness_age_minutes: float = 0.0
    active_warnings: List[OfficialWarningDetail] = []
