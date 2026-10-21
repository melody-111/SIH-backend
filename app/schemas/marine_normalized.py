from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import datetime

class DataType(str, Enum):
    OBSERVATION = "OBSERVATION"
    FORECAST = "FORECAST"
    ADVISORY = "ADVISORY"
    DERIVED_ESTIMATE = "DERIVED_ESTIMATE"
    SATELLITE_CATALOG = "SATELLITE_CATALOG"

class QualityStatus(str, Enum):
    VALID = "VALID"
    SUSPECT = "SUSPECT"
    UNAVAILABLE = "UNAVAILABLE"
    OFFLINE = "OFFLINE"

class MarineTask(str, Enum):
    GENERAL = "GENERAL"
    FISHING = "FISHING"
    MARINE_WEATHER = "MARINE_WEATHER"
    HAZARDS_WARNINGS = "HAZARDS_WARNINGS"
    NAVIGATION_ROUTE = "NAVIGATION_ROUTE"

class WarningSeverity(str, Enum):
    INFO = "INFO"
    ADVISORY = "ADVISORY"
    WATCH = "WATCH"
    WARNING = "WARNING"
    DANGER = "DANGER"

class NormalizedMarineRecord(BaseModel):
    """
    Standardized atomic measurement or forecast point for any marine variable.
    """
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    location_name: Optional[str] = None
    variable_name: str
    value: Optional[Any] = None
    unit: str
    data_type: DataType = DataType.OBSERVATION
    timestamp: str  # ISO 8601 UTC observation or target forecast time
    retrieval_timestamp: str  # ISO 8601 UTC when fetched by server
    source_provider: str
    forecast_horizon_hours: Optional[float] = None
    quality_status: QualityStatus = QualityStatus.VALID
    is_missing: bool = False
    notes: Optional[str] = None

class ConnectorStatus(str, Enum):
    LIVE_VERIFIED = "LIVE_VERIFIED"
    CONFIGURED_UNVERIFIED = "CONFIGURED_UNVERIFIED"
    METADATA_ONLY = "METADATA_ONLY"
    MOCKED = "MOCKED"
    REQUIRES_ACCESS = "REQUIRES_ACCESS"
    UNAVAILABLE = "UNAVAILABLE"

class MarineWarningRecord(BaseModel):
    """
    Structured alert / warning record normalized across official agencies and models.
    """
    warning_type: str  # e.g., GALE_WIND, HIGH_WAVE, CYCLONE, THUNDERSTORM, FISHERMEN_WARNING
    severity: WarningSeverity
    message: str
    source: str
    is_official: bool = False
    valid_from: str
    valid_until: Optional[str] = None
    affected_coordinates: Optional[Dict[str, float]] = None
    provider: str = "IMD"
    official_bulletin_type: Optional[str] = None
    issue_time: Optional[str] = None
    affected_geographic_area: Optional[str] = None
    severity_as_supplied: Optional[str] = None
    original_source_reference: Optional[str] = None
    retrieval_time: Optional[str] = None
    verification_status: str = "OFFICIAL_VERIFIED"

class DataSourceStatus(BaseModel):
    """
    Availability, latency, and status of an external provider.
    """
    provider_name: str
    endpoint: str
    status: str  # OPERATIONAL, DEGRADED, REQUIRES_AUTH, OFFLINE
    connector_status: Optional[ConnectorStatus] = None
    requires_auth: bool = False
    is_authenticated: bool = False
    last_checked: str
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None

class MarineQueryRequest(BaseModel):
    """
    Structured request for query-driven marine data retrieval.
    """
    question: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    location_name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    intended_task: MarineTask = MarineTask.GENERAL
    past_days: int = Field(0, ge=0, le=7)
    forecast_days: int = Field(1, ge=0, le=7)

class NormalizedMarineSnapshot(BaseModel):
    """
    Consolidated, normalized marine data snapshot for an area and point in time,
    ready for multi-agent reasoning, map layers, or mobile consumption.
    """
    latitude: float
    longitude: float
    location_name: Optional[str] = None
    is_location_ambiguous: bool = False
    target_time: str
    retrieval_time: str
    
    # Categorized normalized observations
    weather_records: List[NormalizedMarineRecord] = []
    ocean_records: List[NormalizedMarineRecord] = []
    satellite_records: List[NormalizedMarineRecord] = []
    advisory_records: List[NormalizedMarineRecord] = []
    
    # Active hazards and alerts
    active_warnings: List[MarineWarningRecord] = []
    
    # Provider health and attribution
    source_statuses: Dict[str, DataSourceStatus] = {}
    
    # Machine-readable evidence items for AI Agents
    evidence_items: List[Dict[str, Any]] = []
    
    # Unavailable or failed sources
    unavailable_sources: List[str] = []
