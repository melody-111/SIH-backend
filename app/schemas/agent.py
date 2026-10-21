from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class AgentResult(BaseModel):
    agent: str
    status: str  # "SUCCESS", "PARTIAL", "FAILED", "REQUIRES_AUTH", "UNAVAILABLE", "SKIPPED"
    data: Dict[str, Any] = {}
    confidence: float = 1.0
    sources: List[str] = []
    timestamp: str
    correlation_id: Optional[str] = None
    data_type: Optional[str] = None  # "OBSERVATION", "FORECAST", "ADVISORY", "DERIVED_ESTIMATE"
    units: Optional[Dict[str, str]] = None
    geographical_coverage: Optional[Dict[str, Any]] = None
    quality_status: str = "VALID"
    missing_fields: List[str] = []
    errors: List[str] = []
    notes: Optional[str] = None

