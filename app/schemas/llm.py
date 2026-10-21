from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import datetime

class TaskType(str, Enum):
    INTENT_CLASSIFICATION = "INTENT_CLASSIFICATION"
    QUERY_PLANNING = "QUERY_PLANNING"
    EVIDENCE_SUMMARIZATION = "EVIDENCE_SUMMARIZATION"
    FINAL_SYNTHESIS = "FINAL_SYNTHESIS"

class LLMTelemetryRecord(BaseModel):
    task: str
    provider: str
    model: str
    status: str  # SUCCESS, FAILED, FALLBACK
    duration_ms: float
    tokens_used: Optional[Dict[str, int]] = None
    cost_usd: Optional[float] = None
    correlation_id: str
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

class IntentClassificationOutput(BaseModel):
    intent: str = Field(..., description="Classified intent (e.g. HAZARD, FISHING_LOCATION, OCEAN_INFO, MARINE_CONDITION, ROUTE, GENERAL_EXPLANATORY, FOLLOW_UP)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0")
    reasoning: str = Field(default="", description="Brief justification for classification")
    secondary_intents: List[str] = Field(default_factory=list, description="Optional secondary intents")

class QueryPlanOutput(BaseModel):
    selected_agents: List[str] = Field(default_factory=list, description="Ordered list of specialist agents to execute")
    rationale: str = Field(default="", description="Reason for agent selection")
    requires_live_sensors: bool = Field(default=True, description="Whether live telemetry or external APIs are needed")

class SynthesizedAdvisoryOutput(BaseModel):
    answer: str = Field(..., description="Clear natural language response in user's language")
    language: str = Field(default="en", description="BCP-47 language tag (e.g. hi, en)")
    risk_assessment: str = Field(default="LOW", description="Overall risk level: LOW, MODERATE, HIGH, UNKNOWN")
    recommended_coordinates: Optional[str] = Field(default=None, description="Optional lat,lon string if verified PFZ exists")
    safety_guidance: List[str] = Field(default_factory=list, description="Actionable safety points for fishermen")
    caveats: List[str] = Field(default_factory=list, description="Limitations, missing data warnings, or cache disclosures")
    map_action_type: str = Field(default="center", description="Map action type: center, highlight, corridor")
