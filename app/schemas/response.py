from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ProgressStep(BaseModel):
    step: str
    status: str = "COMPLETED"  # "STARTED", "COMPLETED", "FAILED", "SKIPPED"
    agent: Optional[str] = None
    timestamp: str = ""
    details: Optional[str] = None

class FinalResponse(BaseModel):
    answer: str
    language: str
    risk: str  # "LOW", "MODERATE", "HIGH", "UNKNOWN"
    recommended_location: Optional[str] = None
    reasons: List[str] = []
    warnings: List[str] = []
    evidence: List[str] = []
    map_actions: List[Dict[str, Any]] = []
    agents_used: List[str] = []
    timestamp: str
    
    # Progress and orchestrator metadata
    progress_steps: List[ProgressStep] = []
    execution_plan: List[str] = []
    is_cached: bool = False
    cached_at: Optional[str] = None
    source_attributions: List[Dict[str, Any]] = []
    configuration_version: Optional[str] = "1.0.0"
    mode: Optional[str] = "CHAT"  # "CHAT", "AGENT", "HYBRID"
    provider_info: Optional[Dict[str, Any]] = None

