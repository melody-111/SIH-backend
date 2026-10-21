from pydantic import BaseModel
from typing import Optional, Dict, Any

class MarineObservation(BaseModel):
    source: str
    provider: str
    timestamp: str
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    latitude: float
    longitude: float
    value: Any
    unit: str
    quality: Optional[str] = "unknown"
    confidence: float

class WeatherObservation(MarineObservation):
    parameter: str # e.g., temperature, wind_speed

class OceanObservation(MarineObservation):
    parameter: str # e.g., wave_height, current

class SSTObservation(MarineObservation):
    parameter: str = "sst"

class ChlorophyllObservation(MarineObservation):
    parameter: str = "chlorophyll_a"

class PFZObservation(MarineObservation):
    is_active: bool
    
class HazardObservation(BaseModel):
    hazard_type: str
    severity: str
    location: Dict[str, float]
    valid_from: str
    valid_until: str
    source: str
    confidence: float

class EvidenceItem(BaseModel):
    claim: str
    source: str
    measurement: str
    value: Any
    unit: str
    timestamp: str
    confidence: float
