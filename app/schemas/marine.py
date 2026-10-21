from pydantic import BaseModel
from typing import Dict, Any, Optional

class MarineContext(BaseModel):
    location: Optional[Dict[str, Any]] = None
    time: Optional[str] = None
    weather: Optional[Dict[str, Any]] = None
    ocean: Optional[Dict[str, Any]] = None
    pfz: Optional[Dict[str, Any]] = None
    hazards: Optional[Dict[str, Any]] = None
    hazard: Optional[Dict[str, Any]] = None
    gis: Optional[Dict[str, Any]] = None
    sst: Optional[Dict[str, Any]] = None
    chlorophyll: Optional[Dict[str, Any]] = None
    satellite: Optional[Dict[str, Any]] = None
    route: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
