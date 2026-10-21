from typing import Dict, Any, Optional
from app.services.hazard_assessment_service import HazardAssessmentService

class HazardService:
    """
    Backwards-compatible wrapper around HazardAssessmentService.
    Replaces earlier placeholder implementation with real location-aware hazard evaluation.
    """
    def __init__(self):
        self._service = HazardAssessmentService()

    async def get_hazards(self, lat: float, lon: float, location_name: Optional[str] = None) -> Dict[str, Any]:
        assessment = await self._service.assess_hazard(lat, lon, location_name)
        return assessment.model_dump()
