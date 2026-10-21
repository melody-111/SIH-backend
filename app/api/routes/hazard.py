from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from app.schemas.hazard_alert import (
    HazardAssessmentResponse,
    OfficialWarningDetail
)
from app.services.hazard_assessment_service import HazardAssessmentService
from app.services.official_warning_provider import OfficialWarningProvider

router = APIRouter(prefix="/hazard", tags=["hazard"])

hazard_service = HazardAssessmentService()
warning_provider = OfficialWarningProvider()

@router.get("/assessment", response_model=HazardAssessmentResponse)
async def get_hazard_assessment(
    latitude: float = Query(..., description="Latitude between -90.0 and 90.0"),
    longitude: float = Query(..., description="Longitude between -180.0 and 180.0"),
    location_name: Optional[str] = Query(None, description="Optional coastal location name")
):
    """
    Returns structured location-aware marine hazard assessment combining
    official government bulletins and verified physical model indicators.
    """
    try:
        return await hazard_service.assess_hazard(
            lat=latitude,
            lon=longitude,
            location_name=location_name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hazard assessment failed: {str(e)}")

@router.get("/warnings", response_model=List[OfficialWarningDetail])
@router.get("/bulletins", response_model=List[OfficialWarningDetail])
async def get_official_warnings(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
    region_name: Optional[str] = Query(None, description="Optional region name"),
    include_expired: bool = Query(False, description="Include expired or cancelled warnings")
):
    """
    Returns verified official government warnings covering the specified location or region.
    """
    try:
        return warning_provider.get_warnings_for_location(
            lat=latitude,
            lon=longitude,
            region_name=region_name,
            include_expired=include_expired
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed retrieving warnings: {str(e)}")

@router.get("/disasters")
async def get_live_disasters(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
    radius_km: float = Query(1000.0, description="Search radius in kilometers, default 1000km"),
    location_name: Optional[str] = Query(None, description="Optional location name (e.g. Lucknow)")
):
    """
    Returns real-time global natural disasters (cyclones, earthquakes, tsunamis, floods)
    within the specified radius (default 1000 km) and pinpoints nearest global cyclone.
    """
    try:
        from app.services.disaster_intelligence_service import DisasterIntelligenceService
        disaster_svc = DisasterIntelligenceService()
        return await disaster_svc.fetch_disasters_around_location(
            lat=latitude,
            lon=longitude,
            radius_km=radius_km,
            location_name=location_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed retrieving live disaster data: {str(e)}")

