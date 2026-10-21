from fastapi import APIRouter, Query, HTTPException, status
from typing import Optional, List, Dict
from app.services.marine_data_service import MarineDataService
from app.schemas.marine_normalized import (
    MarineQueryRequest,
    NormalizedMarineSnapshot,
    MarineWarningRecord,
    DataSourceStatus,
    MarineTask
)

router = APIRouter()
marine_service = MarineDataService()

@router.get(
    "/marine",
    response_model=NormalizedMarineSnapshot,
    summary="Get current normalized marine conditions for coordinates"
)
@router.get(
    "/marine/conditions",
    response_model=NormalizedMarineSnapshot,
    summary="Get detailed marine conditions with forecast"
)
@router.get(
    "/marine/snapshot",
    response_model=NormalizedMarineSnapshot,
    summary="Get snapshot of normalized marine conditions"
)
async def get_marine_conditions(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Latitude between -90 and 90"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Longitude between -180 and 180"),
    location_name: Optional[str] = Query(None, description="Optional coastal location name"),
    past_days: int = Query(0, ge=0, le=7, description="Past observation days (0-7)"),
    forecast_days: int = Query(2, ge=0, le=7, description="Forecast days (0-7)")
):
    try:
        query = MarineQueryRequest(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            intended_task=MarineTask.GENERAL,
            past_days=past_days,
            forecast_days=forecast_days
        )
        snapshot = await marine_service.get_marine_snapshot(query)
        return snapshot
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve marine conditions: {str(e)}"
        )

@router.post(
    "/marine/query",
    response_model=NormalizedMarineSnapshot,
    summary="Task-prioritized marine data query"
)
async def query_marine_data(request: MarineQueryRequest):
    try:
        snapshot = await marine_service.get_marine_snapshot(request)
        return snapshot
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing marine query: {str(e)}"
        )

@router.get(
    "/marine/warnings",
    response_model=List[MarineWarningRecord],
    summary="Get active weather, wave, and maritime warnings for location"
)
async def get_marine_warnings(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    location_name: Optional[str] = Query(None)
):
    try:
        query = MarineQueryRequest(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            intended_task=MarineTask.HAZARDS_WARNINGS,
            forecast_days=1
        )
        snapshot = await marine_service.get_marine_snapshot(query)
        return snapshot.active_warnings
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve warnings: {str(e)}"
        )

@router.get(
    "/marine/sources",
    response_model=Dict[str, DataSourceStatus],
    summary="Check real-time health, latency, and auth status of marine sources"
)
async def get_sources_status():
    try:
        statuses = await marine_service.check_all_sources_health()
        return statuses
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check sources status: {str(e)}"
        )
