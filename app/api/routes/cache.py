from fastapi import APIRouter, Query, Body, HTTPException
from typing import Dict, Any
from app.schemas.hazard_alert import (
    CacheSyncRequest,
    CacheSyncResponse
)
from app.services.offline_cache import OfflineMarineCache

router = APIRouter(prefix="/cache", tags=["cache"])

cache_service = OfflineMarineCache()

@router.post("/sync", response_model=CacheSyncResponse)
async def sync_marine_cache(request: CacheSyncRequest):
    """
    Downloads and caches 2-day past and 2-day forecast data for offline availability.
    Supports incremental synchronization via last_sync_timestamp.
    """
    try:
        return await cache_service.sync_cache_window(
            lat=request.latitude,
            lon=request.longitude,
            location_name=request.location_name or "",
            past_days=request.past_days,
            forecast_days=request.forecast_days,
            last_sync_timestamp=request.last_sync_timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache synchronization failed: {str(e)}")

@router.get("/freshness")
async def get_cache_freshness(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0)
):
    """
    Checks cache age, staleness, and available offline records for location.
    """
    return cache_service.get_offline_data(latitude, longitude)

@router.post("/toggle-offline")
async def toggle_offline_mode(payload: Dict[str, bool] = Body(...)):
    """
    Simulates offline device state for testing disconnected marine capabilities.
    """
    enabled = payload.get("enabled", False)
    cache_service.set_force_offline(enabled)
    return {
        "status": "success",
        "forced_offline": cache_service.is_forced_offline,
        "message": "Offline simulation mode enabled" if enabled else "Online mode restored"
    }
