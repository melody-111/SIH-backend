from fastapi import APIRouter, Request, Query
from typing import Optional
from app.services.location_detection_service import LocationDetectionService

router = APIRouter(prefix="/location", tags=["location"])

loc_service = LocationDetectionService()

@router.get("/detect")
async def detect_user_location(request: Request):
    """
    Detects user's real-time physical location, city, state, country,
    and coordinates (latitude, longitude) from client IP network.
    """
    # Check X-Forwarded-For or client host
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else None

    return await loc_service.detect_location(client_ip=client_ip)
