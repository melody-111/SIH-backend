from typing import Dict, Any, Optional
from app.services.marine_gis_service import MarineGISService

class GISService:
    """
    Backwards-compatible wrapper around MarineGISService.
    Replaces earlier placeholder implementation.
    """
    def __init__(self):
        self._service = MarineGISService()

    async def get_gis_context(
        self,
        user_lat: float,
        user_lon: float,
        target_lat: Optional[float] = None,
        target_lon: Optional[float] = None
    ) -> Dict[str, Any]:
        result = {
            "user_location": {"lat": user_lat, "lon": user_lon},
            "status": "success",
            "source": "Marine GIS Service"
        }
        if target_lat is not None and target_lon is not None:
            dist = self._service._haversine_distance(user_lat, user_lon, target_lat, target_lon) if hasattr(self._service, "_haversine_distance") else None
            if dist is None:
                import math
                dlat = math.radians(target_lat - user_lat)
                dlon = math.radians(target_lon - user_lon)
                a = math.sin(dlat/2)**2 + math.cos(math.radians(user_lat)) * math.cos(math.radians(target_lat)) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                dist = 6371.0 * c
            result["distance_km"] = round(dist, 2)
            result["target_location"] = {"lat": target_lat, "lon": target_lon}
        return result
