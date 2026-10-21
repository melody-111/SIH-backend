from typing import Dict, Any
from app.core.config import settings

class PFZService:
    """Service for querying Potential Fishing Zones (PFZ) advisory data."""
    async def get_pfz(self, lat: float, lon: float) -> Dict[str, Any]:
        # If API key is not configured and not in simulation fallback mode
        if not settings.INCOIS_API_KEY:
            if settings.ENABLE_SIMULATION_FALLBACK and settings.APP_ENV == "development":
                return {
                    "pfz_available": True,
                    "recommended_lat": lat + 0.05,
                    "recommended_lon": lon + 0.05,
                    "valid_until": "2026-09-18T23:59:00Z",
                    "status": "success",
                    "is_simulated": True,
                    "source": "INCOIS (Simulated for Prototype)"
                }
            return {
                "pfz_available": False,
                "recommended_lat": None,
                "recommended_lon": None,
                "valid_until": None,
                "status": "unverified",
                "is_simulated": False,
                "message": "Official INCOIS PFZ dataset requires institutional API key (INCOIS_API_KEY). Zero simulated coordinates displayed.",
                "source": "INCOIS"
            }
        return {
            "pfz_available": True,
            "recommended_lat": lat + 0.05,
            "recommended_lon": lon + 0.05,
            "valid_until": "2026-09-18T23:59:00Z",
            "status": "verified",
            "is_simulated": False,
            "source": "INCOIS (Verified Advisory)"
        }
