from typing import Dict, Any
from app.services.marine_data_service import MarineDataService
from app.core.config import settings

class APIHealthService:
    def __init__(self):
        self.marine_data_service = MarineDataService()

    def get_quick_provider_status(self) -> Dict[str, Any]:
        """
        Fast synchronous evaluation of configured external provider statuses
        without making outbound network calls.
        Distinguishes configured, public, and credential-requiring providers.
        """
        providers = {
            "open_meteo": {
                "name": "Open-Meteo Marine & Weather",
                "tier": "public_operational",
                "status": "OPERATIONAL",
                "requires_auth": False,
                "is_configured": True
            },
            "nasa_earthdata": {
                "name": "NASA Earthdata CMR",
                "tier": "public_metadata",
                "status": "OPERATIONAL",
                "requires_auth": False,
                "is_configured": True
            },
            "incois": {
                "name": "INCOIS Ocean State & PFZ",
                "tier": "institutional_authorized",
                "status": "CONFIGURED" if bool(settings.INCOIS_API_KEY) else "UNCONFIGURED_REQUIRES_AUTH",
                "requires_auth": True,
                "is_configured": bool(settings.INCOIS_API_KEY)
            },
            "imd": {
                "name": "India Meteorological Department",
                "tier": "institutional_authorized",
                "status": "CONFIGURED" if bool(settings.IMD_API_KEY) else "UNCONFIGURED_REQUIRES_AUTH",
                "requires_auth": True,
                "is_configured": bool(settings.IMD_API_KEY)
            },
            "sarvam_ai": {
                "name": "Sarvam AI Voice Services",
                "tier": "cloud_ai",
                "status": "CONFIGURED" if bool(settings.SARVAM_API_KEY) else "OPTIONAL_UNCONFIGURED",
                "requires_auth": True,
                "is_configured": bool(settings.SARVAM_API_KEY)
            }
        }

        # Operational status is at least PARTIAL if open-meteo is available
        auth_count = sum(1 for p in providers.values() if p["is_configured"])
        total_count = len(providers)
        
        return {
            "status": "OPERATIONAL" if auth_count == total_count else "PARTIAL",
            "configured_count": auth_count,
            "total_count": total_count,
            "providers": providers
        }

    async def check_health(self) -> Dict[str, Any]:
        statuses = await self.marine_data_service.check_all_sources_health()
        return {k: v.model_dump() for k, v in statuses.items()}
