from typing import Dict, Any

class INCOISService:
    async def get_ocean_state(self, lat: float, lon: float) -> Dict[str, Any]:
        return {
            "status": "PENDING_VERIFICATION",
            "error": "Requires API Token",
            "source": "INCOIS"
        }
