from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from app.services.earthdata_service import EarthdataService
from app.services.ocean_service import OceanService
from typing import Dict, Any
import datetime

class SSTAgent(BaseAgent):
    def __init__(self):
        super().__init__("sst", "Retrieves Sea Surface Temperature data")
        self.service = EarthdataService()
        self.ocean_service = OceanService()
        
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        lat = input_data.get("latitude", 0.0)
        lon = input_data.get("longitude", 0.0)
        
        # 1. Real measured SST from Open-Meteo Marine
        ocean_data = await self.ocean_service.get_ocean_conditions(lat, lon)
        real_sst = ocean_data.get("sst")
        
        # 2. NASA Earthdata CMR satellite collections for SST
        data = await self.service.search_collections("SST")
        status = data.pop("status", "unknown")
        source = data.pop("source", "NASA Earthdata CMR")
        
        sources = [source]
        if ocean_data.get("status") == "success":
            sources.append("Open-Meteo Marine API")
            
        data_payload = {
            "collections_found": len(data.get("data", [])),
            "sea_surface_temperature": real_sst,
            "unit": "°C",
            "satellite_source": "NASA Earthdata MODIS/VIIRS",
            "marine_source": "Open-Meteo Marine"
        }
        
        effective_status = status if status == "success" else ("SUCCESS" if real_sst is not None else "UNAVAILABLE")
        
        return AgentResult(
            agent=self.name,
            status=effective_status,
            data=data_payload,
            confidence=0.9 if (status == "success" or real_sst is not None) else 0.0,
            sources=sources,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            units={"sea_surface_temperature": "°C"}
        )

