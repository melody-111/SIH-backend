from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from app.services.ocean_service import OceanService
from typing import Dict, Any
import datetime

class OceanAgent(BaseAgent):
    def __init__(self):
        super().__init__("ocean", "Retrieves marine and ocean conditions including real SST and waves")
        self.service = OceanService()

    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        lat = input_data.get("latitude", 0.0)
        lon = input_data.get("longitude", 0.0)
        corr_id = input_data.get("correlation_id", "req_default")

        data = await self.service.get_ocean_conditions(lat, lon)
        status = data.pop("status", "unknown")
        source = data.pop("source", "Open-Meteo Marine API")
        now = datetime.datetime.utcnow().isoformat() + "Z"

        return AgentResult(
            agent=self.name,
            status="SUCCESS" if status == "success" else "UNAVAILABLE",
            data=data,
            confidence=0.95 if status == "success" else 0.0,
            sources=[source],
            timestamp=now,
            correlation_id=corr_id,
            data_type="OBSERVATION",
            units={
                "wave_height": "m",
                "wave_direction": "°",
                "wave_period": "s",
                "current_velocity": "km/h",
                "sst": "°C"
            },
            quality_status="VALID" if status == "success" else "UNAVAILABLE"
        )
