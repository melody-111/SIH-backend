from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from app.services.mosdac_service import MOSDACService
from typing import Dict, Any
import datetime

class ChlorophyllAgent(BaseAgent):
    def __init__(self):
        super().__init__("chlorophyll", "Retrieves Chlorophyll concentration")
        self.service = MOSDACService()
        
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        lat = input_data.get("latitude", 0.0)
        lon = input_data.get("longitude", 0.0)
        
        data = await self.service.get_chlorophyll(lat, lon)
        status = data.pop("status", "unknown")
        source = data.pop("source", "unknown")
        
        return AgentResult(
            agent=self.name,
            status=status,
            data=data,
            confidence=0.90 if status == "success" else 0.0,
            sources=[source],
            timestamp=datetime.datetime.utcnow().isoformat()
        )
