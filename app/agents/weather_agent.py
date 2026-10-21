from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from app.services.weather_service import WeatherService
from typing import Dict, Any
import datetime

class WeatherAgent(BaseAgent):
    def __init__(self):
        super().__init__("weather", "Retrieves weather conditions for a given location")
        self.service = WeatherService()

    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        lat = input_data.get("latitude", 0.0)
        lon = input_data.get("longitude", 0.0)
        corr_id = input_data.get("correlation_id", "req_default")

        data = await self.service.get_weather(lat, lon)
        status = data.pop("status", "unknown")
        source = data.pop("source", "Open-Meteo Weather API")
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
                "temperature": "°C",
                "wind_speed": "km/h",
                "wind_direction": "°",
                "precipitation": "mm"
            },
            quality_status="VALID" if status == "success" else "UNAVAILABLE"
        )
