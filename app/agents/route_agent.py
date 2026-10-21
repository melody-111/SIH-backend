from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from typing import Dict, Any, Optional
import datetime
import math

class RouteAgent(BaseAgent):
    """
    Performs route and transit calculations using available environmental data.
    Explicitly states navigation limitations.
    """
    def __init__(self):
        super().__init__("route", "Calculates maritime distances and transit estimates")

    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_lat = input_data.get("latitude", 0.0)
        start_lon = input_data.get("longitude", 0.0)
        target_lat = input_data.get("target_latitude", start_lat + 0.1)
        target_lon = input_data.get("target_longitude", start_lon + 0.1)
        corr_id = input_data.get("correlation_id", "req_default")
        now = datetime.datetime.utcnow().isoformat() + "Z"

        # Calculate distance (Haversine)
        R = 6371.0
        dlat = math.radians(target_lat - start_lat)
        dlon = math.radians(target_lon - start_lon)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(start_lat)) * math.cos(math.radians(target_lat)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_km = round(R * c, 2)
        distance_nm = round(distance_km * 0.539957, 2)  # Nautical miles

        # Calculate initial compass bearing
        y = math.sin(dlon) * math.cos(math.radians(target_lat))
        x = math.cos(math.radians(start_lat)) * math.sin(math.radians(target_lat)) - math.sin(math.radians(start_lat)) * math.cos(math.radians(target_lat)) * math.cos(dlon)
        bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

        # Estimate transit time assuming 10 knots (~18.5 km/h) for typical small fishing craft
        cruising_speed_kmh = 18.5
        est_hours = round(distance_km / cruising_speed_kmh, 2)
        est_minutes = int(est_hours * 60)

        data = {
            "origin": {"latitude": start_lat, "longitude": start_lon},
            "destination": {"latitude": target_lat, "longitude": target_lon},
            "distance_km": distance_km,
            "distance_nautical_miles": distance_nm,
            "heading_degrees": round(bearing, 1),
            "estimated_transit_time_minutes": est_minutes,
            "assumed_speed_knots": 10.0,
            "navigational_limitation": "Straight-line nautical estimate only. Does not replace navigational charts, bathymetry, or harbor navigation channels."
        }

        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            data=data,
            confidence=0.85,
            sources=["Internal Nautical Kinematics"],
            timestamp=now,
            correlation_id=corr_id,
            data_type="DERIVED_ESTIMATE",
            quality_status="VALID",
            notes=data["navigational_limitation"]
        )
