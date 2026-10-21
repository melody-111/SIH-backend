from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from app.services.gis_service import GISService
from app.services.marine_data_service import KNOWN_COASTAL_LOCATIONS
from typing import Dict, Any, Optional
import datetime
import math

class GISAgent(BaseAgent):
    """
    Handles coordinate validation, boundary checking, coastal port resolution,
    and map-ready action structures.
    """
    def __init__(self):
        super().__init__("gis", "Handles location validation, port matching, and geospatial calculations")
        self.service = GISService()

    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        lat = input_data.get("latitude", 0.0)
        lon = input_data.get("longitude", 0.0)
        corr_id = input_data.get("correlation_id", "req_default")
        target_lat = input_data.get("target_latitude")
        target_lon = input_data.get("target_longitude")
        now = datetime.datetime.utcnow().isoformat() + "Z"

        # 1. Coordinate range validation
        errors = []
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            errors.append(f"Invalid geographical coordinates: ({lat}, {lon}) are out of range.")
            return AgentResult(
                agent=self.name,
                status="FAILED",
                data={"valid_coordinates": False},
                confidence=0.0,
                sources=["Internal GIS Engine"],
                timestamp=now,
                correlation_id=corr_id,
                errors=errors,
                quality_status="UNAVAILABLE"
            )

        # 2. Match nearest known Indian coastal port
        nearest_port = None
        min_port_dist = float("inf")
        for hub_key, hub in KNOWN_COASTAL_LOCATIONS.items():
            dlat = math.radians(hub["lat"] - lat)
            dlon = math.radians(hub["lon"] - lon)
            a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(hub["lat"])) * math.sin(dlon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            dist = 6371.0 * c
            if dist < min_port_dist:
                min_port_dist = dist
                nearest_port = hub

        # 3. GIS Context and distance
        data = await self.service.get_gis_context(lat, lon, target_lat, target_lon)
        status = data.pop("status", "success")
        source = data.pop("source", "Internal GIS Engine")

        data["valid_coordinates"] = True
        if nearest_port:
            data["nearest_coastal_port"] = nearest_port["name"]
            data["distance_to_port_km"] = round(min_port_dist, 1)

        # Map ready actions
        data["map_actions"] = [
            {"type": "center", "lat": lat, "lon": lon},
            {"type": "add_marker", "lat": lat, "lon": lon, "title": "Your Location"}
        ]
        if target_lat and target_lon:
            data["map_actions"].append({
                "type": "add_marker",
                "lat": target_lat,
                "lon": target_lon,
                "title": "Target Destination"
            })

        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            data=data,
            confidence=1.0,
            sources=[source],
            timestamp=now,
            correlation_id=corr_id,
            data_type="DERIVED_ESTIMATE",
            quality_status="VALID"
        )
