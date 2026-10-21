import datetime
import math
from typing import Dict, Any, List, Optional
import httpx
from app.connectors.base_connector import BaseConnector, ConnectorResult
from app.schemas.marine_normalized import ConnectorStatus

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two geographic coordinates in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class GDACSConnector(BaseConnector):
    """
    Live real-time connector for the Global Disaster Alert and Coordination System (GDACS).
    Joint initiative of the United Nations Office for the Coordination of Humanitarian Affairs (OCHA)
    and the European Commission.
    Provides live real-time tracking for:
      - Tropical Cyclones (TC) / Depressions / Storms
      - Tsunamis (TS)
      - Marine and Coastal Flooding (FL)
    Public real-time GeoJSON API requiring no proprietary API key.
    """
    def __init__(self):
        super().__init__(
            name="GDACS (United Nations / EC)",
            base_url="https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
        )
        self.status = ConnectorStatus.LIVE_VERIFIED

    async def fetch_active_disasters(
        self,
        lat: float,
        lon: float,
        radius_km: float = 1200.0
    ) -> ConnectorResult:
        """
        Fetches live active disasters and filters for events affecting user coordinates
        or the Indian Ocean / Arabian Sea / Bay of Bengal basins.
        """
        params = {"eventtypes": "TC,TS,FL"}
        headers = {
            "Accept": "application/json",
            "User-Agent": "NeerAI-Maritime-Safety-Platform/1.0"
        }
        res = await self.get_with_retry(self.base_url, params=params, headers=headers)
        if not res.success or not res.data:
            return ConnectorResult(
                success=False,
                source=self.name,
                connector_status=ConnectorStatus.OFFLINE.value,
                error=res.error or "GDACS service unavailable"
            )

        features = res.data.get("features", [])
        active_cyclones: List[Dict[str, Any]] = []
        regional_alerts: List[Dict[str, Any]] = []

        # Indian Maritime Basin Bounds (Arabian Sea, Bay of Bengal, North Indian Ocean)
        # Lat: 0 to 28 N, Lon: 55 to 98 E
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [0.0, 0.0])
            event_lon, event_lat = coords[0], coords[1]

            event_type = props.get("eventtype", "")
            name = props.get("name") or props.get("eventname") or "Unnamed Weather Event"
            alert_level = props.get("alertlevel", "Green")
            description = props.get("htmldescription") or props.get("description") or ""

            dist_km = round(haversine_km(lat, lon, event_lat, event_lon), 1)

            # Check if event is in proximate danger radius OR inside Indian Ocean basin
            is_in_indian_basin = (0.0 <= event_lat <= 28.0 and 55.0 <= event_lon <= 98.0)
            is_within_range = dist_km <= radius_km

            if is_within_range or is_in_indian_basin:
                item = {
                    "event_type": event_type,
                    "type_name": "TROPICAL_CYCLONE" if event_type == "TC" else ("TSUNAMI" if event_type == "TS" else "COASTAL_FLOOD"),
                    "name": name,
                    "alert_level": alert_level,
                    "severity": alert_level.upper(),
                    "latitude": event_lat,
                    "longitude": event_lon,
                    "distance_km": dist_km,
                    "description": description,
                    "from_date": props.get("fromdate"),
                    "to_date": props.get("todate"),
                    "is_in_indian_basin": is_in_indian_basin,
                    "source": "GDACS (UN / European Commission)"
                }
                if event_type == "TC":
                    active_cyclones.append(item)
                else:
                    regional_alerts.append(item)

        return ConnectorResult(
            success=True,
            source=self.name,
            connector_status=ConnectorStatus.LIVE_VERIFIED.value,
            data={
                "total_events_scanned": len(features),
                "active_cyclones": active_cyclones,
                "has_cyclone_alert": len(active_cyclones) > 0,
                "regional_disaster_alerts": regional_alerts,
                "scanned_at": datetime.datetime.utcnow().isoformat() + "Z"
            }
        )
