from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from app.connectors.imd_connector import IMDWarningsConnector
from app.connectors.gdacs_connector import GDACSConnector
from typing import Dict, Any, List
import datetime

class HazardAgent(BaseAgent):
    """
    Retrieves official marine warnings and supported hazard indicators.
    Official warnings take precedence. Model-derived indicators are clearly tagged.
    Missing hazard feed is never interpreted as proof of safety.
    """
    def __init__(self):
        super().__init__("hazard", "Retrieves marine hazards and official cyclone warnings")
        self.imd_connector = IMDWarningsConnector()
        self.gdacs_connector = GDACSConnector()

    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        lat = input_data.get("latitude", 0.0)
        lon = input_data.get("longitude", 0.0)
        corr_id = input_data.get("correlation_id", "req_default")
        now = datetime.datetime.utcnow().isoformat() + "Z"

        # 1. Official warning check
        imd_res = await self.imd_connector.fetch_warnings(lat, lon)
        official_warnings: List[Dict[str, Any]] = []
        is_official_feed_active = imd_res.success


        if imd_res.success and imd_res.data:
            official_warnings.append({
                "type": "OFFICIAL_CYCLONE_BULLETIN",
                "is_official": True,
                "bulletin": imd_res.data.get("bulletin_text"),
                "active_cyclone": imd_res.data.get("active_cyclone", False)
            })

        # Also query registered official bulletins from verified warning provider
        from app.services.official_warning_provider import OfficialWarningProvider
        provider_warnings = OfficialWarningProvider().get_warnings_for_location(lat, lon, include_expired=False)
        for pw in provider_warnings:
            official_warnings.append({
                "type": pw.warning_type,
                "is_official": True,
                "bulletin": pw.bulletin_text,
                "source": pw.source,
                "severity": pw.source_severity,
                "warning_id": pw.warning_id
            })
            is_official_feed_active = True

        # Query live real-time disaster alerts from GDACS, USGS, NASA EONET (UN / EC / Global)
        disaster_intel = None
        try:
            from app.services.disaster_intelligence_service import DisasterIntelligenceService
            disaster_svc = DisasterIntelligenceService()
            radius = input_data.get("radius_km", 1000.0)
            disaster_intel = await disaster_svc.fetch_disasters_around_location(
                lat=lat,
                lon=lon,
                radius_km=radius,
                location_name=input_data.get("location_name")
            )
            if disaster_intel:
                is_official_feed_active = True
                # Add any active disasters within 1000km radius as official warnings
                for ev in disaster_intel.get("events_within_radius", []):
                    official_warnings.append({
                        "type": f"DISASTER_{ev.get('event_type')}",
                        "is_official": True,
                        "bulletin": f"Live Regional Alert: {ev.get('name')} ({ev.get('severity')} Alert). Distance: {ev.get('distance_km')} km towards {ev.get('direction')}.",
                        "active_cyclone": ev.get("is_cyclone", False),
                        "source": ev.get("source", "GDACS / USGS"),
                        "severity": ev.get("severity", "WARNING"),
                        "distance_km": ev.get("distance_km"),
                        "direction": ev.get("direction")
                    })
        except Exception as e:
            pass


        # 2. Model-derived hazard indicators from input data if already passed
        derived_alerts: List[Dict[str, Any]] = []
        weather_data = input_data.get("weather_data", {})
        ocean_data = input_data.get("ocean_data", {})

        wind_speed = weather_data.get("wind_speed")
        if wind_speed and wind_speed >= 40.0:
            derived_alerts.append({
                "type": "MODEL_GALE_WIND",
                "severity": "WARNING",
                "message": f"Strong gale wind observed: {wind_speed} km/h.",
                "is_official": False
            })

        wave_height = ocean_data.get("wave_height")
        if wave_height and wave_height >= 2.5:
            derived_alerts.append({
                "type": "MODEL_HIGH_WAVE",
                "severity": "WARNING",
                "message": f"High sea swell observed: {wave_height} m wave height.",
                "is_official": False
            })

        notes = []
        if not is_official_feed_active:
            notes.append("Official IMD cyclone feed is unauthenticated. Safety cannot be guaranteed solely on absence of official feed.")

        return AgentResult(
            agent=self.name,
            status="SUCCESS" if (official_warnings or derived_alerts or is_official_feed_active) else "PARTIAL",
            data={
                "has_hazards": bool(official_warnings or derived_alerts),
                "official_warnings": official_warnings,
                "derived_alerts": derived_alerts,
                "disaster_intelligence": disaster_intel,
                "is_official_feed_active": is_official_feed_active
            },
            confidence=0.95 if is_official_feed_active else 0.70,
            sources=["IMD (Official)" if is_official_feed_active else "IMD (Unconfigured Feed)"],
            timestamp=now,
            correlation_id=corr_id,
            data_type="ADVISORY",
            quality_status="VALID" if is_official_feed_active else "SUSPECT",
            missing_fields=[] if is_official_feed_active else ["imd_official_feed"],
            notes="; ".join(notes) if notes else None
        )
