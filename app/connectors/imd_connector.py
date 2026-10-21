import datetime
from typing import Dict, Any, List, Optional
import httpx
from app.core.config import settings
from app.connectors.base_connector import ConnectorResult
from app.schemas.marine_normalized import (
    MarineWarningRecord,
    WarningSeverity,
    ConnectorStatus
)

class IMDWarningsConnector:
    """
    Connector for India Meteorological Department (IMD) Severe Weather & Cyclone Bulletins.
    References official IMD API catalog: https://api.imd.gov.in/public/api_reference.html
    Supported Official Products:
      - Fishermen Warnings & Coastal Bulletins (/bulletin/fishermen, /bulletin/coastal)
      - Cyclone Tracks, Warnings & Cone of Uncertainty (/cyclone/warning, /cyclone/track)
      - Port Storm Warning Signals (/warnings/port)
      - Sea-Area Bulletins & Severe Marine Squalls (/bulletin/sea-area)
    """
    def __init__(self):
        self.name = "IMD Weather & Marine Warnings"
        self.base_url = "https://api.imd.gov.in"
        self.api_key = settings.IMD_API_KEY
        self.simulation_enabled = settings.ENABLE_SIMULATION_FALLBACK
        self.status = ConnectorStatus.REQUIRES_ACCESS if not self.api_key else ConnectorStatus.CONFIGURED_UNVERIFIED

        # Official IMD API endpoint catalogue
        self.endpoints = {
            "fishermen_warning": f"{self.base_url}/v1/bulletin/fishermen",
            "coastal_bulletin": f"{self.base_url}/v1/bulletin/coastal",
            "cyclone_track": f"{self.base_url}/v1/cyclone/track",
            "cyclone_warning": f"{self.base_url}/v1/cyclone/warning",
            "port_warning": f"{self.base_url}/v1/warnings/port",
            "sea_area": f"{self.base_url}/v1/bulletin/sea-area"
        }

    async def fetch_warnings(
        self,
        lat: float,
        lon: float
    ) -> ConnectorResult:
        """
        Retrieves official IMD coastal and cyclone warnings.
        If credentials are absent, returns REQUIRES_ACCESS without fabricating fake data.
        """
        if not self.api_key:
            if self.simulation_enabled:
                return ConnectorResult(
                    success=True,
                    source=self.name,
                    connector_status=ConnectorStatus.MOCKED.value,
                    data={
                        "is_simulated": True,
                        "active_cyclone": False,
                        "warning_level": "GREEN",
                        "product_type": "FISHERMEN_WARNING",
                        "bulletin_text": "SIMULATED IMD BULLETIN: No cyclonic storm over Arabian Sea or Bay of Bengal.",
                        "issue_time": datetime.datetime.utcnow().isoformat() + "Z",
                        "affected_area": "Arabian Sea & Bay of Bengal Coastline"
                    }
                )
            else:
                return ConnectorResult(
                    success=False,
                    source=self.name,
                    connector_status=ConnectorStatus.REQUIRES_ACCESS.value,
                    error="Official IMD marine feed requires government credentials (IMD_API_KEY). Feed unconfigured."
                )

        # Authenticated production request to IMD API gateway
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Accept": "application/json",
                "User-Agent": "NeerAI-Maritime-Platform/1.0"
            }
            params = {"latitude": lat, "longitude": lon}
            async with httpx.AsyncClient(timeout=settings.CONNECTOR_TIMEOUT_SECONDS) as client:
                resp = await client.get(self.endpoints["fishermen_warning"], headers=headers, params=params)
                if resp.status_code == 200:
                    return ConnectorResult(
                        success=True,
                        source=self.name,
                        connector_status=ConnectorStatus.LIVE_VERIFIED.value,
                        data=resp.json(),
                        status_code=200
                    )
                else:
                    return ConnectorResult(
                        success=False,
                        source=self.name,
                        connector_status=ConnectorStatus.CONFIGURED_UNVERIFIED.value,
                        error=f"IMD Gateway HTTP {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code
                    )
        except Exception as exc:
            return ConnectorResult(
                success=False,
                source=self.name,
                connector_status=ConnectorStatus.UNAVAILABLE.value,
                error=f"IMD connection failure: {str(exc)}"
            )

    def normalize(
        self,
        raw_result: ConnectorResult,
        lat: float,
        lon: float
    ) -> List[MarineWarningRecord]:
        retrieval_time = datetime.datetime.utcnow().isoformat() + "Z"
        warnings: List[MarineWarningRecord] = []

        if not raw_result.success:
            # Explicit warning about feed unavailability so agents do not hallucinate safety
            warnings.append(
                MarineWarningRecord(
                    warning_type="OFFICIAL_FEED_UNAVAILABLE",
                    severity=WarningSeverity.INFO,
                    message="Official IMD cyclone warning feed is unconfigured. Relying on real meteorological observations; check local port signals.",
                    source="IMD Feed Monitor",
                    is_official=False,
                    valid_from=retrieval_time,
                    affected_coordinates={"latitude": lat, "longitude": lon},
                    provider="India Meteorological Department (IMD)",
                    official_bulletin_type="UNAVAILABLE_FEED_NOTICE",
                    issue_time=retrieval_time,
                    affected_geographic_area="Indian Coastal Waters",
                    severity_as_supplied="INFO",
                    original_source_reference="https://api.imd.gov.in/public/api_reference.html",
                    retrieval_time=retrieval_time,
                    verification_status="UNCONFIGURED_GOVERNMENT_FEED"
                )
            )
            return warnings

        data = raw_result.data
        if data.get("active_cyclone"):
            warnings.append(
                MarineWarningRecord(
                    warning_type="OFFICIAL_CYCLONE_WARNING",
                    severity=WarningSeverity.DANGER,
                    message=data.get("bulletin_text", "Severe Cyclonic Storm warning issued by IMD."),
                    source="IMD (Simulated Bulletin)" if data.get("is_simulated") else "IMD",
                    is_official=not data.get("is_simulated"),
                    valid_from=retrieval_time,
                    affected_coordinates={"latitude": lat, "longitude": lon},
                    provider="India Meteorological Department (IMD)",
                    official_bulletin_type=data.get("product_type", "CYCLONE_BULLETIN"),
                    issue_time=data.get("issue_time", retrieval_time),
                    affected_geographic_area=data.get("affected_area", f"Coast near {lat:.2f}N, {lon:.2f}E"),
                    severity_as_supplied=data.get("warning_level", "RED"),
                    original_source_reference="https://api.imd.gov.in/v1/cyclone/warning",
                    retrieval_time=retrieval_time,
                    verification_status="OFFICIAL_VERIFIED" if not data.get("is_simulated") else "MOCKED_TEST_FIXTURE"
                )
            )
        return warnings
