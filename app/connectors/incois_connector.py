import datetime
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.connectors.base_connector import ConnectorResult
from app.schemas.marine_normalized import (
    NormalizedMarineRecord,
    DataType,
    QualityStatus,
    ConnectorStatus
)

class INCOISConnector:
    """
    Connector for Indian National Centre for Ocean Information Services (INCOIS).
    References official portal: https://www.incois.gov.in/oceanservices/osfforecast.jsp
    Products:
      - Ocean State Forecast (OSF): Waves, swell, winds, ocean currents, Sea Surface Temperature (SST)
      - Potential Fishing Zone (PFZ): Satellite ocean color (chlorophyll) & thermal frontal boundaries
    Per government policy, official digital dissemination requires institutional authorization (INCOIS_API_KEY).
    """
    def __init__(self):
        self.name = "INCOIS (PFZ & Ocean State)"
        self.portal_url = "https://www.incois.gov.in/oceanservices/osfforecast.jsp"
        self.api_key = settings.INCOIS_API_KEY
        self.simulation_enabled = settings.ENABLE_SIMULATION_FALLBACK
        self.status = ConnectorStatus.REQUIRES_ACCESS if not self.api_key else ConnectorStatus.CONFIGURED_UNVERIFIED

    async def fetch_advisory(
        self,
        lat: float,
        lon: float
    ) -> ConnectorResult:
        if not self.api_key:
            if self.simulation_enabled:
                return ConnectorResult(
                    success=True,
                    source=self.name,
                    connector_status=ConnectorStatus.MOCKED.value,
                    data={
                        "is_simulated": True,
                        "pfz_status": "PROSPECTIVE_ZONE",
                        "bearing_degrees": 240,
                        "distance_km": 14.5,
                        "valid_until": (datetime.datetime.utcnow() + datetime.timedelta(days=1)).isoformat() + "Z",
                        "advisory_notes": "SIMULATED TEST ADVISORY: Potential fishing zone based on thermal front pattern."
                    }
                )
            else:
                return ConnectorResult(
                    success=False,
                    source=self.name,
                    connector_status=ConnectorStatus.REQUIRES_ACCESS.value,
                    error="INCOIS official data requires institutional API token (INCOIS_API_KEY). Register at incois.gov.in."
                )

        # Real production endpoint when token is present
        return ConnectorResult(
            success=False,
            source=self.name,
            connector_status=ConnectorStatus.CONFIGURED_UNVERIFIED.value,
            error="INCOIS authorized endpoint connection pending government clearance."
        )

    def normalize(
        self,
        raw_result: ConnectorResult,
        lat: float,
        lon: float,
        location_name: str = ""
    ) -> List[NormalizedMarineRecord]:
        retrieval_time = datetime.datetime.utcnow().isoformat() + "Z"
        records: List[NormalizedMarineRecord] = []

        if not raw_result.success:
            records.append(
                NormalizedMarineRecord(
                    latitude=lat,
                    longitude=lon,
                    location_name=location_name or None,
                    variable_name="pfz_advisory",
                    value=None,
                    unit="advisory",
                    data_type=DataType.ADVISORY,
                    timestamp=retrieval_time,
                    retrieval_timestamp=retrieval_time,
                    source_provider="INCOIS",
                    quality_status=QualityStatus.OFFLINE,
                    is_missing=True,
                    notes=raw_result.error or "INCOIS feed not authenticated."
                )
            )
            return records

        # If simulated feed was explicitly enabled
        data = raw_result.data
        records.append(
            NormalizedMarineRecord(
                latitude=lat,
                longitude=lon,
                location_name=location_name or None,
                variable_name="pfz_advisory",
                value=data.get("pfz_status"),
                unit="status",
                data_type=DataType.DERIVED_ESTIMATE if data.get("is_simulated") else DataType.ADVISORY,
                timestamp=retrieval_time,
                retrieval_timestamp=retrieval_time,
                source_provider="INCOIS (Simulated Environment)" if data.get("is_simulated") else "INCOIS",
                quality_status=QualityStatus.SUSPECT if data.get("is_simulated") else QualityStatus.VALID,
                is_missing=False,
                notes=data.get("advisory_notes")
            )
        )
        return records
