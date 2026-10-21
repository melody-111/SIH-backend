from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from app.connectors.incois_connector import INCOISConnector
from typing import Dict, Any
import datetime

class PFZAgent(BaseAgent):
    """
    Handles Potential Fishing Zone (PFZ) intelligence.
    If no verified official PFZ product is accessible, it transparently
    explains this limitation and NEVER invents fake fishing coordinates.
    """
    def __init__(self):
        super().__init__("pfz", "Retrieves official Potential Fishing Zones from INCOIS")
        self.connector = INCOISConnector()

    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        lat = input_data.get("latitude", 0.0)
        lon = input_data.get("longitude", 0.0)
        corr_id = input_data.get("correlation_id", "req_default")
        now = datetime.datetime.utcnow().isoformat() + "Z"

        res = await self.connector.fetch_advisory(lat, lon)

        if not res.success:
            return AgentResult(
                agent=self.name,
                status="REQUIRES_AUTH",
                data={
                    "pfz_available": False,
                    "provider": "INCOIS",
                    "reason": res.error or "Official institutional credentials required"
                },
                confidence=0.0,
                sources=["INCOIS"],
                timestamp=now,
                correlation_id=corr_id,
                data_type="ADVISORY",
                quality_status="OFFLINE",
                missing_fields=["official_pfz_advisory"],
                errors=[res.error or "Authentication required"],
                notes="Official INCOIS PFZ service requires institutional API authorization. Fishing coordinates will not be fabricated without verified data."
            )

        data = res.data
        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            data=data,
            confidence=0.5 if data.get("is_simulated") else 0.9,
            sources=["INCOIS (Simulated)" if data.get("is_simulated") else "INCOIS"],
            timestamp=now,
            correlation_id=corr_id,
            data_type="DERIVED_ESTIMATE" if data.get("is_simulated") else "ADVISORY",
            quality_status="SUSPECT" if data.get("is_simulated") else "VALID",
            notes=data.get("advisory_notes")
        )
