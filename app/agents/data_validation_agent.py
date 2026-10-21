from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from typing import Dict, Any, List
import datetime

class DataValidationAgent(BaseAgent):
    """
    Checks data freshness, units, timestamps, physical value bounds,
    and flags inconsistencies or missing fields.
    """
    def __init__(self):
        super().__init__("data_validation", "Validates data freshness, sanity bounds, and consistency")

    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        collected_results: Dict[str, AgentResult] = input_data.get("agent_results", {})
        corr_id = input_data.get("correlation_id", "req_default")
        now = datetime.datetime.utcnow().isoformat() + "Z"

        validation_flags: List[str] = []
        unavailable_agents: List[str] = []
        is_all_valid = True

        for agent_name, result in collected_results.items():
            if result.status in ["FAILED", "UNAVAILABLE", "REQUIRES_AUTH"]:
                unavailable_agents.append(agent_name)
                is_all_valid = False
                continue

            data = result.data
            # Check wave bounds
            if "wave_height" in data and data["wave_height"] is not None:
                wh = data["wave_height"]
                if wh < 0.0 or wh > 25.0:
                    validation_flags.append(f"Suspicious wave height value: {wh} m from {agent_name}")
                    result.quality_status = "SUSPECT"

            # Check wind bounds
            if "wind_speed" in data and data["wind_speed"] is not None:
                ws = data["wind_speed"]
                if ws < 0.0 or ws > 300.0:
                    validation_flags.append(f"Suspicious wind speed value: {ws} km/h from {agent_name}")
                    result.quality_status = "SUSPECT"

            # Check SST bounds
            if "sst" in data and data["sst"] is not None:
                sst = data["sst"]
                if sst < 0.0 or sst > 45.0:
                    validation_flags.append(f"Suspicious SST value: {sst} °C from {agent_name}")
                    result.quality_status = "SUSPECT"

        return AgentResult(
            agent=self.name,
            status="SUCCESS" if is_all_valid else "PARTIAL",
            data={
                "all_valid": is_all_valid,
                "validation_flags": validation_flags,
                "unavailable_specialists": unavailable_agents
            },
            confidence=1.0 if is_all_valid else 0.8,
            sources=["Internal Validation Pipeline"],
            timestamp=now,
            correlation_id=corr_id,
            quality_status="VALID" if not validation_flags else "SUSPECT"
        )

    def validate_results(self, collected_results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """
        Synchronous evaluation of collected specialist agent results.
        Checks bounds, units, timestamp freshness, and flags stale data.
        """
        validation_flags = []
        unavailable = []
        is_all_valid = True
        now = datetime.datetime.utcnow()

        for agent_name, result in collected_results.items():
            if result.status in ["FAILED", "UNAVAILABLE", "REQUIRES_AUTH"]:
                unavailable.append(agent_name)
                is_all_valid = False
                continue

            # Freshness evaluation (> 24 hours = stale warning)
            if result.timestamp:
                try:
                    clean_ts = result.timestamp.replace("Z", "+00:00")
                    ts = datetime.datetime.fromisoformat(clean_ts)
                    diff_hours = (now.replace(tzinfo=datetime.timezone.utc) - ts).total_seconds() / 3600.0
                    if diff_hours > 24.0:
                        validation_flags.append(f"Stale data from {agent_name}: observation is {round(diff_hours, 1)} hours old")
                except Exception:
                    pass

            data = result.data or {}
            if "wave_height" in data and data["wave_height"] is not None:
                wh = data["wave_height"]
                if wh < 0.0 or wh > 25.0:
                    validation_flags.append(f"Suspicious wave height: {wh} m from {agent_name}")
            if "wind_speed" in data and data["wind_speed"] is not None:
                ws = data["wind_speed"]
                if ws < 0.0 or ws > 300.0:
                    validation_flags.append(f"Suspicious wind speed: {ws} km/h from {agent_name}")
            if "sst" in data and data["sst"] is not None:
                sst = data["sst"]
                if sst < 0.0 or sst > 45.0:
                    validation_flags.append(f"Suspicious SST: {sst} °C from {agent_name}")

        return {
            "all_valid": is_all_valid and len(validation_flags) == 0,
            "validation_flags": validation_flags,
            "unavailable": unavailable
        }

