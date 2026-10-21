from typing import List
from app.core.agent_config import AgentConfigManager

class QueryPlanner:
    """
    Selects specialist agents based on intent according to the active versioned agent configuration.
    Never dispatches unnecessary external calls for general educational queries.
    """
    def __init__(self):
        self.config_manager = AgentConfigManager()

    def plan_agents(self, intent: str) -> List[str]:
        if intent in ["GENERAL_EXPLANATORY", "CONVERSATIONAL", "CLARIFICATION_NEEDED"]:
            return []

        if intent == "GIS_MAP":
            return ["gis", "ocean", "weather"]

        active_config = self.config_manager.get_active_config()
        routing = active_config.routing_policies.get(intent)
        if routing is not None:
            return list(routing)

        # Default safe fallback
        return ["weather", "ocean", "hazard"]

