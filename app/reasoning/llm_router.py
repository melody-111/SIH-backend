from app.schemas.chat import UserQuery
from typing import List

class LLMRouter:
    def determine_agents(self, query: UserQuery) -> List[str]:
        # Based on intent extraction, for now return all active for prototype
        return ["weather", "ocean", "pfz", "gis", "hazard", "sst", "chlorophyll", "satellite"]
