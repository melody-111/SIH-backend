from abc import ABC, abstractmethod
from app.schemas.agent import AgentResult
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        pass
