# FastAPI dependencies
def get_orchestrator():
    from app.agents.orchestrator import Orchestrator
    return Orchestrator()
