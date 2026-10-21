import pytest
from app.reasoning.intent_engine import IntentEngine
from app.reasoning.context_resolver import ContextResolver
from app.reasoning.time_resolver import TimeResolver
from app.reasoning.query_planner import QueryPlanner

def test_intent_engine():
    ie = IntentEngine()
    assert ie.determine_intent("cyclone risk") == "HAZARD"
    assert ie.determine_intent("fishing jagah batao") == "FISHING_LOCATION"
    assert ie.determine_intent("wahan waves kaisi hain") == "MARINE_CONDITION" or ie.determine_intent("wahan waves kaisi hain") == "FOLLOW_UP"

def test_context_resolver():
    cr = ContextResolver()
    cr.save_context("test_session_1", 12.34, 56.78)
    
    # Resolves to saved if wahan is used
    res = cr.resolve_context("test_session_1", "wahan ki info do", "FOLLOW_UP", 0.0, 0.0)
    assert res["latitude"] == 12.34
    assert res["longitude"] == 56.78
    
    # Resolves to GPS if not wahan
    res2 = cr.resolve_context("test_session_1", "yahan ka weather", "WEATHER", 10.0, 20.0)
    assert res2["latitude"] == 10.0
    assert res2["longitude"] == 20.0

def test_query_planner():
    qp = QueryPlanner()
    agents = qp.plan_agents("FISHING_LOCATION")
    assert "pfz" in agents
    assert "sst" in agents
    assert "weather" in agents
    
    agents2 = qp.plan_agents("HAZARD")
    assert "hazard" in agents2
