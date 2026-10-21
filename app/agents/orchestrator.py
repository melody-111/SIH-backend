import asyncio
import datetime
import uuid
import json
import time
import logging
import re
from typing import Dict, Any, List, Optional, AsyncGenerator

from app.core.config import settings
from app.services.observability_service import (
    ObservabilityService,
    RequestTrace,
    SafeErrorCategory,
    sanitize_location
)

from app.schemas.chat import UserQuery
from app.schemas.response import FinalResponse, ProgressStep
from app.schemas.marine import MarineContext
from app.schemas.agent import AgentResult

from app.agents.weather_agent import WeatherAgent
from app.agents.ocean_agent import OceanAgent
from app.agents.pfz_agent import PFZAgent
from app.agents.gis_agent import GISAgent
from app.agents.hazard_agent import HazardAgent
from app.agents.sst_agent import SSTAgent
from app.agents.chlorophyll_agent import ChlorophyllAgent
from app.agents.satellite_agent import SatelliteAgent
from app.agents.route_agent import RouteAgent
from app.agents.data_validation_agent import DataValidationAgent
from app.agents.response_agent import ResponseAgent

from app.reasoning.intent_engine import IntentEngine
from app.reasoning.context_resolver import ContextResolver
from app.reasoning.time_resolver import TimeResolver
from app.reasoning.query_planner import QueryPlanner
from app.reasoning.risk_engine import RiskEngine
from app.reasoning.recommendation_engine import RecommendationEngine
from app.reasoning.evidence_engine import EvidenceEngine

from app.services.offline_cache import OfflineMarineCache

logger = logging.getLogger("neerai.orchestrator")

class Orchestrator:
    """
    Central Multi-Agent Orchestrator for NeerAI.
    Coordinates intent parsing, selective agent execution, validation,
    real progress event tracking, bounded concurrency, and offline fallback.
    """
    def __init__(self):
        # Specialist Agents
        self.weather = WeatherAgent()
        self.ocean = OceanAgent()
        self.pfz = PFZAgent()
        self.gis = GISAgent()
        self.hazard = HazardAgent()
        self.sst = SSTAgent()
        self.chlorophyll = ChlorophyllAgent()
        self.satellite = SatelliteAgent()
        self.route = RouteAgent()
        self.validator = DataValidationAgent()
        self.response_agent = ResponseAgent()

        self.agent_map = {
            "weather": self.weather,
            "ocean": self.ocean,
            "pfz": self.pfz,
            "gis": self.gis,
            "hazard": self.hazard,
            "sst": self.sst,
            "chlorophyll": self.chlorophyll,
            "satellite": self.satellite,
            "route": self.route
        }

        # Reasoning & Planning Engines
        self.intent_engine = IntentEngine()
        self.context_resolver = ContextResolver()
        self.time_resolver = TimeResolver()
        self.query_planner = QueryPlanner()
        self.risk_engine = RiskEngine()
        self.recommendation_engine = RecommendationEngine()
        self.evidence_engine = EvidenceEngine()
        self.cache = OfflineMarineCache()
        from app.services.conversational_memory import ConversationalMemoryService
        self.memory_service = ConversationalMemoryService()

        # Observability and bounded concurrency
        self.observability = ObservabilityService()
        self._agent_semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._agent_semaphore is None:
            self._agent_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_AGENT_TASKS)
        return self._agent_semaphore

    async def process_query(self, query: UserQuery) -> FinalResponse:
        """
        Standard single-request pipeline returning FinalResponse.
        """
        async for item in self._execute_pipeline(query):
            if item.get("type") == "final":
                return item["response"]

        # Fallback if stream unexpectedly completed without final
        return FinalResponse(
            answer="Pipeline could not complete response.",
            language=query.language,
            risk="UNKNOWN",
            timestamp=datetime.datetime.utcnow().isoformat() + "Z"
        )

    async def process_query_stream(self, query: UserQuery) -> AsyncGenerator[str, None]:
        """
        Streams Server-Sent Events (SSE) progress and final response.
        Ensures clean connection termination with [DONE] and disconnect cancellation.
        """
        try:
            async for item in self._execute_pipeline(query):
                if item.get("type") == "progress":
                    yield f"data: {json.dumps(item)}\n\n"
                elif item.get("type") == "final":
                    resp_json = item["response"].model_dump()
                    yield f"data: {json.dumps({'type': 'final', 'response': resp_json})}\n\n"
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            logger.info("Client disconnected from SSE stream, resources cleanly released.")
            raise
        except Exception as e:
            err_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err_data)}\n\n"
            yield "data: [DONE]\n\n"

    async def _execute_pipeline(self, query: UserQuery) -> AsyncGenerator[Dict[str, Any], None]:
        correlation_id = f"req_{uuid.uuid4().hex[:8]}"
        now_str = datetime.datetime.utcnow().isoformat() + "Z"
        progress_steps: List[ProgressStep] = []

        # Telemetry & Observability state
        self.observability.increment_active_requests()
        req_start = time.perf_counter()
        status = "SUCCESS"
        error_cat = SafeErrorCategory.NONE
        error_summary = None
        agents_completed: List[str] = []
        agents_failed: List[str] = []
        agents_timed_out: List[str] = []
        is_cached: bool = False
        cache_status: str = "miss"
        intent: str = "UNKNOWN"
        lat: Optional[float] = None
        lon: Optional[float] = None
        agent_names: List[str] = []
        unavailable_sources: List[str] = []

        def record_step(step_name: str, agent: Optional[str] = None, details: Optional[str] = None) -> ProgressStep:
            p = ProgressStep(
                step=step_name,
                status="COMPLETED",
                agent=agent,
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                details=details
            )
            progress_steps.append(p)
            return p

        try:
            # --- STEP 1: Understand Query & Intent ---
            step1 = record_step("Understanding query", agent="intent_engine", details="Understanding your question: Classifying request")
            yield {"type": "progress", **step1.model_dump()}

            intent = await self.intent_engine.determine_intent_async(query.message, correlation_id=correlation_id)
            mode = self.intent_engine.determine_mode(query.message, intent)
            conv_id = query.conversation_id or "default_session"

            # Language mirroring: detect if user writes or speaks in Hindi/Hinglish
            hindi_keywords = [
                "kaisa", "kaisi", "kaise", "kya", "hai", "hain", "aaj", "kal", "samundar", "samunder",
                "lehrein", "lehar", "hawa", "toofan", "machhli", "machli", "kahan", "jaana", "jana",
                "namaste", "namaskar", "batao", "bataiye", "bata", "door", "paas", "khatra", "surakshit",
                "shukriya", "dhanyawad", "achha", "lag raha", "hoon", "theek", "madad", "kab", "aana",
                "wapas", "time", "samay", "jagah", "bhi", "aur", "or", "yeh", "ye", "vo", "wo", "karo",
                "sab", "sare", "de", "dee", "baare", "bare", "k=baare", "pucho", "phir", "bas", "kyuki",
                "kisi", "kisis", "hogaa", "dekhyegaa", "bataega", "karte", "hum", "mujhe", "apna",
                "tum", "kaun", "bhai", "chal", "raha", "rha", "scene", "hal", "haal", "sunao", "batao",
                "me", "mein", "pe", "par", "se", "ko", "ke", "ki", "ka"
            ]
            msg_lower = (query.message or "").lower()
            tokens = set(re.findall(r"\b\w+\b", msg_lower))
            if any(k in tokens for k in hindi_keywords) or any(k in msg_lower for k in ["machhli", "samundar", "toofan", "lehrein", "k=baare"]) or any('\u0900' <= c <= '\u097F' for c in (query.message or "")):
                query.language = "hi"

            # --- CHAT MODE: Greetings, Small Talk, Feelings, Capabilities, Clarifications ---
            if mode == "CHAT" and intent != "GENERAL_EXPLANATORY":
                step_chat = record_step("Preparing your answer", agent="conversational_brain", details="Synthesizing natural conversational response")
                yield {"type": "progress", **step_chat.model_dump()}

                history = self.memory_service.get_conversation_context(conv_id)
                chat_resp = await self.response_agent.generate_conversational_response(
                    query=query,
                    intent=intent,
                    history=history
                )
                chat_resp.progress_steps = progress_steps
                chat_resp.execution_plan = []
                chat_resp.agents_used = []
                chat_resp.mode = "CHAT"

                # Persist turn in memory
                self.memory_service.add_turn(conv_id, "user", query.message, intent=intent)
                self.memory_service.add_turn(conv_id, "assistant", chat_resp.answer, intent=intent)

                yield {"type": "final", "response": chat_resp}
                return

            # --- STEP 2: Checking Location and Time (AGENT / HYBRID MODE) ---
            step2 = record_step("Checking location and time", agent="context_resolver", details="Resolving coordinates and temporal scope")
            yield {"type": "progress", **step2.model_dump()}

            if query.latitude is not None and query.longitude is not None:
                base_lat = query.latitude
                base_lon = query.longitude
            else:
                try:
                    from app.services.location_detection_service import LocationDetectionService
                    detected = await LocationDetectionService().detect_location()
                    base_lat = detected.get("latitude", 26.8467)
                    base_lon = detected.get("longitude", 80.9462)
                except Exception:
                    base_lat = 26.8467
                    base_lon = 80.9462


            # Check follow-up references from conversational memory
            followup = self.memory_service.resolve_followup_reference(conv_id, query.message, base_lat, base_lon)
            if followup.get("is_followup"):
                if followup.get("resolved_latitude") is not None:
                    base_lat = followup["resolved_latitude"]
                if followup.get("resolved_longitude") is not None:
                    base_lon = followup["resolved_longitude"]
                if intent == "FOLLOW_UP" and followup.get("inherited_intent"):
                    intent = followup["inherited_intent"]
                if followup.get("needs_clarification"):
                    clarif_resp = FinalResponse(
                        answer=followup.get("clarification_prompt", "Which coastal location or harbor should I check?"),
                        language=query.language,
                        risk="LOW",
                        timestamp=now_str,
                        progress_steps=progress_steps,
                        mode="CHAT"
                    )
                    yield {"type": "final", "response": clarif_resp}
                    return


            # Validate coordinate bounds
            if not (-90.0 <= base_lat <= 90.0) or not (-180.0 <= base_lon <= 180.0):
                status = "FAILED"
                error_cat = SafeErrorCategory.PARSE_ERROR
                error_summary = "Invalid geographical coordinates out of physical bounds."
                err_resp = FinalResponse(
                    answer="Invalid coordinates: latitude must be between -90 and 90, and longitude between -180 and 180.",
                    language=query.language,
                    risk="UNKNOWN",
                    warnings=["Invalid geographical coordinates provided."],
                    timestamp=now_str,
                    progress_steps=progress_steps
                )
                yield {"type": "final", "response": err_resp}
                return

            resolved_ctx = await self.context_resolver.resolve_context_async(conv_id, query.message, intent, base_lat, base_lon)
            resolved_time = self.time_resolver.resolve_time(query.message)
            lat = resolved_ctx["latitude"]
            lon = resolved_ctx["longitude"]
            query.latitude = lat
            query.longitude = lon

            # --- STEP 3: Offline Handling ---
            if self.cache.is_forced_offline:
                step_offline = record_step("Checking offline cache", agent="offline_cache", details="Device is operating offline")
                yield {"type": "progress", **step_offline.model_dump()}

                cached_hit = self.cache.find_nearest_cache(lat, lon)
                if cached_hit:
                    is_cached = True
                    cached_entry, dist_km = cached_hit
                    cache_status = "stale" if cached_entry.get("is_stale") else "fresh"
                    cached_snapshot = cached_entry["data"]
                    cached_context = MarineContext(**cached_snapshot)
                    risk_eval = self.risk_engine.calculate_risk(cached_context)
                    rec_eval = self.recommendation_engine.recommend(cached_snapshot, risk_eval)
                    
                    final_cached_resp = await self.response_agent.generate_response(
                        query=query,
                        intent=intent,
                        marine_context=cached_context,
                        risk=risk_eval,
                        recommendation=rec_eval,
                        agents_used=list(cached_snapshot.keys()),
                        unavailable_sources=[],
                        is_cached=True,
                        cached_at=cached_entry["cached_at"]
                    )
                    final_cached_resp.progress_steps = progress_steps
                    yield {"type": "final", "response": final_cached_resp}
                    return
                else:
                    # No cache available
                    status = "PARTIAL_FAILURE"
                    cache_status = "miss"
                    error_summary = "Offline cache miss for requested coordinates."
                    offline_unavailable = FinalResponse(
                        answer=(
                            "You are currently offline, and no cached marine observations are available for this sector. "
                            "Newly issued severe weather warnings or sea state changes cannot be verified without connectivity. "
                            "For safety, do not venture into open sea without consulting local harbor signals."
                        ),
                        language=query.language,
                        risk="UNKNOWN",
                        warnings=["No cached marine observations found for this coordinate while offline."],
                        timestamp=now_str,
                        progress_steps=progress_steps,
                        is_cached=False
                    )
                    yield {"type": "final", "response": offline_unavailable}
                    return

            # --- STEP 4: Execution Planning ---
            agent_names = self.query_planner.plan_agents(intent)
            agents_to_run = [self.agent_map[name] for name in agent_names if name in self.agent_map]

            step_plan = record_step(
                "Creating execution plan",
                agent="query_planner",
                details=f"Selected {len(agents_to_run)} specialist agents: {', '.join(agent_names) if agent_names else 'None (Educational query)'}"
            )
            yield {"type": "progress", **step_plan.model_dump()}

            # Language mirroring: detect if user writes or speaks in Hindi/Hinglish
            hindi_keywords = [
                "kaisa", "kaisi", "kaise", "kya", "hai", "hain", "aaj", "kal", "samundar", "samunder",
                "lehrein", "lehar", "hawa", "toofan", "machhli", "machli", "kahan", "jaana", "namaste",
                "namaskar", "batao", "bataiye", "door", "paas", "khatra", "surakshit", "shukriya", "dhanyawad"
            ]
            msg_lower = (query.message or "").lower()
            if any(k in msg_lower for k in hindi_keywords) or any('\u0900' <= c <= '\u097F' for c in (query.message or "")):
                query.language = "hi"

            # If coordinates are inland, route ocean/weather sensor queries to closest coast
            target_lat = resolved_ctx.get("nearest_coast_lat", lat) if resolved_ctx.get("is_inland") else lat
            target_lon = resolved_ctx.get("nearest_coast_lon", lon) if resolved_ctx.get("is_inland") else lon

            input_data = {
                "latitude": target_lat,
                "longitude": target_lon,
                "target_latitude": resolved_ctx.get("target_latitude"),
                "target_longitude": resolved_ctx.get("target_longitude"),
                "message": query.message,
                "target_time": resolved_time["utc_timestamp"],
                "correlation_id": correlation_id
            }

            # --- STEP 5: Parallel Specialist Execution with Bounded Concurrency & Timeouts ---
            results: List[AgentResult] = []

            if agents_to_run:
                if any(a.name in ["ocean", "sst", "weather"] for a in agents_to_run):
                    step_ocean = record_step("Fetching ocean data", agent="ocean", details="Retrieving marine data: sea temperature and wave swell")
                    yield {"type": "progress", **step_ocean.model_dump()}

                if any(a.name == "hazard" for a in agents_to_run):
                    step_hazard = record_step("Checking official warnings", agent="hazard", details="Checking IMD bulletins and storm alerts")
                    yield {"type": "progress", **step_hazard.model_dump()}

                async def safe_execute(agent):
                    exec_id = f"exec_{uuid.uuid4().hex[:8]}"
                    agent_start = time.perf_counter()
                    agent_input = dict(input_data)
                    # Weather and Hazard agents evaluate conditions directly at the resolved target coordinates
                    if agent.name in ["weather", "hazard"]:
                        agent_input["latitude"] = lat
                        agent_input["longitude"] = lon
                        agent_input["radius_km"] = 1000.0
                        agent_input["location_name"] = resolved_ctx.get("port_name") or resolved_ctx.get("display_name")

                    async with self._get_semaphore():
                        try:
                            res = await asyncio.wait_for(
                                agent.execute(agent_input),
                                timeout=settings.AGENT_TIMEOUT_SECONDS
                            )
                            agent_ms = (time.perf_counter() - agent_start) * 1000.0
                            agents_completed.append(agent.name)
                            self.observability.record_agent_execution(
                                execution_id=exec_id,
                                correlation_id=correlation_id,
                                agent_name=agent.name,
                                status="SUCCESS",
                                execution_time_ms=agent_ms
                            )
                            return res
                        except asyncio.TimeoutError:
                            agent_ms = (time.perf_counter() - agent_start) * 1000.0
                            agents_timed_out.append(agent.name)
                            self.observability.record_agent_execution(
                                execution_id=exec_id,
                                correlation_id=correlation_id,
                                agent_name=agent.name,
                                status="TIMEOUT",
                                execution_time_ms=agent_ms
                            )
                            return AgentResult(
                                agent=agent.name,
                                status="TIMEOUT",
                                data={},
                                confidence=0.0,
                                sources=[agent.name],
                                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                                correlation_id=correlation_id,
                                errors=[f"Agent execution timed out after {settings.AGENT_TIMEOUT_SECONDS}s"]
                            )
                        except asyncio.CancelledError:
                            agent_ms = (time.perf_counter() - agent_start) * 1000.0
                            self.observability.record_agent_execution(
                                execution_id=exec_id,
                                correlation_id=correlation_id,
                                agent_name=agent.name,
                                status="CANCELLED",
                                execution_time_ms=agent_ms
                            )
                            raise
                        except Exception as ex:
                            agent_ms = (time.perf_counter() - agent_start) * 1000.0
                            agents_failed.append(agent.name)
                            self.observability.record_agent_execution(
                                execution_id=exec_id,
                                correlation_id=correlation_id,
                                agent_name=agent.name,
                                status="FAILED",
                                execution_time_ms=agent_ms
                            )
                            return AgentResult(
                                agent=agent.name,
                                status="FAILED",
                                data={},
                                confidence=0.0,
                                sources=[agent.name],
                                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                                correlation_id=correlation_id,
                                errors=[str(ex)]
                            )

                tasks = [safe_execute(agent) for agent in agents_to_run]
                results = await asyncio.gather(*tasks)

            # Build context
            context_data = {"time": now_str, "location": resolved_ctx}
            agents_used = []
            collected_map: Dict[str, AgentResult] = {}

            for res in results:
                agents_used.append(res.agent)
                collected_map[res.agent] = res
                context_data[res.agent] = res.model_dump()
                if res.agent == "hazard":
                    context_data["hazards"] = res.model_dump()
                if res.status in ["FAILED", "UNAVAILABLE", "TIMEOUT"]:
                    unavailable_sources.append(res.agent)

            if agents_failed or agents_timed_out:
                status = "PARTIAL_FAILURE"
                if agents_timed_out:
                    error_cat = SafeErrorCategory.AGENT_TIMEOUT
                error_summary = f"Partial failure: {len(agents_failed)} failed, {len(agents_timed_out)} timed out."

            # --- STEP 6: Validating Evidence & Checking Freshness ---
            step_val = record_step("Validating evidence", agent="data_validation", details="Checking data freshness: verifying physical bounds")
            yield {"type": "progress", **step_val.model_dump()}

            validation_res = await self.validator.execute({
                "agent_results": collected_map,
                "correlation_id": correlation_id
            })
            context_data["validation"] = validation_res.model_dump()

            marine_context = MarineContext(**context_data)

            # --- STEP 7: Risk & Recommendation Evaluation ---
            step_risk = record_step("Evaluating maritime safety", agent="risk_engine", details="Computing wave and wind risk thresholds")
            yield {"type": "progress", **step_risk.model_dump()}

            risk = self.risk_engine.calculate_risk(marine_context)
            recommendation = self.recommendation_engine.recommend(context_data, risk)

            # Context memory persistence
            if recommendation.get("recommended") and recommendation.get("area"):
                self.context_resolver.save_context(conv_id, recommendation["area"]["latitude"], recommendation["area"]["longitude"])

            # Extract Evidence Telemetry
            evidence_items = self.evidence_engine.extract_evidence(context_data)
            evidence_strings = [f"{e.claim} ({e.source} @ {e.timestamp})" for e in evidence_items]

            # --- STEP 8: Preparing Response ---
            step_resp = record_step("Preparing response", agent="response_agent", details="Preparing your answer: synthesizing evidence")
            yield {"type": "progress", **step_resp.model_dump()}

            final_response = await self.response_agent.generate_response(
                query=query,
                intent=intent,
                marine_context=marine_context,
                risk=risk,
                recommendation=recommendation,
                agents_used=agents_used,
                unavailable_sources=unavailable_sources
            )

            final_response.evidence.extend(evidence_strings)
            final_response.progress_steps = progress_steps
            final_response.execution_plan = agent_names
            final_response.configuration_version = self.query_planner.config_manager.get_active_version()
            final_response.mode = mode


            # Cache observation for offline availability
            if agents_used:
                self.cache.store_observation(lat, lon, context_data, location_name=f"{lat},{lon}")

            # Record turns in conversational memory
            loc_label = resolved_ctx.get("port_name") or resolved_ctx.get("display_name")
            self.memory_service.add_turn(conv_id, "user", query.message, intent=intent, latitude=lat, longitude=lon, location_name=loc_label)
            self.memory_service.add_turn(conv_id, "assistant", final_response.answer, intent=intent, latitude=lat, longitude=lon, location_name=loc_label)

            yield {"type": "final", "response": final_response}

        except asyncio.CancelledError:
            status = "CANCELLED"
            error_cat = SafeErrorCategory.CLIENT_DISCONNECT
            error_summary = "Client disconnected or request was cancelled"
            raise
        except Exception as ex:
            status = "FAILED"
            error_cat = SafeErrorCategory.INTERNAL_ERROR
            error_summary = str(ex)
            raise
        finally:
            self.observability.decrement_active_requests()
            total_duration_ms = (time.perf_counter() - req_start) * 1000.0
            trace = RequestTrace(
                correlation_id=correlation_id,
                intent=intent,
                query_length=len(query.message or ""),
                coarse_location=sanitize_location(lat, lon),
                agents_planned=agent_names,
                agents_completed=agents_completed,
                agents_failed=agents_failed,
                agents_timed_out=agents_timed_out,
                unavailable_sources=unavailable_sources,
                is_cached=is_cached,
                cache_status=cache_status,
                duration_ms=round(total_duration_ms, 2),
                status=status,
                error_category=error_cat,
                error_summary=error_summary
            )
            self.observability.record_trace(trace)

