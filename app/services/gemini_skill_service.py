import logging
import time
from typing import Dict, Any, List, Optional
from app.services.llm_service import MultiModelLLMService
from app.services.location_intelligence_service import LocationIntelligenceService

logger = logging.getLogger("neerai.gemini_skill")

GEMINI_SYSTEM_PERSONA = (
    "You are NeerAI, powered by Google Gemini intelligence — an expert, friendly, and deeply knowledgeable "
    "Earth, Ocean, and Climate Copilot. You converse with users just like a human friend and expert mentor would.\n\n"
    "Your Skills & Traits:\n"
    "1. Conversational Fluency: Speak completely naturally in English, Hindi (हिंदी), or Hinglish depending on what the user speaks. "
    "   Use warm, polite, and lively language (e.g. 'Haan bilkul', 'Aap bataiye', 'Main aapki madad karta hoon'). "
    "   Avoid sounding like an automated robotic IVR.\n"
    "2. Deep World Knowledge: You understand global geography, meteorology, ocean physics, cyclones, earthquakes, marine life, and everyday human conversations.\n"
    "3. Empathy: If a user shares their feelings, fatigue, or mood, be kind, supportive, and compassionate.\n"
    "4. Zero Fabrication of Live Sensor Facts: When quoting real-time weather, wave height, or storm alerts, strictly ground your response in "
    "   the verified live context provided. If no live warning exists, say so clearly.\n"
    "5. Actionable Guidance: Provide safety tips, practical advice, and clear explanations."
)

class GeminiSkillSetService:
    """
    Dedicated Gemini Conversational & Reasoning Skillset Module.
    Implements Google Gemini's trained conversational capabilities (natural chat,
    empathy, scientific reasoning, multilingual fluency in Hindi/Hinglish/English)
    and integrates live geospatial and hazard observations into dialogue.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiSkillSetService, cls).__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self):
        self.llm_service = MultiModelLLMService()
        self.loc_service = LocationIntelligenceService()

    async def get_status(self) -> Dict[str, Any]:
        """Returns Gemini model configuration, health, and skill status."""
        reg = await self.llm_service.get_provider_registry_status()
        gemini_info = reg.get("providers", {}).get("gemini", {})
        return {
            "module": "GeminiSkillSetService",
            "active_provider": reg.get("active_provider"),
            "gemini_status": gemini_info.get("status", "UNCONFIGURED"),
            "is_available": gemini_info.get("is_available", False),
            "model_fast": self.llm_service.fast_model,
            "model_reasoning": self.llm_service.reasoning_model,
            "skills": [
                "Natural Conversational Chat (ChatGPT / Gemini style)",
                "Multilingual Fluency (Hindi, Hinglish, English)",
                "Global Real-Time Disaster Radius Analysis (1000 km)",
                "Physical Oceanography & Marine Science Explanations",
                "Atmospheric Weather & Forecasting Interpretation",
                "Emotional Empathy & Conversational Memory"
            ]
        }

    async def generate_conversational_turn(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        live_location: Optional[Dict[str, Any]] = None,
        language: str = "en",
        correlation_id: str = "gemini_turn"
    ) -> Dict[str, Any]:
        """
        Executes a natural conversational turn using Gemini's trained weights.
        Grounds responses with user's detected location and real-time environment.
        """
        start = time.perf_counter()
        prov_name, prov = self.llm_service.resolve_provider()

        # Build prompt context if live coordinates or city are provided
        loc_context_str = ""
        if live_location:
            c_name = live_location.get("display_name") or live_location.get("city") or "User Location"
            lat = live_location.get("latitude")
            lon = live_location.get("longitude")
            loc_context_str = f"\n[User Live Location Context: {c_name} at ({lat}, {lon})]"

        system_prompt = GEMINI_SYSTEM_PERSONA
        if loc_context_str:
            system_prompt += f"\n{loc_context_str}"

        # Format message history
        formatted_messages = []
        if conversation_history:
            for turn in conversation_history[-8:]:
                role = "user" if turn.get("role", "").lower() in ["user", "human"] else "assistant"
                formatted_messages.append({"role": role, "content": turn.get("content") or turn.get("text", "")})
        formatted_messages.append({"role": "user", "content": message})

        # 1. Real Gemini execution
        if prov_name == "gemini" and prov.is_available:
            try:
                content, tokens, cost = await prov.generate_chat_completion(
                    messages=formatted_messages,
                    system_prompt=system_prompt,
                    model=self.llm_service.fast_model,
                    temperature=0.7,
                    timeout_seconds=self.llm_service.timeout
                )
                dur_ms = (time.perf_counter() - start) * 1000.0
                return {
                    "answer": content.strip(),
                    "source": "Google Gemini (Official AI Studio)",
                    "model": self.llm_service.fast_model,
                    "is_live_llm": True,
                    "duration_ms": round(dur_ms, 2),
                    "tokens": tokens
                }
            except Exception as e:
                logger.warning(f"Gemini execution error, using conversational engine: {e}")

        # 2. High-fidelity conversational fallback engine
        from app.agents.response_agent import ResponseAgent
        from app.schemas.chat import UserQuery
        resp_agent = ResponseAgent()
        query_obj = UserQuery(message=message, language=language)
        chat_resp = await resp_agent.generate_conversational_response(
            query=query_obj,
            intent="CONVERSATIONAL",
            history=conversation_history
        )
        dur_ms = (time.perf_counter() - start) * 1000.0
        return {
            "answer": chat_resp.answer,
            "source": "NeerAI Conversational Intelligence (Fallback Engine)",
            "model": "rule_based_conversational",
            "is_live_llm": False,
            "duration_ms": round(dur_ms, 2),
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
