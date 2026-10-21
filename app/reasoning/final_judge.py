import datetime
import os
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI

from app.schemas.chat import UserQuery
from app.schemas.marine import MarineContext
from app.schemas.response import FinalResponse
from app.services.llm_service import MultiModelLLMService

class FinalJudge:
    """
    Reasoning synthesis component for NeerAI.
    Uses MultiModelLLMService with dynamic model routing,
    structured output validation, and deterministic safety fallback.
    """
    def __init__(self):
        self.llm_service = MultiModelLLMService()
        # Keep client attribute for backward compatibility
        self.client = AsyncOpenAI(api_key=os.getenv("LLM_API_KEY", "dummy"))

    async def judge(
        self,
        query: UserQuery,
        context: MarineContext,
        risk: Dict[str, Any],
        recommendation: Dict[str, Any],
        agents_used: List[str]
    ) -> FinalResponse:
        context_dict = context.model_dump(exclude_none=True)
        now_str = datetime.datetime.utcnow().isoformat() + "Z"

        # Attempt structured reasoning synthesis via multi-model service
        advisory = await self.llm_service.synthesize_advisory(
            query_message=query.message,
            language=query.language or "en",
            marine_context_dict=context_dict,
            risk_level=risk.get("level", "LOW"),
            recommendation_dict=recommendation,
            correlation_id=getattr(query, "correlation_id", "final_judge")
        )

        if advisory:
            rec_str = advisory.recommended_coordinates
            if not rec_str and recommendation.get("recommended") and recommendation.get("area"):
                rec_str = f"{recommendation['area']['latitude']},{recommendation['area']['longitude']}"

            merged_warnings = list(risk.get("warnings", []))
            for caveat in advisory.caveats:
                if caveat not in merged_warnings:
                    merged_warnings.append(caveat)

            return FinalResponse(
                answer=advisory.answer,
                language=advisory.language,
                risk=advisory.risk_assessment or risk.get("level", "LOW"),
                recommended_location=rec_str,
                reasons=recommendation.get("reasons", risk.get("reasons", [])),
                warnings=merged_warnings,
                evidence=[],
                map_actions=[{"type": advisory.map_action_type or "center", "lat": query.latitude or 15.3493, "lon": query.longitude or 74.1740}],
                agents_used=agents_used,
                timestamp=now_str
            )

        # Deterministic safety fallback
        return self._mock_response(query.language, risk, recommendation, agents_used)

    def _mock_response(self, lang: str, risk: Dict[str, Any], recommendation: Dict[str, Any], agents_used: List[str], err: str = "") -> FinalResponse:
        ans = "Conditions are favorable based on available data."
        if lang in ["hi", "hi-IN"]:
            ans = "Aapke aas-paas conditions favorable lag rahi hain. Maine map par area highlight kar diya hai."

        rec_str = None
        if recommendation.get("recommended") and recommendation.get("area"):
            rec_str = f"{recommendation['area']['latitude']},{recommendation['area']['longitude']}"

        return FinalResponse(
            answer=ans,
            language=lang,
            risk=risk.get("level", "LOW"),
            recommended_location=rec_str,
            reasons=recommendation.get("reasons", risk.get("reasons", [])),
            warnings=risk.get("warnings", []),
            evidence=[],
            map_actions=[{"type": "center", "lat": 15.3493, "lon": 74.1740}],
            agents_used=agents_used,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z"
        )
