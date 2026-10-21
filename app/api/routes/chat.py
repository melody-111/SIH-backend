from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.schemas.chat import UserQuery
from app.schemas.response import FinalResponse
from app.api.dependencies import get_orchestrator

router = APIRouter()

@router.post(
    "/chat",
    response_model=FinalResponse,
    summary="Main conversational marine intelligence endpoint"
)
async def chat_endpoint(query: UserQuery, orchestrator=Depends(get_orchestrator)):
    if query.audio_base64:
        if not query.message:
            query.message = "Can you check the marine conditions around here?"

    if not query.message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either message or audio_base64 is required."
        )

    try:
        response = await orchestrator.process_query(query)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestration error: {str(e)}"
        )

@router.post(
    "/chat/stream",
    summary="Real-time Server-Sent Events (SSE) streaming progress and response"
)
async def chat_stream_endpoint(query: UserQuery, orchestrator=Depends(get_orchestrator)):
    if query.audio_base64 and not query.message:
        query.message = "Can you check the marine conditions around here?"

    if not query.message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either message or audio_base64 is required."
        )

    return StreamingResponse(
        orchestrator.process_query_stream(query),
        media_type="text/event-stream"
    )

@router.get(
    "/chat/skills",
    summary="Get Gemini conversational skillset capabilities and status"
)
async def get_skills_endpoint():
    from app.services.gemini_skill_service import GeminiSkillSetService
    return await GeminiSkillSetService().get_status()

@router.post(
    "/chat/skill-turn",
    summary="Direct conversational turn using Gemini skill set with live location grounding"
)
async def chat_skill_turn_endpoint(payload: dict):
    from app.services.gemini_skill_service import GeminiSkillSetService
    message = payload.get("message", "")
    history = payload.get("history", [])
    location = payload.get("location")
    lang = payload.get("language", "en")
    service = GeminiSkillSetService()
    return await service.generate_conversational_turn(
        message=message,
        conversation_history=history,
        live_location=location,
        language=lang
    )

