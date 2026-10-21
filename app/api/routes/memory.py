from fastapi import APIRouter, HTTPException, Query, Header, Body
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from app.services.conversational_memory import ConversationalMemoryService

router = APIRouter(prefix="/memory", tags=["memory"])

memory_service = ConversationalMemoryService()

def _resolve_user_id(header_user: Optional[str], param_user: Optional[str]) -> str:
    uid = header_user or param_user
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication required: Provide user_id or X-User-ID header.")
    return uid.strip()

class ResolveFollowUpRequest(BaseModel):
    conversation_id: str
    query_text: str
    current_latitude: Optional[float] = None
    current_longitude: Optional[float] = None

class CreateMemoryRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=1000)
    category: str = "PREFERENCE"

class UpdateMemoryRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

class FeedbackRequest(BaseModel):
    message_id: str = Field(..., max_length=100)
    rating: int = Field(..., ge=1, le=5)
    feedback_notes: Optional[str] = Field(None, max_length=500)
    consent_given: bool = True
    correlation_id: Optional[str] = Field(None, max_length=100)
    category: str = Field("GENERAL", max_length=50)

@router.get("/context/{conversation_id}")
async def get_context(conversation_id: str):
    """Returns recent conversational context turns."""
    turns = memory_service.get_conversation_context(conversation_id)
    return {"conversation_id": conversation_id, "turns": turns, "count": len(turns)}

@router.delete("/context/{conversation_id}")
async def clear_context(conversation_id: str):
    """Clears short-term conversation context."""
    memory_service.clear_conversation(conversation_id)
    return {"status": "success", "message": f"Context for {conversation_id} cleared."}

@router.post("/resolve-followup")
async def resolve_followup(request: ResolveFollowUpRequest):
    """
    Resolves conversational follow-ups such as 'what about tomorrow?' or 'compare it with yesterday'.
    """
    return memory_service.resolve_followup_reference(
        conversation_id=request.conversation_id,
        query_text=request.query_text,
        current_lat=request.current_latitude,
        current_lon=request.current_longitude
    )

@router.get("/preferences")
async def get_user_preferences(
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Retrieves user preferences (language, units, saved locations, voice settings)."""
    uid = _resolve_user_id(x_user_id, user_id)
    return memory_service.get_preferences(uid)

@router.put("/preferences")
async def update_user_preferences(
    payload: Dict[str, Any] = Body(...),
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Updates user preferences."""
    uid = _resolve_user_id(x_user_id, user_id or payload.get("user_id"))
    try:
        return memory_service.update_preferences(uid, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/memories")
async def list_memories(
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    limit: int = Query(50, ge=1, le=100, description="Max memories to return (1-100)"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """Lists explicit user-approved saved memories with tenant isolation and pagination."""
    uid = _resolve_user_id(x_user_id, user_id)
    return memory_service.list_saved_memories(uid, limit=limit, offset=offset)

@router.post("/memories")
async def save_memory(
    request: CreateMemoryRequest,
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Saves user-approved explicit memory.
    Enforces privacy policy: strictly rejects credentials, keys, or passwords.
    """
    uid = _resolve_user_id(x_user_id, user_id)
    try:
        return memory_service.save_memory(
            user_id=uid,
            title=request.title,
            content=request.content,
            category=request.category
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/memories/{memory_id}")
async def update_memory(
    memory_id: str,
    request: UpdateMemoryRequest,
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Updates an existing user memory."""
    uid = _resolve_user_id(x_user_id, user_id)
    updated = memory_service.update_memory(
        user_id=uid,
        memory_id=memory_id,
        title=request.title,
        content=request.content
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return updated

@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Deletes an explicit user memory."""
    uid = _resolve_user_id(x_user_id, user_id)
    deleted = memory_service.delete_memory(uid, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"status": "success", "message": f"Memory {memory_id} removed."}

@router.post("/feedback")
async def record_feedback(
    request: FeedbackRequest,
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Records user feedback under governance rules and persists in SQLite.
    Does not automatically alter model weights or train on user chats.
    """
    uid = _resolve_user_id(x_user_id, user_id)
    try:
        return memory_service.record_feedback(
            user_id=uid,
            message_id=request.message_id,
            rating=request.rating,
            feedback_notes=request.feedback_notes,
            consent_given=request.consent_given,
            correlation_id=request.correlation_id,
            category=request.category
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/feedback")
async def list_feedback(
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Lists feedback entries submitted by authenticated user with data isolation."""
    uid = _resolve_user_id(x_user_id, user_id)
    feedbacks = memory_service.list_user_feedback(user_id=uid, limit=limit, offset=offset)
    return {"feedback": feedbacks, "count": len(feedbacks)}

@router.delete("/feedback/{feedback_id}")
async def delete_feedback(
    feedback_id: str,
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Deletes a feedback record submitted by user. Rejects deletion if not owned by user."""
    uid = _resolve_user_id(x_user_id, user_id)
    deleted = memory_service.delete_user_feedback(user_id=uid, feedback_id=feedback_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Feedback entry not found or unauthorized.")
    return {"status": "success", "message": f"Feedback {feedback_id} removed."}

@router.get("/feedback/export")
async def export_feedback_for_evaluation(
    limit: int = Query(100, ge=1, le=500)
):
    """
    Exports consent-granted, anonymized feedback items for controlled evaluation loop.
    Enforces that private conversations and user identities are never exposed.
    """
    exported = memory_service.export_anonymized_feedback(limit=limit)
    return {"status": "success", "exported_count": len(exported), "evaluation_samples": exported}
