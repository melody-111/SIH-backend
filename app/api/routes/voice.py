from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel
from app.services.voice_service import VoiceService
from app.services.language_service import LanguageService

router = APIRouter(prefix="/voice", tags=["voice"])

voice_service = VoiceService()
language_service = LanguageService()

class STTRequest(BaseModel):
    audio_base64: str
    language: str = "en"

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

class DetectLanguageRequest(BaseModel):
    text: str

@router.get("/languages")
async def get_supported_languages():
    """
    Returns verified provider capability list for Indian languages,
    including native voice locales and STT/TTS availability.
    """
    return {
        "status": "success",
        "languages": language_service.get_supported_languages()
    }

@router.post("/detect-language")
async def detect_language(request: DetectLanguageRequest):
    """
    Detects query language based on Unicode script ranges and coastal keywords.
    """
    return language_service.detect_language(request.text)

@router.post("/stt")
async def speech_to_text(request: STTRequest):
    """
    Transcribes audio speech stream to text.
    Reports unconfigured cloud STT cleanly without pretending offline transcription works.
    """
    res = await voice_service.transcribe_audio(
        audio_base64=request.audio_base64,
        language=request.language
    )
    if not res.get("success") and res.get("status") == "PROVIDER_UNCONFIGURED":
        return {
            "status": "UNCONFIGURED",
            "message": res.get("error"),
            "transcript": ""
        }
    return res

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Synthesizes speech response. Sanitizes text to strictly prevent internal logs,
    JSON diagnostics, or tool payloads from being read aloud to users.
    """
    return await voice_service.synthesize_speech(
        text=request.text,
        language=request.language
    )
