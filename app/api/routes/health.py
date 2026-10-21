import datetime
from fastapi import APIRouter, HTTPException, status, Query
from app.services.api_health_service import APIHealthService
from app.db.database import DatabaseManager
from app.core.config import settings

router = APIRouter()
health_service = APIHealthService()
db_manager = DatabaseManager()

@router.get("/health", summary="Liveness probe")
async def health_check():
    """Returns application liveness status without dependency overhead."""
    return {
        "status": "ok",
        "service": "neerai-backend",
        "version": "1.0.0",
        "environment": settings.APP_ENV,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@router.get("/health/ready", summary="Readiness probe verifying DB, app, and external dependencies")
async def readiness_check(check_live_external: bool = Query(False, description="Whether to probe live external marine networks")):
    """
    Readiness probe that distinguishes:
    1. Application availability
    2. Database availability and migration integrity
    3. External marine provider availability
    """
    # 1. Database readiness check
    db_health = db_manager.check_db_health()
    if db_health.get("status") != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "message": "Database is not ready to accept connections.",
                "application": {"status": "healthy"},
                "database": db_health,
                "external_providers": health_service.get_quick_provider_status()
            }
        )

    # 2. External provider evaluation
    provider_status = health_service.get_quick_provider_status()
    if check_live_external:
        live_details = await health_service.check_health()
        provider_status["live_probes"] = live_details

    return {
        "status": "ready",
        "service": "neerai-backend",
        "application": {
            "status": "healthy",
            "version": "1.0.0",
            "environment": settings.APP_ENV
        },
        "database": db_health,
        "external_providers": provider_status,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@router.get("/system/api-status", summary="External marine sources connectivity status")
async def api_status():
    """Detailed live network health probe across all upstream data providers."""
    return await health_service.check_health()

@router.get("/health/metrics", summary="Lightweight real-time observability and telemetry summary")
async def observability_metrics():
    """Returns telemetry aggregates including p50/p95 latency, error rates, and cache hits."""
    from app.services.observability_service import ObservabilityService
    return ObservabilityService().get_metrics_summary()

@router.get("/system/llm-status", summary="Multi-model AI providers operational and health status")
async def llm_provider_status():
    """Probes and returns real runtime status of Gemini, Ollama, OpenAI, and deterministic providers."""
    from app.services.llm_service import MultiModelLLMService
    return await MultiModelLLMService().get_provider_registry_status()

