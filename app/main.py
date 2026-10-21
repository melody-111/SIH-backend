import uuid
import logging
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import health, chat, voice, marine, map, hazard, alerts, cache, memory, learning, location, auth
from app.core.config import settings
from app.core.logging import setup_secure_logging

setup_secure_logging(settings.LOG_LEVEL)
logger = logging.getLogger("neerai.api")

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract correlation ID from header or generate new one
        correlation_id = (
            request.headers.get("x-correlation-id")
            or request.headers.get("x-request-id")
            or str(uuid.uuid4())
        )
        request.state.correlation_id = correlation_id
        
        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        except Exception as exc:
            logger.error(
                f"[CorrelationID: {correlation_id}] Unhandled error on {request.method} {request.url.path}: {str(exc)}",
                exc_info=True
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred while processing the request.",
                    "correlation_id": correlation_id
                },
                headers={"X-Correlation-ID": correlation_id}
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles graceful startup and shutdown for NeerAI backend."""
    # Validate environment settings before accepting traffic
    settings.validate_environment()
    
    from app.db.database import DatabaseManager
    db = DatabaseManager()
    health_status = db.check_db_health()

    # Initialize MongoDB Atlas indexes
    try:
        from app.db.mongodb import MongoDBManager
        await MongoDBManager().init_indexes()
    except Exception as me:
        logger.warning(f"MongoDB Atlas startup warning: {me}")

    logger.info(
        f"NeerAI Backend initialized. Environment: {settings.APP_ENV}. "
        f"Database: {health_status.get('status')} at {db.db_path}"
    )
    yield
    logger.info("NeerAI Backend gracefully shutting down.")

app = FastAPI(
    title="NeerAI Marine Intelligence API",
    description="Cross-platform marine AI intelligence, GIS ocean layers, and location-aware hazard reasoning API.",
    version="1.0.0",
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    lifespan=lifespan
)

# Correlation & tracing middleware
app.add_middleware(CorrelationIdMiddleware)

# CORS Middleware
origins: List[str] = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if not origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True if origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Correlation-ID", "X-User-ID"]
)

# API Routers
app.include_router(health.router)
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(voice.router, prefix="/api")
app.include_router(marine.router, prefix="/api")
app.include_router(map.router, prefix="/api")
app.include_router(hazard.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(cache.router, prefix="/api")
app.include_router(memory.router, prefix="/api")
app.include_router(learning.router, prefix="/api")
app.include_router(location.router, prefix="/api")

