from fastapi import APIRouter, Query, HTTPException, Body
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from app.services.learning_scheduler import LearningPipelineScheduler
from app.services.marine_knowledge_service import MarineKnowledgeService
from app.services.model_training_pipeline import ModelTrainingPipeline

router = APIRouter(prefix="/learning", tags=["learning"])
scheduler = LearningPipelineScheduler()
knowledge_service = MarineKnowledgeService()
training_pipeline = ModelTrainingPipeline()

class ApproveModelRequest(BaseModel):
    training_id: str
    approver: str = "lead_architect"

class RollbackModelRequest(BaseModel):
    training_id: str

class RunTrainingRequest(BaseModel):
    model_name: str = "neerai-marine-llama3-lora"
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_version: str = "v1.0"
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_rank: int = 16

@router.get("/dashboard", summary="Continuous learning & pipeline monitoring dashboard")
async def get_learning_dashboard():
    """
    Returns real persisted dashboard metrics including last ingestion,
    data freshness, benchmark evaluation scores, model training status, and pending approvals.
    """
    return scheduler.get_dashboard_status()

@router.post("/ingest", summary="Trigger marine data ingestion run")
async def trigger_ingestion():
    """
    Triggers an immediate incremental data ingestion across Indian coastal stations.
    """
    return await scheduler.trigger_ingestion_now()

@router.post("/evaluate", summary="Trigger benchmark evaluation run")
async def trigger_evaluation():
    """
    Runs full offline regression benchmark, assessing zero fabrication and routing accuracy.
    """
    return await scheduler.trigger_evaluation_now()

@router.get("/training/readiness", summary="Inspect model training compute and dataset readiness")
async def check_training_readiness():
    """
    Inspects accelerator devices (CUDA/MPS/CPU), dataset readiness, and governance status.
    """
    return training_pipeline.assess_training_readiness()

@router.post("/training/prepare-dataset", summary="Prepare versioned dataset splits")
async def prepare_training_dataset(version: str = Query("v1.0")):
    """
    Creates reproducible, PII-scrubbed train/val/test splits in evals/data/{version}/.
    """
    return training_pipeline.prepare_versioned_dataset(version=version)

@router.post("/training/run", summary="Run reproducible fine-tuning job")
async def run_training(request: RunTrainingRequest):
    """
    Executes fine-tuning job and registers candidate model with PENDING_APPROVAL status.
    """
    return training_pipeline.run_training_job(
        model_name=request.model_name,
        base_model=request.base_model,
        dataset_version=request.dataset_version,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        lora_rank=request.lora_rank
    )

@router.post("/training/approve", summary="Human Architect Gate: Approve and deploy model")
async def approve_model(request: ApproveModelRequest):
    """
    STRICT APPROVAL GATE: Promotes an approved model candidate to active production status.
    """
    result = training_pipeline.approve_and_deploy(request.training_id, request.approver)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))
    return result

@router.post("/training/rollback", summary="Rollback deployed model to previous version")
async def rollback_model(request: RollbackModelRequest):
    """
    Reverts active production deployment to previous rollback version.
    """
    result = training_pipeline.rollback_model(request.training_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))
    return result

@router.get("/knowledge/search", summary="Search shared marine knowledge documents")
async def search_knowledge(
    query: str = Query(..., description="Search terms"),
    category: Optional[str] = Query(None, description="Category filter (ECOLOGY, SAFETY_PROTOCOL, REGULATION)"),
    coastal_region: Optional[str] = Query(None, description="Region filter")
):
    """
    Retrieves verified marine knowledge documents with source attribution and citations.
    """
    return knowledge_service.search_unstructured_knowledge(
        query=query,
        category=category,
        coastal_region=coastal_region
    )

@router.get("/knowledge/timeseries", summary="Query numeric marine timeseries records")
async def query_timeseries(
    parameter: Optional[str] = Query(None, description="Parameter: wave_height, wind_speed, sst, surface_pressure"),
    latitude: Optional[float] = Query(None, ge=-90.0, le=90.0),
    longitude: Optional[float] = Query(None, ge=-180.0, le=180.0),
    radius_km: float = Query(75.0, ge=1.0, le=500.0)
):
    """
    Executes precise numeric SQL queries against ingested historical and forecast observations.
    """
    return knowledge_service.query_numeric_timeseries(
        parameter=parameter,
        lat=latitude,
        lon=longitude,
        radius_km=radius_km
    )

@router.get("/knowledge/summary", summary="Summary of shared knowledge documents and records")
async def get_knowledge_summary():
    return knowledge_service.get_knowledge_summary()
