import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from Generative_AI.generative_ai import Gen_mode
from Orchestrator.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

orchestration_router = APIRouter(prefix="/orchestration", tags=["Orchestration"])
orchestrator = Orchestrator()
VALID_PIPELINE_STEPS = [
    "1_fetch_and_extract",
    "2_create_metadata_template",
    "3_merge_check_summary",
    "4_ingest",
    "5_generate_and_evaluate",
]

def _parse_gen_mode(mode: Optional[str]) -> Gen_mode:
    if not mode:
        return Gen_mode.FAST
    return Gen_mode.THINKING if str(mode).lower() == "thinking" else Gen_mode.FAST

class RunPipelineRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Book title or search query to run the pipeline for")
    steps: Optional[list[str]] = Field(
        default=None,
        description=(
            "Pipeline steps to run. None = all 5 steps. Valid: 1_fetch_and_extract, "
            "2_create_metadata_template, 3_merge_check_summary, 4_ingest, 5_generate_and_evaluate."
        )
    )
    mode: Optional[str] = Field(default="fast", description="Generation mode: 'fast' or 'thinking'")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "The Adventures of Sherlock Holmes",
                "steps": [
                    "1_fetch_and_extract",
                    "2_create_metadata_template",
                    "3_merge_check_summary",
                    "4_ingest",
                    "5_generate_and_evaluate",
                ],
                "mode": "fast",
            }
        }
    )

class RunPipelineResponse(BaseModel):
    success: bool
    message: str
    title: str
    steps_done: list[str] = Field(default_factory=list)
    error: Optional[str] = None

class RunStepRequest(BaseModel):
    step: str = Field(..., description="One pipeline step to run")
    title: str = Field(default="manual_step", min_length=1, description="Title/query used by steps that need it")
    mode: Optional[str] = Field(default="fast", description="Generation mode: 'fast' or 'thinking'")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step": "4_ingest",
                "title": "manual_step",
                "mode": "fast",
            }
        }
    )

class RunStepResponse(BaseModel):
    success: bool
    step: str
    message: str
    steps_done: list[str] = Field(default_factory=list)
    error: Optional[str] = None

class StatusResponse(BaseModel):
    success: bool
    message: str
    type: str

@orchestration_router.post("/run_pipeline", response_model=RunPipelineResponse)
async def run_pipeline(request: RunPipelineRequest):
    try:
        result = await asyncio.to_thread(
            orchestrator.run_pipeline,
            title=request.title,
            steps=request.steps,
            gen_mode=_parse_gen_mode(request.mode),
        )
        return RunPipelineResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            title=request.title,
            steps_done=result.get("steps_done", []),
            error=result.get("error"),
        )
    except Exception as e:
        logging.exception("run_pipeline failed")
        return RunPipelineResponse(
            success=False,
            message="Pipeline failed",
            title=request.title,
            steps_done=[],
            error=str(e),
        )

@orchestration_router.post("/run_step", response_model=RunStepResponse)
async def run_step(request: RunStepRequest):
    """Run only one pipeline step while keeping the endpoint surface small."""
    if request.step not in VALID_PIPELINE_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid step '{request.step}'. Valid steps: {', '.join(VALID_PIPELINE_STEPS)}",
        )
    try:
        result = await asyncio.to_thread(
            orchestrator.run_pipeline,
            title=request.title,
            steps=[request.step],
            gen_mode=_parse_gen_mode(request.mode),
        )
        return RunStepResponse(
            success=result.get("success", False),
            step=request.step,
            message=result.get("message", ""),
            steps_done=result.get("steps_done", []),
            error=result.get("error"),
        )
    except Exception as e:
        logging.exception("run_step failed")
        return RunStepResponse(
            success=False,
            step=request.step,
            message="Step failed",
            steps_done=[],
            error=str(e),
        )

@orchestration_router.get("/status", response_model=StatusResponse)
async def orchestration_status():
    return StatusResponse(success=True, message="API is running", type="orchestration")
