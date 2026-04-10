import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from Generative_AI.generative_ai import Gen_mode, parse_gen_mode, parse_story_type
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

class RunPipelineRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Book title or search query to run the pipeline for")
    steps: Optional[list[str]] = Field(
        default=None,
        description=(
            "Pipeline steps to run. Omit (or null) = all 5 steps. "
            "Valid values: 1_fetch_and_extract, 2_create_metadata_template, "
            "3_merge_check_summary, 4_ingest, 5_generate_and_evaluate."
        )
    )
    mode: Optional[str] = Field(default="fast", description="Generation mode: 'fast' or 'thinking'")
    story_type: Optional[str] = Field(
        default="mix",
        description="Story type filter: 'single' (standalone), 'series' (chapter-based), 'mix' (both).",
    )
    extract_style: Optional[bool] = Field(
        default=False,
        description="If true, Layer 1 also extracts narrative style/tone from source material.",
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Minimal — run ALL 5 steps (only title required)",
                    "value": {
                        "title": "Hansel and Gretel",
                    },
                },
                {
                    "summary": "Generate and evaluate only (step 5)",
                    "value": {
                        "title": "The Golden Bird",
                        "steps": ["5_generate_and_evaluate"],
                        "mode": "fast",
                        "story_type": "mix",
                    },
                },
                {
                    "summary": "Ingest then generate (steps 4 + 5)",
                    "value": {
                        "title": "Rapunzel",
                        "steps": ["4_ingest", "5_generate_and_evaluate"],
                        "mode": "fast",
                        "story_type": "single",
                    },
                },
                {
                    "summary": "Series stories with style extraction",
                    "value": {
                        "title": "Peter Pan",
                        "steps": ["5_generate_and_evaluate"],
                        "mode": "fast",
                        "story_type": "series",
                        "extract_style": True,
                    },
                },
            ]
        }
    )

class RunPipelineResponse(BaseModel):
    success: bool
    message: str
    title: str
    steps_done: list[str] = Field(default_factory=list)
    error: Optional[str] = None

class RunStepRequest(BaseModel):
    step: str = Field(
        ...,
        description=(
            "One pipeline step to run. Valid values: 1_fetch_and_extract, "
            "2_create_metadata_template, 3_merge_check_summary, 4_ingest, 5_generate_and_evaluate."
        ),
    )
    title: str = Field(default="manual_step", min_length=1, description="Title/query (required for steps 1 and 5)")
    mode: Optional[str] = Field(default="fast", description="Generation mode: 'fast' or 'thinking'")
    story_type: Optional[str] = Field(
        default="mix",
        description="Story type filter: 'single' (standalone), 'series' (chapter-based), 'mix' (both).",
    )
    extract_style: Optional[bool] = Field(
        default=False,
        description="If true, Layer 1 also extracts narrative style/tone from source material.",
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Fetch and extract a book (step 1)",
                    "value": {
                        "step": "1_fetch_and_extract",
                        "title": "Hansel and Gretel",
                    },
                },
                {
                    "summary": "Create metadata templates (step 2) — title optional",
                    "value": {
                        "step": "2_create_metadata_template",
                        "title": "manual_step",
                    },
                },
                {
                    "summary": "Merge + check summaries (step 3) — title optional",
                    "value": {
                        "step": "3_merge_check_summary",
                        "title": "manual_step",
                    },
                },
                {
                    "summary": "Ingest into vector store (step 4) — title optional",
                    "value": {
                        "step": "4_ingest",
                        "title": "manual_step",
                    },
                },
                {
                    "summary": "Generate and evaluate story (step 5)",
                    "value": {
                        "step": "5_generate_and_evaluate",
                        "title": "The Golden Bird",
                        "mode": "fast",
                        "story_type": "mix",
                    },
                },
                {
                    "summary": "Step 5 — series only with style extraction",
                    "value": {
                        "step": "5_generate_and_evaluate",
                        "title": "Peter Pan",
                        "mode": "fast",
                        "story_type": "series",
                        "extract_style": True,
                    },
                },
            ]
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
            gen_mode=parse_gen_mode(request.mode),
            extract_style=bool(request.extract_style),
            story_type=parse_story_type(request.story_type),
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
            gen_mode=parse_gen_mode(request.mode),
            extract_style=bool(request.extract_style),
            story_type=parse_story_type(request.story_type),
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
