import asyncio
import logging
from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from storyforge.rag.generative_ai import parse_gen_mode, parse_story_type
from storyforge.orchestrator.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

orchestration_router = APIRouter(prefix="/orchestration", tags=["Orchestration"])
orchestrator = Orchestrator()
VALID_PIPELINE_STEPS = [
    "0_fetch_and_extract",
    "1_prepare_and_enrich_story_json",
    "2_reset_vector_store",
    "3_ingest_stories",
    "4_generate_story_3step",
    "4_generate_story_agentic",
]


class RunPipelineRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Book title or search query to run the pipeline for")
    steps: Optional[list[str]] = Field(
        default=None,
        description=(
            "Pipeline steps to run. Omit (or null) = all default steps. "
            "Valid values: 0_fetch_and_extract, 1_prepare_and_enrich_story_json, "
            "2_reset_vector_store, 3_ingest_stories, 4_generate_story_3step, "
            "4_generate_story_agentic."
        ),
    )
    mode: Optional[str] = Field(default="fast", description="Generation mode: 'fast' or 'thinking'")
    story_type: Optional[str] = Field(
        default="mix",
        description="Story type filter: 'single' (standalone), 'series' (chapter-based), 'mix' (both).",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional free-form metadata payload (accepted but currently unused by the pipeline).",
        alias="Metadata",
        validation_alias="Metadata",
    )
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "summary": "Minimal — run ALL 5 steps (only title required)",
                    "value": {
                        "title": "Hansel and Gretel",
                    },
                },
                {
                    "summary": "Fetch + extract only",
                    "value": {
                        "title": "The Golden Bird",
                        "steps": ["0_fetch_and_extract"],
                        "mode": "fast",
                    },
                },
                {
                    "summary": "Ingest then generate",
                    "value": {
                        "title": "Rapunzel",
                        "steps": ["3_ingest_stories", "4_generate_story_3step"],
                        "mode": "fast",
                        "story_type": "single",
                    },
                },
                {
                    "summary": "Generate only — series stories",
                    "value": {
                        "title": "Peter Pan",
                        "steps": ["4_generate_story_3step"],
                        "mode": "fast",
                        "story_type": "series",
                    },
                },
            ]
        },
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
            "One pipeline step to run. Valid values: 0_fetch_and_extract, "
            "1_prepare_and_enrich_story_json, 2_reset_vector_store, 3_ingest_stories, "
            "4_generate_story_3step, 4_generate_story_agentic."
        ),
    )
    title: str = Field(default="manual_step", min_length=1, description="Title/query (required for steps 1 and 5)")
    mode: Optional[str] = Field(default="fast", description="Generation mode: 'fast' or 'thinking'")
    story_type: Optional[str] = Field(
        default="mix",
        description="Story type filter: 'single' (standalone), 'series' (chapter-based), 'mix' (both).",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional free-form metadata payload (accepted but currently unused by the pipeline).",
        alias="Metadata",
        validation_alias="Metadata",
    )
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "summary": "Fetch and extract a book",
                    "value": {
                        "step": "0_fetch_and_extract",
                        "title": "Hansel and Gretel",
                    },
                },
                {
                    "summary": "Prepare and enrich story JSON records",
                    "value": {
                        "step": "1_prepare_and_enrich_story_json",
                        "title": "manual_step",
                    },
                },
                {
                    "summary": "Reset vector store",
                    "value": {
                        "step": "2_reset_vector_store",
                        "title": "manual_step",
                    },
                },
                {
                    "summary": "Ingest stories into vector store",
                    "value": {
                        "step": "3_ingest_stories",
                        "title": "manual_step",
                    },
                },
                {
                    "summary": "Generate story (3-step RAG)",
                    "value": {
                        "step": "4_generate_story_3step",
                        "title": "The Golden Bird",
                        "mode": "fast",
                        "story_type": "mix",
                    },
                },
                {
                    "summary": "Generate story (agentic looped RAG)",
                    "value": {
                        "step": "4_generate_story_agentic",
                        "title": "The Golden Bird",
                        "mode": "fast",
                        "story_type": "mix",
                    },
                },
            ]
        },
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
