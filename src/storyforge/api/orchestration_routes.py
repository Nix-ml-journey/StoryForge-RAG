import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from storyforge.config.config import load_config
from storyforge.rag.extraction import extract_grounded_facts
from storyforge.rag.generative_ai import parse_gen_mode, parse_story_type
from storyforge.rag.retrieval import _docs_to_chunks, retrieve_docs
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


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------

class GenerateStreamRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The story generation query")
    mode: Optional[str] = Field(default="fast", description="Generation mode: 'fast' or 'thinking'")
    n_stories: int = Field(default=3, ge=1, le=10, description="Number of source stories to retrieve")
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Stream a fast story",
                    "value": {"query": "A warrior monk's journey through the desert", "mode": "fast", "n_stories": 3},
                }
            ]
        }
    )


async def _stream_story_sse(request: GenerateStreamRequest) -> AsyncIterator[str]:
    """Core SSE generator: Steps 1–2 sync in thread, Step 3 streams via Ollama."""

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    cfg = load_config()
    mode = parse_gen_mode(request.mode)

    # Step 1 — retrieve
    yield _sse({"step": "retrieve", "status": "start"})
    try:
        docs = await asyncio.to_thread(
            retrieve_docs, request.query, cfg, n_stories=request.n_stories
        )
        chunks = _docs_to_chunks(docs)
    except Exception as exc:
        yield _sse({"step": "retrieve", "status": "error", "detail": str(exc)})
        return
    yield _sse({"step": "retrieve", "status": "done", "n_chunks": len(chunks)})

    # Step 2 — extract grounded facts
    yield _sse({"step": "extract", "status": "start"})
    try:
        grounded_raw, parsed = await asyncio.to_thread(
            extract_grounded_facts, request.query, chunks, cfg
        )
    except Exception as exc:
        yield _sse({"step": "extract", "status": "error", "detail": str(exc)})
        return
    yield _sse({"step": "extract", "status": "done"})

    # Step 3 — stream story generation via Ollama
    yield _sse({"step": "generate", "status": "start"})
    try:
        from storyforge.rag.attribution import format_facts_for_prompt
        from storyforge.rag.extraction import _get_generation_prompts
        from storyforge.rag.generation import (
            _flow_section_headers,
            _is_thinking_mode,
            _mode_generation_params,
        )
        from storyforge.rag.generation_backend import (
            ollama_base_url,
            ollama_model_id,
            strip_thinking_tags,
        )
        from langchain_core.messages import HumanMessage
        from langchain_ollama import ChatOllama  # type: ignore

        prompts = _get_generation_prompts()
        formatted_facts = format_facts_for_prompt(parsed)
        facts_for_prompt = formatted_facts if formatted_facts else grounded_raw

        story_prompt = (
            f"{(prompts['story_system'] or '').strip()}\n\n"
            + prompts["story_user"]
            .format(
                query=request.query,
                section_headers=_flow_section_headers(cfg),
                grounded_facts=facts_for_prompt,
            )
            .strip()
        )

        max_new, temperature, top_p = _mode_generation_params(cfg, mode=mode)
        repeat_penalty = float(cfg.get("Generation_repetition_penalty") or 1.08)
        llm = ChatOllama(
            model=ollama_model_id(cfg),
            base_url=ollama_base_url(cfg),
            temperature=temperature,
            top_p=top_p,
            num_predict=max_new,
            options={"repeat_penalty": repeat_penalty},
            think=_is_thinking_mode(mode),
        )

        async for chunk in llm.astream([HumanMessage(content=story_prompt)]):
            token = strip_thinking_tags(str(getattr(chunk, "content", "") or ""))
            if token:
                yield _sse({"step": "generate", "token": token})

    except Exception as exc:
        logging.exception("generate_stream Step 3 failed")
        yield _sse({"step": "generate", "status": "error", "detail": str(exc)})
        return

    yield _sse({"step": "generate", "status": "done"})
    yield "data: [DONE]\n\n"


@orchestration_router.post(
    "/generate_stream",
    summary="Stream story generation (SSE)",
    response_description="Server-Sent Events stream of generation progress and tokens",
)
async def generate_stream(request: GenerateStreamRequest):
    """Three-step RAG with streamed Step 3 output over Server-Sent Events.

    Events are JSON objects on ``data:`` lines:

    - ``{"step": "retrieve", "status": "start|done|error", "n_chunks": N}``
    - ``{"step": "extract",  "status": "start|done|error"}``
    - ``{"step": "generate", "status": "start|done|error"}``
    - ``{"step": "generate", "token": "<text fragment>"}``  — repeated per token
    - ``[DONE]``  — terminal event
    """
    return StreamingResponse(
        _stream_story_sse(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
