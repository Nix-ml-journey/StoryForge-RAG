import asyncio
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import yaml
from pathlib import Path

from Generative_AI.generative_ai import Gen_mode, parse_gen_mode, parse_story_type
from Orchestrator.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_config_file = Path(__file__).parent.parent / "setup.yaml"
_config = yaml.safe_load(_config_file.read_text(encoding="utf-8"))
_DEFAULT_N_RESULTS = _config.get("Story_generation_n_results", 1)

create_eval_router = APIRouter(prefix="/create-eval", tags=["Create Eval"])
orchestrator = Orchestrator()


class StoryGenerateRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Story topic or query")
    generation_type: str = Field(default="full_story", description="'summary' or 'full_story'")
    save: bool = Field(default=True)
    n_results: int = Field(default=_DEFAULT_N_RESULTS, ge=1, le=20, description="Number of context chunks from vector store (default from setup.yaml)")
    mode: Optional[str] = Field(default="fast", description="Generation mode: 'fast' or 'thinking'")
    story_type: Optional[str] = Field(
        default="mix",
        description="Story type filter: 'single' (standalone), 'series' (chapter-based), 'mix' (both). Defaults to 'mix'.",
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Minimal — only query is required",
                    "value": {
                        "query": "A brave knight and a cunning dragon",
                    },
                },
                {
                    "summary": "Full options (standalone stories, thinking mode)",
                    "value": {
                        "query": "A lost child finds a magical forest",
                        "generation_type": "full_story",
                        "save": True,
                        "n_results": _DEFAULT_N_RESULTS,
                        "mode": "thinking",
                        "story_type": "single",
                    },
                },
                {
                    "summary": "Series stories with style extraction",
                    "value": {
                        "query": "Peter Pan adventures in Neverland",
                        "generation_type": "full_story",
                        "save": True,
                        "n_results": _DEFAULT_N_RESULTS,
                        "mode": "fast",
                        "story_type": "series",
                    },
                },
            ]
        }
    )


class StoryGenerateResponse(BaseModel):
    success: bool
    query: str
    generation_type: str
    content: str
    timestamp: str
    saved: bool = False
    saved_path: Optional[str] = None
    mode: Optional[str] = None
    gen_params: Optional[dict] = None


class SummaryGenerateRequest(BaseModel):
    story_path: str = Field(..., description="Relative path to the generated story .txt file")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "story_path": "Generated_Stories/20260401_120000_generated_story.txt"
            }
        }
    )


class SummaryGenerateResponse(BaseModel):
    success: bool
    summary: str
    timestamp: str
    saved: bool = False
    saved_path: Optional[str] = None


class StoryEvaluateRequest(BaseModel):
    story: str = Field(..., min_length=10, description="Full story text to evaluate")
    context: Optional[str] = None
    save: bool = False
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "story": "Once upon a time in a faraway kingdom, a brave young girl set out on a journey through the enchanted forest...",
                "save": True,
            }
        }
    )


class StoryEvaluateFileRequest(BaseModel):
    story_path: str = Field(..., description="Relative path to the generated story .txt file")
    context_path: Optional[str] = None
    save: bool = False
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "story_path": "Generated_Stories/20260401_120000_generated_story.txt",
                "save": True,
            }
        }
    )


class StoryEvaluateResponse(BaseModel):
    success: bool
    average_score: float
    scores: dict
    summary: str
    suggestions: list
    metadata: dict
    conclusion: str


class SummaryEvaluateRequest(BaseModel):
    summary_path: str = Field(..., description="Relative path to the summary .txt file")
    context: Optional[str] = None
    save: bool = False
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary_path": "Summarized_Stories/20260401_summary.txt",
                "save": True,
            }
        }
    )


class SummaryEvaluateFileRequest(BaseModel):
    summary_path: str = Field(..., description="Relative path to the summary .txt file")
    context_path: Optional[str] = Field(default=None, description="Optional: relative path to the original story .txt for comparison")
    save: bool = False
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary_path": "Summarized_Stories/20260401_summary.txt",
                "context_path": "Generated_Stories/20260401_120000_generated_story.txt",
                "save": True,
            }
        }
    )


class SummaryEvaluateResponse(BaseModel):
    success: bool
    average_score: float
    scores: dict
    summary: str
    suggestions: list
    metadata: dict
    conclusion: str


class StatusResponse(BaseModel):
    success: bool
    message: str
    type: str


@create_eval_router.post("/story_generate", response_model=StoryGenerateResponse)
async def story_generate(request: StoryGenerateRequest):
    try:
        result = await asyncio.to_thread(
            orchestrator.generate_story,
            query=request.query,
            generation_type=request.generation_type,
            save=request.save,
            n_results=request.n_results,
            mode=parse_gen_mode(request.mode),
            story_type=parse_story_type(request.story_type),
        )
        return StoryGenerateResponse(
            success=result.get("success", False),
            query=request.query,
            generation_type=request.generation_type,
            content=result.get("content", ""),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            saved=result.get("saved", False),
            saved_path=result.get("saved_path"),
            mode=result.get("mode"),
            gen_params=result.get("gen_params"),
        )
    except Exception as e:
        logging.exception("story_generate failed")
        raise HTTPException(status_code=500, detail=str(e))


@create_eval_router.post("/story_evaluate", response_model=StoryEvaluateResponse)
async def story_evaluate(request: StoryEvaluateRequest):
    try:
        result = await asyncio.to_thread(
            orchestrator.evaluate_story,
            story_text=request.story,
            context=request.context,
            save=request.save,
        )
        return StoryEvaluateResponse(
            success=result.get("success", False),
            average_score=result.get("average_score", 0.0),
            scores=result.get("scores", {}),
            summary=result.get("summary", ""),
            suggestions=result.get("suggestions", []),
            metadata=result.get("metadata", {}),
            conclusion=result.get("conclusion", ""),
        )
    except Exception as e:
        logging.exception("story_evaluate failed")
        raise HTTPException(status_code=500, detail=str(e))


@create_eval_router.post("/story_evaluate_file", response_model=StoryEvaluateResponse)
async def story_evaluate_file(request: StoryEvaluateFileRequest):
    try:
        result = await asyncio.to_thread(
            orchestrator.evaluate_story_file,
            story_path=request.story_path,
            context_path=request.context_path,
            save=request.save,
        )
        return StoryEvaluateResponse(
            success=result.get("success", False),
            average_score=result.get("average_score", 0.0),
            scores=result.get("scores", {}),
            summary=result.get("summary", ""),
            suggestions=result.get("suggestions", []),
            metadata=result.get("metadata", {}),
            conclusion=result.get("conclusion", ""),
        )
    except Exception as e:
        logging.exception("story_evaluate_file failed")
        raise HTTPException(status_code=500, detail=str(e))


@create_eval_router.post("/summary_generate", response_model=SummaryGenerateResponse)
async def summary_generate(request: SummaryGenerateRequest):
    try:
        result = await asyncio.to_thread(
            orchestrator.generate_summary,
            story_path=request.story_path,
        )
        return SummaryGenerateResponse(
            success=result.get("success", False),
            summary=result.get("summary", ""),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            saved=result.get("saved", False),
            saved_path=result.get("saved_path"),
        )
    except Exception as e:
        logging.exception("summary_generate failed")
        raise HTTPException(status_code=500, detail=str(e))


@create_eval_router.post("/summary_evaluate", response_model=SummaryEvaluateResponse)
async def summary_evaluate(request: SummaryEvaluateRequest):
    try:
        result = await asyncio.to_thread(
            orchestrator.evaluate_summary,
            summary_path=request.summary_path,
            context=request.context,
            save=request.save,
        )
        return SummaryEvaluateResponse(
            success=result.get("success", False),
            average_score=result.get("average_score", 0.0),
            scores=result.get("scores", {}),
            summary=result.get("summary", ""),
            suggestions=result.get("suggestions", []),
            metadata=result.get("metadata", {}),
            conclusion=result.get("conclusion", ""),
        )
    except Exception as e:
        logging.exception("summary_evaluate failed")
        raise HTTPException(status_code=500, detail=str(e))


@create_eval_router.post("/summary_evaluate_file", response_model=SummaryEvaluateResponse)
async def summary_evaluate_file(request: SummaryEvaluateFileRequest):
    try:
        result = await asyncio.to_thread(
            orchestrator.evaluate_summary,
            summary_path=request.summary_path,
            context=request.context_path,
            save=request.save,
        )
        return SummaryEvaluateResponse(
            success=result.get("success", False),
            average_score=result.get("average_score", 0.0),
            scores=result.get("scores", {}),
            summary=result.get("summary", ""),
            suggestions=result.get("suggestions", []),
            metadata=result.get("metadata", {}),
            conclusion=result.get("conclusion", ""),
        )
    except Exception as e:
        logging.exception("summary_evaluate_file failed")
        raise HTTPException(status_code=500, detail=str(e))


@create_eval_router.get("/status", response_model=StatusResponse)
async def create_eval_status():
    return StatusResponse(success=True, message="API is running", type="create_eval")
