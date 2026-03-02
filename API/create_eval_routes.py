import asyncio
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import yaml
from pathlib import Path

from Generative_AI.generative_ai import Gen_mode
from Orchestrator.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_config_file = Path(__file__).parent.parent / "setup.yaml"
_config = yaml.safe_load(_config_file.read_text(encoding="utf-8"))
_DEFAULT_N_RESULTS = _config.get("Story_generation_n_results", 1)

create_eval_router = APIRouter(prefix="/create-eval", tags=["Create Eval"])
orchestrator = Orchestrator()


def _parse_gen_mode(mode: Optional[str]) -> Gen_mode:
    if not mode:
        return Gen_mode.FAST
    return Gen_mode.THINKING if str(mode).lower() == "thinking" else Gen_mode.FAST


class StoryGenerateRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Story topic or query")
    generation_type: str = Field(default="full_story", description="'summary' or 'full_story'")
    save: bool = Field(default=True)
    n_results: int = Field(default=_DEFAULT_N_RESULTS, ge=1, le=20, description="Number of context chunks from vector store (default from setup.yaml)")
    mode: Optional[str] = Field(default="fast", description="Generation mode: 'fast' or 'thinking'")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "A mystery in old London",
                "generation_type": "full_story",
                "save": True,
                "n_results": _DEFAULT_N_RESULTS,
                "mode": "fast",
            }
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
    story_path: str = Field(..., description="Path to the story file")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "story_path": "C:/Users/User/Documents/Mini-project/New folder/Generated_Stories/20260219_062746_Timestamp_generated_story.txt"
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
    story: str = Field(..., min_length=10)
    context: Optional[str] = None
    save: bool = False
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "story": "Once upon a time, a detective solved a mystery in London...",
                "context": None,
                "save": True,
            }
        }
    )


class StoryEvaluateFileRequest(BaseModel):
    story_path: str = Field(...)
    context_path: Optional[str] = None
    save: bool = False
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "story_path": "C:/Users/User/Documents/Mini-project/New folder/Generated_Stories/20260219_062746_Timestamp_generated_story.txt",
                "context_path": None,
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
    summary_path: str = Field(...)
    context: Optional[str] = None
    save: bool = False
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary_path": "C:/Users/User/Documents/Mini-project/New folder/Summarized_Stories/20260219_summary_example.txt",
                "context": None,
                "save": True,
            }
        }
    )


class SummaryEvaluateFileRequest(BaseModel):
    summary_path: str = Field(...)
    context_path: Optional[str] = None
    save: bool = False
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary_path": "C:/Users/User/Documents/Mini-project/New folder/Summarized_Stories/20260219_summary_example.txt",
                "context_path": "C:/Users/User/Documents/Mini-project/New folder/Generated_Stories/20260219_062746_Timestamp_generated_story.txt",
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
            mode=_parse_gen_mode(request.mode),
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
