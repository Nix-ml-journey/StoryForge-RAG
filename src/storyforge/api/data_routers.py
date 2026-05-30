import asyncio
import logging
from pathlib import Path
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from storyforge.orchestrator.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_orchestrator = Orchestrator()

book_router = APIRouter(prefix="/book", tags=["Book"])


class SearchBookRequest(BaseModel):
    query: str = Field(..., min_length=1)
    n_results: int = Field(default=5, ge=1, le=20)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Hansel and Gretel",
                "n_results": 5,
            }
        }
    )


class DownloadBookRequest(BaseModel):
    query: str = Field(..., min_length=1)
    urls: list[str] = Field(default_factory=list)
    formats: list[str] = Field(
        default_factory=lambda: ["pdf", "epub"],
        description="Download formats: 'pdf', 'epub'. Default: pdf and epub.",
    )
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Hansel and Gretel",
                "urls": ["https://archive.org/details/hansel-and-gretel"],
                "formats": ["pdf", "epub"],
            }
        }
    )


class SearchBookResponse(BaseModel):
    success: bool
    query: str
    n_results: int
    results: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    urls: list[dict] = Field(default_factory=list)


class StatusResponse(BaseModel):
    success: bool
    message: str
    type: str


class DownloadBookResponse(BaseModel):
    success: bool
    query: str
    n_results: int = 0
    saved: bool = False
    saved_path: Optional[str] = None
    urls: list[str] = Field(default_factory=list)


class ExtractTextResponse(BaseModel):
    success: bool
    message: str = ""
    processed_scope: str = "all_files_in_downloaded_folder"
    input_folder: str = ""
    output_folder: str = ""


class ExtractTextRequest(BaseModel):
    use_default_paths: bool = Field(
        default=True,
        description="Use input/output paths from setup.yaml",
    )
    target: Literal["all"] = Field(
        default="all",
        description="Current extractor runs on all supported files in the downloaded book folder.",
    )
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "use_default_paths": True,
                "target": "all",
            },
            "description": "Both fields are optional — send {} or omit body to use defaults.",
        }
    )


@book_router.post("/search", response_model=SearchBookResponse)
async def search_book(request: SearchBookRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.search_book,
            query=request.query,
            n_results=request.n_results,
        )
        return SearchBookResponse(
            success=result.get("success", False),
            query=request.query,
            n_results=request.n_results,
            results=result.get("results", []),
            metadata=result.get("metadata", {}),
            urls=result.get("urls", []),
        )
    except Exception as e:
        logging.exception("search_book failed")
        raise HTTPException(status_code=500, detail=str(e))


@book_router.post("/download", response_model=DownloadBookResponse)
async def download_book(request: DownloadBookRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.download_book,
            query=request.query,
            urls=request.urls,
            formats=request.formats,
        )
        return DownloadBookResponse(
            success=result.get("success", False),
            query=request.query,
            n_results=len(request.urls),
            saved=result.get("saved", False),
            saved_path=result.get("saved_path"),
            urls=request.urls,
        )
    except Exception as e:
        logging.exception("download_book failed")
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: Legacy merged-data and metadata template endpoints removed.


@book_router.post("/extract_text", response_model=ExtractTextResponse)
async def extract_text(request: ExtractTextRequest):
    if not request.use_default_paths:
        raise HTTPException(
            status_code=400,
            detail="Custom paths are not supported yet. Set use_default_paths=true.",
        )
    if request.target != "all":
        raise HTTPException(status_code=400, detail="Only target='all' is supported.")

    base_path = Path(str(_orchestrator.base_path))
    input_folder = str(base_path / str(_orchestrator.downloaded_rawbook_dir))
    output_folder = str(base_path / str(_orchestrator.config.get("Raw_extracted_dir", "data/raw_extracted")))

    try:
        result = await asyncio.to_thread(_orchestrator.extract_text)
        return ExtractTextResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            processed_scope="all_files_in_downloaded_folder",
            input_folder=input_folder,
            output_folder=output_folder,
        )
    except Exception as e:
        logging.exception("extract_text failed")
        raise HTTPException(status_code=500, detail=str(e))


@book_router.get("/status", response_model=StatusResponse)
async def book_status():
    return StatusResponse(success=True, message="API is running", type="book")

