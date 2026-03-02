import asyncio
import logging
from pathlib import Path
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from Orchestrator.orchestrator import Orchestrator
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_orchestrator = Orchestrator()

book_router = APIRouter(prefix="/book", tags=["Book"])
data_router = APIRouter(prefix="/data", tags=["Data"])

class SearchBookRequest(BaseModel):
    query: str = Field(..., min_length=1)
    n_results: int = Field(default=5, ge=1, le=20)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "The Adventures of Sherlock Holmes",
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
                "query": "The Adventures of Sherlock Holmes",
                "urls": [
                    "https://archive.org/details/example-book"
                ],
                "formats": ["pdf", "epub"],
            }
        }
    )

class MetadataCheckRequest(BaseModel):
    folder_path: str
    file_name: str
    metadata: Optional[dict] = None
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "folder_path": "C:/Users/User/Documents/Mini-project/New folder",
                "file_name": "Metadata_input",
                "metadata": {},
            }
        }
    )

class MetadataUpdateRequest(BaseModel):
    metadata: dict
    folder_path: str
    file_name: str
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metadata": {
                    "Author": "Arthur Conan Doyle",
                    "Title": "The Adventures of Sherlock Holmes",
                    "Summary": "A collection of detective stories.",
                },
                "folder_path": "C:/Users/User/Documents/Mini-project/New folder",
                "file_name": "example_metadata.json",
            }
        }
    )

class MergedDataRequest(BaseModel):
    metadata_input: Optional[str] = None
    story_input: Optional[str] = None
    merged_output: Optional[str] = None
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metadata_input": "Metadata_input",
                "story_input": "Story_input",
                "merged_output": "Data_Merged",
            }
        }
    )

class DataSummaryCheckRequest(BaseModel):
    folder_path: Optional[str] = None
    file_name: Optional[str] = None
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "folder_path": "C:/Users/User/Documents/Mini-project/New folder",
                "file_name": "Metadata_input",
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
            }
        }
    )

class MetadataCheckResponse(BaseModel):
    success: bool
    metadata: dict = Field(default_factory=dict)
    folder_path: str
    file_name: str
    issues: list = Field(default_factory=list)
    number_books: int = 0
    empty_number: int = 0

class MetadataUpdateResponse(BaseModel):
    success: bool
    metadata: dict
    folder_path: str
    file_name: str

class MergedDataResponse(BaseModel):
    success: bool
    metadata: dict = Field(default_factory=dict)
    folder_path: str = ""
    file_name: str = ""
    total_results: int = 0

class DataMetadataTemplateResponse(BaseModel):
    success: bool
    message: str = ""
    template_count: int = 0

class DataSummaryCheckResponse(BaseModel):
    success: bool
    folder_path: str = ""
    file_name: str = ""
    issues: list = Field(default_factory=list)
    number_books: int = 0
    empty_number: int = 0

class DataSummariesCreateResponse(BaseModel):
    success: bool
    summaries_created: int = 0
    error: Optional[str] = None

class DataSummariesCreateAndCheckResponse(BaseModel):
    success: bool
    summaries_created: int = 0
    empty_count: int = 0
    created_count: int = 0
    counts_match: bool = False
    error: Optional[str] = None

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
    output_folder = str(base_path / str(_orchestrator.config.get("Extracted_text_dir", "")))

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

@book_router.post("/metadata_check", response_model=MetadataCheckResponse)
async def metadata_check(request: MetadataCheckRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.metadata_check,
            folder_path=request.folder_path,
            file_name=request.file_name,
            number_books=0,
            empty_number=0,
        )
        return MetadataCheckResponse(
            success=result.get("success", False),
            metadata=request.metadata or {},
            folder_path=request.folder_path,
            file_name=request.file_name,
            issues=result.get("issues", []),
            number_books=result.get("number_books", 0),
            empty_number=result.get("empty_number", 0),
        )
    except Exception as e:
        logging.exception("metadata_check failed")
        raise HTTPException(status_code=500, detail=str(e))

@book_router.post("/metadata_update", response_model=MetadataUpdateResponse)
async def metadata_update(request: MetadataUpdateRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.metadata_update,
            metadata=request.metadata,
            folder_path=request.folder_path,
            file_name=request.file_name,
        )
        return MetadataUpdateResponse(
            success=result.get("success", False),
            metadata=request.metadata,
            folder_path=request.folder_path,
            file_name=request.file_name,
        )
    except Exception as e:
        logging.exception("metadata_update failed")
        raise HTTPException(status_code=500, detail=str(e))

@book_router.get("/status", response_model=StatusResponse)
async def book_status():
    return StatusResponse(success=True, message="API is running", type="book")

@data_router.post("/merged", response_model=MergedDataResponse)
async def merged_data(request: MergedDataRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.merge_data,
            metadata_input=request.metadata_input,
            story_input=request.story_input,
            merged_output=request.merged_output,
        )
        return MergedDataResponse(
            success=result.get("success", False),
            metadata={},
            folder_path="",
            file_name="",
            total_results=result.get("total_results", 0),
        )
    except Exception as e:
        logging.exception("merged_data failed")
        raise HTTPException(status_code=500, detail=str(e))

@data_router.post("/metadata_template_create", response_model=DataMetadataTemplateResponse)
async def metadata_template_create():
    try:
        result = await asyncio.to_thread(_orchestrator.create_metadata_template)
        return DataMetadataTemplateResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            template_count=result.get("template_count", 0),
        )
    except Exception as e:
        logging.exception("metadata_template_create failed")
        raise HTTPException(status_code=500, detail=str(e))

@data_router.post("/metadata_summary_check", response_model=DataSummaryCheckResponse)
async def metadata_summary_check(request: DataSummaryCheckRequest):
    folder_path = request.folder_path or _orchestrator.base_path
    file_name = request.file_name or _orchestrator.metadata_input
    try:
        result = await asyncio.to_thread(
            _orchestrator.metadata_check,
            folder_path=folder_path,
            file_name=file_name,
            number_books=0,
            empty_number=0,
        )
        return DataSummaryCheckResponse(
            success=result.get("success", False),
            folder_path=folder_path,
            file_name=file_name,
            issues=result.get("issues", []),
            number_books=result.get("number_books", 0),
            empty_number=result.get("empty_number", 0),
        )
    except Exception as e:
        logging.exception("metadata_summary_check failed")
        raise HTTPException(status_code=500, detail=str(e))

@data_router.post("/summaries_create", response_model=DataSummariesCreateResponse)
async def summaries_create():
    try:
        result = await asyncio.to_thread(_orchestrator.create_summaries)
        return DataSummariesCreateResponse(
            success=result.get("success", False),
            summaries_created=result.get("summaries_created", 0),
            error=result.get("error"),
        )
    except Exception as e:
        logging.exception("summaries_create failed")
        raise HTTPException(status_code=500, detail=str(e))

@data_router.post("/summaries_create_and_check", response_model=DataSummariesCreateAndCheckResponse)
async def summaries_create_and_check():
    try:
        result = await asyncio.to_thread(_orchestrator.create_summaries_and_check)
        return DataSummariesCreateAndCheckResponse(
            success=result.get("success", False),
            summaries_created=result.get("summaries_created", 0),
            empty_count=result.get("empty_count", 0),
            created_count=result.get("created_count", 0),
            counts_match=result.get("counts_match", False),
            error=result.get("error"),
        )
    except Exception as e:
        logging.exception("summaries_create_and_check failed")
        raise HTTPException(status_code=500, detail=str(e))

@data_router.get("/merged_status", response_model=MergedDataResponse)
async def merged_data_status():
    return MergedDataResponse(success=True, metadata={}, folder_path="", file_name="", total_results=0)
