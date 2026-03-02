import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from Orchestrator.orchestrator import Orchestrator
from Vector_Store.chromadb import Collection, get_all_data, get_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

vector_store_inspect_router = APIRouter(tags=["Vector Store"])
vector_store_router = APIRouter(prefix="/vector_store", tags=["Vector Store"])
_orchestrator = Orchestrator()

vector_store_check_router = APIRouter(prefix="/check_vector_store", tags=["Vector Store"])

def _normalize_ids(ids):
    if not ids:
        return []
    if isinstance(ids, list) and ids and isinstance(ids[0], str):
        return ids
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return ids[0]
    return list(ids)

def _normalize_docs(docs):
    if not docs:
        return []
    if isinstance(docs, list) and docs and isinstance(docs[0], str):
        return docs
    if isinstance(docs, list) and docs and isinstance(docs[0], list):
        return docs[0]
    return list(docs)

def _normalize_metas(metas):
    if not metas:
        return []
    if isinstance(metas, list) and metas and isinstance(metas[0], dict):
        return metas
    if isinstance(metas, list) and metas and isinstance(metas[0], list):
        return metas[0]
    return list(metas)

class VectorStoreCountResponse(BaseModel):
    success: bool
    count: int = 0
    error: Optional[str] = None

class VectorStoreIdsResponse(BaseModel):
    success: bool
    ids: list[str] = Field(default_factory=list)
    count: int = 0
    error: Optional[str] = None

class VectorStoreItemResponse(BaseModel):
    success: bool
    id: Optional[str] = None
    document: Optional[str] = None
    metadatas: Optional[dict] = None
    error: Optional[str] = None

class VectorStoreListResponse(BaseModel):
    success: bool
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    limit: Optional[int] = None
    offset: int = 0
    error: Optional[str] = None

class VectorStoreStatusResponse(BaseModel):
    success: bool
    message: str = ""
    collection_name: Optional[str] = None
    count: int = 0
    error: Optional[str] = None

class VectorStoreQueryRequest(BaseModel):
    vector_store: dict = Field(default_factory=dict)
    collection: str = ""
    query: Optional[str] = None
    n_results: int = 5
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vector_store": {},
                "collection": "English_Stories",
                "query": "Sherlock Holmes detective mystery",
                "n_results": 5,
            }
        }
    )

class VectorStoreInsertRequest(BaseModel):
    vector_store: dict
    collection: str
    ids: list[str]
    metadata: dict
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vector_store": {},
                "collection": "English_Stories",
                "ids": ["id_1001"],
                "metadata": {
                    "Title": "The Adventures of Sherlock Holmes",
                    "Author": "Arthur Conan Doyle",
                    "Summary": "A detective mystery collection.",
                    "document": "Mr. Sherlock Holmes, who was usually very late in the mornings...",
                },
            }
        }
    )

class VectorStoreUpdateRequest(BaseModel):
    vector_store: dict
    collection: str
    ids: list[str]
    metadata: dict
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vector_store": {},
                "collection": "English_Stories",
                "ids": ["id_1001"],
                "metadata": {
                    "document": "Updated story content for this id.",
                    "Summary": "Updated summary text.",
                },
            }
        }
    )

class VectorStoreDeleteRequest(BaseModel):
    vector_store: dict
    collection: str
    ids: list[str]
    metadata: Optional[dict] = None
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vector_store": {},
                "collection": "English_Stories",
                "ids": ["id_1001", "id_1002"],
                "metadata": {},
            }
        }
    )

class VectorStoreIngestRequest(BaseModel):
    merged_output: Optional[str] = None
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "merged_output": "Data_Merged"
            }
        }
    )

class VectorStoreQueryResponse(BaseModel):
    success: bool
    vector_store: dict = Field(default_factory=dict)
    collection: str = ""
    results: list = Field(default_factory=list)

class VectorStoreInsertResponse(BaseModel):
    success: bool
    vector_store: dict
    collection: str
    ids: list[str]
    metadata: dict

class VectorStoreUpdateResponse(BaseModel):
    success: bool
    vector_store: dict
    collection: str
    ids: list[str]
    metadata: dict

class VectorStoreDeleteResponse(BaseModel):
    success: bool
    vector_store: dict
    collection: str
    ids: list[str]
    metadata: dict = Field(default_factory=dict)

class VectorStoreIngestResponse(BaseModel):
    success: bool
    count: int = 0
    merged_output: Optional[str] = None
    error: Optional[str] = None

@vector_store_inspect_router.get("/count", response_model=VectorStoreCountResponse)
async def get_vector_store_count():
    try:
        result = await asyncio.to_thread(get_all_data)
        if result is None:
            return VectorStoreCountResponse(success=False, count=0, error="Failed to get collection data")
        ids = _normalize_ids(result.get("ids") or [])
        return VectorStoreCountResponse(success=True, count=len(ids))
    except Exception as e:
        logging.exception("get_vector_store_count failed")
        return VectorStoreCountResponse(success=False, count=0, error=str(e))

@vector_store_inspect_router.get("/ids", response_model=VectorStoreIdsResponse)
async def get_vector_store_ids(limit: Optional[int] = None, offset: int = 0):
    try:
        result = await asyncio.to_thread(get_all_data)
        if result is None:
            return VectorStoreIdsResponse(success=False, ids=[], count=0, error="Failed to get collection data")
        ids = _normalize_ids(result.get("ids") or [])
        total = len(ids)
        if offset > 0:
            ids = ids[offset:]
        if limit is not None and limit > 0:
            ids = ids[:limit]
        return VectorStoreIdsResponse(success=True, ids=ids, count=total)
    except Exception as e:
        logging.exception("get_vector_store_ids failed")
        return VectorStoreIdsResponse(success=False, ids=[], count=0, error=str(e))

@vector_store_inspect_router.get("/item/{item_id}", response_model=VectorStoreItemResponse)
async def get_vector_store_item(item_id: str):
    try:
        result = await asyncio.to_thread(get_data, item_id)
        if result is None or not result.get("ids"):
            return VectorStoreItemResponse(success=False, id=item_id, error="Item not found or error reading collection")
        ids = _normalize_ids(result["ids"])
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        if isinstance(docs, list) and docs and not isinstance(docs[0], str):
            docs = docs[0] if docs else []
        if isinstance(metas, list) and metas and isinstance(metas[0], list):
            metas = metas[0] if metas else []
        doc = docs[0] if docs else None
        meta = metas[0] if metas else None
        return VectorStoreItemResponse(success=True, id=ids[0] if ids else item_id, document=doc, metadatas=meta)
    except Exception as e:
        logging.exception("get_vector_store_item failed")
        return VectorStoreItemResponse(success=False, id=item_id, error=str(e))

@vector_store_inspect_router.get("/list", response_model=VectorStoreListResponse)
async def list_vector_store(limit: Optional[int] = 50, offset: int = 0):
    try:
        result = await asyncio.to_thread(get_all_data)
        if result is None:
            return VectorStoreListResponse(success=False, items=[], count=0, error="Failed to get collection data")
        ids = _normalize_ids(result.get("ids") or [])
        docs = _normalize_docs(result.get("documents") or [])
        metas = _normalize_metas(result.get("metadatas") or [])
        total = len(ids)
        end = min(offset + (limit or total), total)
        slice_ids = ids[offset:end]
        slice_docs = docs[offset:end] if len(docs) >= end else (docs[offset:] + [None] * (end - offset - len(docs)))
        slice_metas = metas[offset:end] if len(metas) >= end else (metas[offset:] + [{}] * (end - offset - len(metas)))
        items = [
            {
                "id": i,
                "metadatas": m,
                "document_preview": (d[:200] + "..." if d and len(d) > 200 else d) if d else None,
            }
            for i, d, m in zip(slice_ids, slice_docs, slice_metas)
        ]
        return VectorStoreListResponse(success=True, items=items, count=total, limit=limit, offset=offset)
    except Exception as e:
        logging.exception("list_vector_store failed")
        return VectorStoreListResponse(success=False, items=[], count=0, error=str(e))

@vector_store_inspect_router.get("/health", response_model=VectorStoreStatusResponse)
async def vector_store_check_status():
    try:
        result = await asyncio.to_thread(get_all_data)
        if result is None:
            return VectorStoreStatusResponse(success=False, message="Collection unreachable", error="get_all_data returned None")
        ids = _normalize_ids(result.get("ids") or [])
        return VectorStoreStatusResponse(
            success=True,
            message="Vector store is reachable",
            collection_name=Collection.name,
            count=len(ids),
        )
    except Exception as e:
        logging.exception("vector_store_health failed")
        return VectorStoreStatusResponse(success=False, message="Error", error=str(e))

@vector_store_inspect_router.get("/status", response_model=VectorStoreStatusResponse)
async def vector_store_check_status_alias():
    return await vector_store_check_status()

@vector_store_router.post("/query", response_model=VectorStoreQueryResponse)
async def vector_store_query(request: VectorStoreQueryRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.query_vector_store,
            query=request.query or "",
            n_results=request.n_results,
            collection=request.collection,
        )
        return VectorStoreQueryResponse(
            success=result.get("success", True),
            vector_store=request.vector_store,
            collection=request.collection,
            results=result.get("results", []),
        )
    except Exception as e:
        logging.exception("vector_store_query failed")
        raise HTTPException(status_code=500, detail=str(e))

@vector_store_router.post("/insert", response_model=VectorStoreInsertResponse)
async def vector_store_insert(request: VectorStoreInsertRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.vector_store_insert,
            collection=request.collection,
            ids=request.ids,
            metadata=request.metadata,
        )
        return VectorStoreInsertResponse(
            success=result.get("success", False),
            vector_store=request.vector_store,
            collection=request.collection,
            ids=request.ids,
            metadata=request.metadata,
        )
    except Exception as e:
        logging.exception("vector_store_insert failed")
        raise HTTPException(status_code=500, detail=str(e))

@vector_store_router.post("/update", response_model=VectorStoreUpdateResponse)
async def vector_store_update(request: VectorStoreUpdateRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.vector_store_update,
            collection=request.collection,
            ids=request.ids,
            metadata=request.metadata,
        )
        return VectorStoreUpdateResponse(
            success=result.get("success", False),
            vector_store=request.vector_store,
            collection=request.collection,
            ids=request.ids,
            metadata=request.metadata,
        )
    except Exception as e:
        logging.exception("vector_store_update failed")
        raise HTTPException(status_code=500, detail=str(e))

@vector_store_router.post("/delete", response_model=VectorStoreDeleteResponse)
async def vector_store_delete(request: VectorStoreDeleteRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.vector_store_delete,
            collection=request.collection,
            ids=request.ids,
            metadata=request.metadata or {},
        )
        return VectorStoreDeleteResponse(
            success=result.get("success", False),
            vector_store=request.vector_store,
            collection=request.collection,
            ids=request.ids,
            metadata=request.metadata or {},
        )
    except Exception as e:
        logging.exception("vector_store_delete failed")
        raise HTTPException(status_code=500, detail=str(e))

@vector_store_router.post("/ingest", response_model=VectorStoreIngestResponse)
async def vector_store_ingest(request: VectorStoreIngestRequest):
    try:
        result = await asyncio.to_thread(
            _orchestrator.ingest,
            merged_output=request.merged_output,
        )
        return VectorStoreIngestResponse(
            success=result.get("success", False),
            count=result.get("count", 0),
            merged_output=request.merged_output,
            error=result.get("error"),
        )
    except Exception as e:
        logging.exception("vector_store_ingest failed")
        raise HTTPException(status_code=500, detail=str(e))

vector_store_router.include_router(vector_store_inspect_router, prefix="")
vector_store_check_router.include_router(vector_store_inspect_router, prefix="")
