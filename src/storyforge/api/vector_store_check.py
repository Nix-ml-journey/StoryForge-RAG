import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from storyforge.vector_store.chromadb import (
    Collection,
    get_all_data,
    get_data,
    reset_vector_store_dir,
    set_active_collection,
)
from storyforge.vector_store.ingest_stories import ingest_stories_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

vector_store_inspect_router = APIRouter(tags=["Vector Store"])
vector_store_router = APIRouter(prefix="/vector_store", tags=["Vector Store"])
# Inspection routes call chromadb helpers directly (not the ingest wrapper).

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
    collection: str = "StoryForgeRag_v1"
    query: Optional[str] = None
    n_results: int = 5
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "a brave knight and a dragon",
                "n_results": 5,
            }
        }
    )


class VectorStoreInsertRequest(BaseModel):
    vector_store: dict = Field(default_factory=dict)
    collection: str = "StoryForgeRag_v1"
    ids: list[str]
    metadata: dict
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ids": ["id_01"],
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
    vector_store: dict = Field(default_factory=dict)
    collection: str = "StoryForgeRag_v1"
    ids: list[str]
    metadata: dict
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ids": ["id_01"],
                "metadata": {
                    "document": "Updated story content goes here.",
                    "Summary": "Updated summary text.",
                },
            }
        }
    )


class VectorStoreDeleteRequest(BaseModel):
    vector_store: dict = Field(default_factory=dict)
    collection: str = "StoryForgeRag_v1"
    ids: list[str]
    metadata: Optional[dict] = None
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ids": ["id_14"],
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


class VectorStoreResetResponse(BaseModel):
    success: bool
    message: str = ""
    chroma_path: Optional[str] = None
    collection_name: Optional[str] = None
    error: Optional[str] = None


class VectorStoreIngestStoriesResponse(BaseModel):
    success: bool
    files_seen: int = 0
    chunks_written: int = 0
    collection_name: str = ""
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
            return VectorStoreItemResponse(
                success=False, id=item_id, error="Item not found or error reading collection"
            )
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
            return VectorStoreStatusResponse(
                success=False, message="Collection unreachable", error="get_all_data returned None"
            )
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
        await asyncio.to_thread(set_active_collection, request.collection)
        raw = await asyncio.to_thread(
            Collection.query,
            query_texts=[request.query or ""],
            n_results=int(request.n_results or 5),
            where={"query_type": "content"},
            include=["distances", "documents", "metadatas"],
        )
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        results = []
        for rid, doc, meta, dist in zip(ids, docs, metas, dists):
            results.append({"id": rid, "document": doc, "metadata": meta, "distance": dist})
        return VectorStoreQueryResponse(
            success=True,
            vector_store=request.vector_store,
            collection=request.collection,
            results=results,
        )
    except Exception as e:
        logging.exception("vector_store_query failed")
        raise HTTPException(status_code=500, detail=str(e))


@vector_store_router.post("/insert", response_model=VectorStoreInsertResponse)
async def vector_store_insert(request: VectorStoreInsertRequest):
    try:
        await asyncio.to_thread(set_active_collection, request.collection)
        doc = request.metadata.get("document", request.metadata.get("content", ""))
        meta = {k: v for k, v in request.metadata.items() if k not in {"document", "content"}}
        await asyncio.to_thread(Collection.add, ids=request.ids, metadatas=[meta] * len(request.ids), documents=[doc] * len(request.ids))
        return VectorStoreInsertResponse(
            success=True,
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
        await asyncio.to_thread(set_active_collection, request.collection)
        new_doc = request.metadata.get("document", request.metadata.get("content", None))
        if new_doc is not None:
            await asyncio.to_thread(Collection.update, ids=request.ids, documents=[str(new_doc)] * len(request.ids))
        return VectorStoreUpdateResponse(
            success=True,
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
        await asyncio.to_thread(set_active_collection, request.collection)
        await asyncio.to_thread(Collection.delete, ids=request.ids)
        return VectorStoreDeleteResponse(
            success=True,
            vector_store=request.vector_store,
            collection=request.collection,
            ids=request.ids,
            metadata=request.metadata or {},
        )
    except Exception as e:
        logging.exception("vector_store_delete failed")
        raise HTTPException(status_code=500, detail=str(e))


@vector_store_router.post("/reset", response_model=VectorStoreResetResponse)
async def vector_store_reset():
    try:
        result = await asyncio.to_thread(
            reset_vector_store_dir,
            new_collection_name="StoryForgeRag_v1",
        )
        return VectorStoreResetResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            chroma_path=result.get("chroma_path"),
            collection_name=result.get("collection_name"),
            error=result.get("error"),
        )
    except Exception as e:
        logging.exception("vector_store_reset failed")
        raise HTTPException(status_code=500, detail=str(e))


@vector_store_router.post("/ingest_stories", response_model=VectorStoreIngestStoriesResponse)
async def vector_store_ingest_stories():
    try:
        # Ensure requests hit the expected new collection.
        await asyncio.to_thread(set_active_collection, "StoryForgeRag_v1")
        res = await asyncio.to_thread(
            ingest_stories_dir,
            collection_name="StoryForgeRag_v1",
        )
        return VectorStoreIngestStoriesResponse(
            success=res.success,
            files_seen=res.files_seen,
            chunks_written=res.chunks_written,
            collection_name=res.collection_name,
            error=res.error,
        )
    except Exception as e:
        logging.exception("vector_store_ingest_stories failed")
        raise HTTPException(status_code=500, detail=str(e))


vector_store_router.include_router(vector_store_inspect_router, prefix="")
vector_store_check_router.include_router(vector_store_inspect_router, prefix="")
