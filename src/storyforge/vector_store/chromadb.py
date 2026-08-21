from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import chromadb

from storyforge.config.config import load_config
from storyforge.vector_store.embeddings import DEFAULT_EMBED_MODEL, embed_query

# Root logging is configured once in storyforge/__init__.py.

ROOT_DIR = Path(__file__).resolve().parents[3]  # repo root

_Client: Optional[chromadb.PersistentClient] = None
_Collection = None


def get_or_create_collection(collection_name: str | None = None):
    global _Client
    cfg = load_config()
    base_path = Path(str(cfg.get("BASE_PATH") or ROOT_DIR))
    chroma_rel = str(cfg.get("Chroma_path") or "data/chroma_db")
    chroma_full_path = (base_path / chroma_rel).resolve()
    chroma_full_path.mkdir(parents=True, exist_ok=True)

    if _Client is None:
        _Client = chromadb.PersistentClient(path=str(chroma_full_path))

    default_collection = cfg.get("Chroma_collection_name") or "StoryForgeRag_v1"
    name = (collection_name or default_collection).strip()
    if not name:
        raise ValueError("collection_name cannot be empty")
    return _Client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


Collection = get_or_create_collection()


def set_active_collection(collection_name: str):
    global Collection
    Collection = get_or_create_collection(collection_name)
    return Collection


def query_data(
    query: str,
    n_results: int = 5,
    query_type: str = "content",
    collection_name: str | None = None,
):
    """Query the collection using the SAME embedding model the corpus was ingested with.

    This previously passed ``query_texts=[query]``. Because the collection is
    created without an ``embedding_function``, that made Chroma embed the query
    with its own built-in default (all-MiniLM-L6-v2, 384-dim). The stored
    vectors are 768-dim BGE, so every call raised a dimension mismatch that was
    swallowed below and returned as ``None`` -- a silent, total failure.

    We now embed the query explicitly with ``Vector_store_model`` (plus the BGE
    query prefix) and pass ``query_embeddings``, mirroring what the ingest path
    and ``rag/retrieval.py`` already do.
    """
    try:
        cfg = load_config()
        embed_model_name = str(cfg.get("Vector_store_model") or DEFAULT_EMBED_MODEL)
        embed_device = str(cfg.get("Embedding_device") or "cpu").strip().lower() or "cpu"
        vector = embed_query(query, embed_model_name, device=embed_device)
        if vector is None:
            logging.error(
                "query_data: could not embed query with %r — refusing to fall back to "
                "Chroma's default embedder, which would not match the stored vectors.",
                embed_model_name,
            )
            return None

        collection = get_or_create_collection(collection_name) if collection_name else Collection
        results = collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            where={"query_type": query_type},
            include=["distances", "documents", "metadatas"],
        )
        return results
    except Exception as e:
        logging.error(f"Error querying data: {e}")
        return None


def delete_data(id: str):
    try:
        Collection.delete(ids=[id])
        return True
    except Exception as e:
        logging.error(f"Error deleting data: {e}")
        return False


def get_data(id: str):
    try:
        results = Collection.get(ids=[id])
        return results
    except Exception as e:
        logging.error(f"Error getting data: {e}")
        return None


def update_data(id: str, new_data: str):
    try:
        Collection.update(ids=[id], documents=[new_data])
        return True
    except Exception as e:
        logging.error(f"Error updating data: {e}")
        return False


def get_all_data():
    try:
        results = Collection.get(include=["documents", "metadatas"])
        return results
    except Exception as e:
        logging.error(f"Error getting all data: {e}")
        return None


def get_chroma_storage_path(*, base_path: str | Path | None = None, chroma_path: str | None = None) -> Path:
    cfg = load_config()
    root = Path(base_path or cfg.get("BASE_PATH") or ROOT_DIR)
    rel = chroma_path or str(cfg.get("Chroma_path") or "data/chroma_db")
    return (root / rel).resolve()


def reset_vector_store_dir(
    *,
    base_path: str | Path | None = None,
    chroma_path: str | None = None,
    new_collection_name: str = "StoryForgeRag_v1",
) -> dict:
    """
    Aggressive reset: deletes the entire Chroma directory on disk and recreates it.
    Also switches the module-level Client/Collection to `new_collection_name`.
    """
    global _Client, Collection
    try:
        storage = get_chroma_storage_path(base_path=base_path, chroma_path=chroma_path)
        if storage.exists():
            shutil.rmtree(storage, ignore_errors=True)
        storage.mkdir(parents=True, exist_ok=True)

        _Client = chromadb.PersistentClient(path=str(storage))
        Collection = _Client.get_or_create_collection(
            name=new_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return {
            "success": True,
            "message": "Vector store reset",
            "chroma_path": str(storage),
            "collection_name": Collection.name,
        }
    except Exception as e:
        logging.exception("reset_vector_store_dir failed")
        return {"success": False, "message": "reset failed", "error": str(e)}
