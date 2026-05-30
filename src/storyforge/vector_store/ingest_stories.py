"""
Ingest story .txt files into Chroma with the same embedding model used at query time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from storyforge.config.config import load_config

from storyforge.vector_store.chromadb import get_or_create_collection

LOG = logging.getLogger(__name__)

_EMBED_MODEL_CACHE: dict = {}  # model_name -> SentenceTransformer


def _get_embed_model(model_name: str):
    """Cached embedding model — must match langchain_rag retrieval vectors."""
    if model_name in _EMBED_MODEL_CACHE:
        return _EMBED_MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOG.info("Loading embedding model %s on %s", model_name, device)
        model = SentenceTransformer(model_name, device=device)
        _EMBED_MODEL_CACHE[model_name] = model
        return model
    except ImportError:
        LOG.warning(
            "sentence-transformers not installed — falling back to chromadb default embeddings. "
            "Run: pip install sentence-transformers"
        )
        return None


def _embed_chunks(model, texts: list[str], *, is_bge: bool = False) -> list[list[float]] | None:
    """Embed a list of texts. Returns None when the model is unavailable."""
    if model is None:
        return None
    try:
        # BGE: passage prefix on ingest (query prefix is applied only at search time).
        if is_bge:
            texts = ["Represent this passage for retrieval: " + t for t in texts]
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]
    except Exception as e:
        LOG.warning("Embedding failed, falling back to chromadb default: %s", e)
        return None


@dataclass(frozen=True)
class IngestResult:
    success: bool
    files_seen: int = 0
    chunks_written: int = 0
    collection_name: str = ""
    error: Optional[str] = None


def _iter_story_files(stories_dir: Path) -> Iterable[Path]:
    if not stories_dir.exists():
        return []
    return sorted(
        [p for p in stories_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"], key=lambda p: p.name
    )


def _chunk_text(text: str, *, max_chars: int = 1800, overlap_chars: int = 200) -> list[str]:
    """Split text into ~max_chars chunks; paragraph-first, then sliding window with overlap."""
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        merged = "\n\n".join(buf).strip()
        if merged:
            chunks.append(merged)
        buf = []
        buf_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            start = 0
            while start < len(para):
                end = min(len(para), start + max_chars)
                # Overlap window: align to word/sentence boundaries (avoid ", ..." or "'t ..." starts).
                if start > 0:
                    while start < len(para) and para[start].isalnum() and para[start - 1].isalnum():
                        start += 1
                    while start < len(para) and para[start].isspace():
                        start += 1
                    lookahead_limit = min(len(para), start + max(60, overlap_chars))
                    if (
                        start < lookahead_limit
                        and para[start].islower()
                        and para[start - 1] not in "\n\r"
                        and para[start - 1] not in ".!?"
                    ):
                        window = para[start:lookahead_limit]
                        m = re.search(r"(?:\n+|[.!?]\s+)", window)
                        if m:
                            start = start + m.end()
                            while start < len(para) and para[start].isspace():
                                start += 1
                    while start < lookahead_limit and para[start] in "\"'`“”‘’.,;:!?)-–—":
                        start += 1
                    while start < len(para) and para[start].isspace():
                        start += 1
                    if (
                        start < lookahead_limit
                        and para[start].islower()
                        and para[start - 1] not in "\n\r"
                        and para[start - 1] not in ".!?"
                    ):
                        window = para[start:lookahead_limit]
                        m = re.search(r"(?:\n+|[.!?]\s+)", window)
                        if m:
                            start = start + m.end()
                            while start < len(para) and para[start].isspace():
                                start += 1
                if end < len(para):
                    while end > start and para[end - 1].isalnum() and para[end].isalnum():
                        end -= 1
                    if end > start + 1 and para[end - 1] in "-–—" and end < len(para) and para[end].isalpha():
                        end -= 1
                        while end > start + 1 and not para[end - 1].isspace():
                            end -= 1
                    if end <= start + 50:
                        end = min(len(para), start + max_chars)

                piece = para[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(para):
                    break
                start = max(0, end - overlap_chars)
            continue

        if buf_len + len(para) + (2 if buf else 0) <= max_chars:
            buf.append(para)
            buf_len += len(para) + (2 if buf_len else 0)
        else:
            flush()
            buf.append(para)
            buf_len = len(para)
    flush()

    out = [c for c in chunks if len(c.split()) >= 20]  # drop tiny fragments
    return out or chunks


def ingest_stories_dir(
    *,
    base_path: str | Path | None = None,
    stories_dir: str | Path | None = None,
    collection_name: str = "StoryForgeRag_v1",
    max_chars: int = 1800,
    overlap_chars: int = 200,
) -> IngestResult:
    """Read Story_input/*.txt, chunk, embed, upsert into Chroma."""
    try:
        cfg = load_config()
        root = Path(base_path or cfg.get("BASE_PATH") or Path(__file__).resolve().parents[1])
        stories_path = Path(stories_dir) if stories_dir else root / (cfg.get("Story_input") or "Stories")
        stories_path = stories_path.resolve()

        collection = get_or_create_collection(collection_name)
        files = list(_iter_story_files(stories_path))
        files_seen = len(files)
        chunks_written = 0

        if files_seen == 0:
            return IngestResult(
                success=False,
                files_seen=0,
                chunks_written=0,
                collection_name=collection.name,
                error=f"No .txt files found in {stories_path}",
            )

        # Same embedding model as langchain_rag (required for BGE / non-default models).
        embed_model_name = str(cfg.get("Vector_store_model") or "all-MiniLM-L6-v2")
        is_bge = "bge" in embed_model_name.lower()
        embed_model = _get_embed_model(embed_model_name)
        if embed_model is not None:
            LOG.info("Ingest using embedding model: %s", embed_model_name)
        else:
            LOG.warning("No embedding model loaded — chromadb will use its built-in default.")

        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = f.read_text(encoding="utf-8", errors="replace")

            chunks = _chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
            if not chunks:
                LOG.warning("Skipping empty story file: %s", f.name)
                continue

            ids: list[str] = []
            metadatas: list[dict] = []
            documents: list[str] = []
            title = f.stem
            for i, ch in enumerate(chunks, start=1):
                chunk_id = f"{title}_chunk_{i}"
                ids.append(chunk_id)
                metadatas.append(
                    {
                        "Title": title,
                        "Author": "",
                        "Summary": "",
                        "query_type": "content",
                        "Is_series": False,
                        "chunk_id": chunk_id,  # used by grounded-facts attribution
                    }
                )
                documents.append(ch)

            embeddings = _embed_chunks(embed_model, documents, is_bge=is_bge)
            if embeddings is not None:
                collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            else:
                # Chroma default embeddings — only OK if Vector_store_model is all-MiniLM-L6-v2.
                collection.upsert(ids=ids, metadatas=metadatas, documents=documents)
            chunks_written += len(ids)

        return IngestResult(
            success=True,
            files_seen=files_seen,
            chunks_written=chunks_written,
            collection_name=collection.name,
            error=None,
        )
    except Exception as e:
        LOG.exception("ingest_stories_dir failed")
        return IngestResult(success=False, error=str(e))
