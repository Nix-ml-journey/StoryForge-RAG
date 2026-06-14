"""Step 1 — Chroma vector retrieval with optional BM25 hybrid fusion.

Public API:
    retrieve_docs(query, cfg, ...) -> list[Document]

Internal helpers also exported for use by extraction.py / langchain_rag.py:
    _docs_to_chunks, _docs_to_context, _format_chunks_for_prompt
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

LOG = logging.getLogger(__name__)

_RERANKER_CACHE: dict = {}    # model_id -> CrossEncoder
_VECTORSTORE_CACHE: dict = {} # (chroma_dir, collection, embed_model) -> Chroma


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def _get_reranker(model_id: str):
    """Load (or return cached) a cross-encoder reranker on GPU."""
    if model_id in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_id]
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOG.info("Loading reranker model %s on %s", model_id, device)
        reranker = CrossEncoder(model_id, device=device)
        _RERANKER_CACHE[model_id] = reranker
        return reranker
    except ImportError:
        LOG.warning(
            "sentence-transformers not installed — reranker disabled. "
            "Run: pip install sentence-transformers"
        )
        return None


def _rerank_docs(
    query: str,
    docs: list,
    *,
    reranker_model_id: str,
    top_n: int = 6,
) -> list:
    """Re-rank retrieved chunks by relevance; return top_n (or unchanged if reranker missing)."""
    if not docs:
        return docs
    reranker = _get_reranker(reranker_model_id)
    if reranker is None:
        return docs[:top_n]
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = [d for _, d in ranked[:top_n]]
    LOG.debug("Reranker kept %d/%d chunks (top_n=%d)", len(result), len(docs), top_n)
    return result


# ---------------------------------------------------------------------------
# Vectorstore
# ---------------------------------------------------------------------------

def _get_paths_and_names(cfg: dict[str, Any]):
    from pathlib import Path
    base = Path(cfg.get("BASE_PATH") or Path(__file__).resolve().parents[3]).resolve()
    chroma_dir = (base / (cfg.get("Chroma_path") or "chroma_db")).resolve()
    collection = cfg.get("Chroma_collection_name") or "StoryForgeRag_v1"
    embed_model = cfg.get("Vector_store_model") or "all-MiniLM-L6-v2"
    return chroma_dir, str(collection), str(embed_model)


def _build_vectorstore(cfg: dict[str, Any]) -> Chroma:
    chroma_dir, collection, embed_model = _get_paths_and_names(cfg)
    cache_key = (chroma_dir, collection, embed_model)
    if cache_key in _VECTORSTORE_CACHE:
        LOG.debug("Reusing cached vectorstore (%s / %s)", collection, embed_model)
        return _VECTORSTORE_CACHE[cache_key]

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    is_bge = "bge" in embed_model.lower()
    encode_kwargs: dict = {"normalize_embeddings": True}
    model_kwargs: dict = {"device": device}

    if is_bge:
        _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

        class _BGEEmbeddings(HuggingFaceEmbeddings):
            def embed_query(self, text: str) -> list[float]:
                return super().embed_query(_BGE_QUERY_PREFIX + text)

        embeddings: HuggingFaceEmbeddings = _BGEEmbeddings(
            model_name=embed_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    vectorstore = Chroma(
        collection_name=collection,
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
    )
    _VECTORSTORE_CACHE[cache_key] = vectorstore
    LOG.info("Vectorstore loaded and cached (%s / %s)", collection, embed_model)
    return vectorstore


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def _docs_to_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for d in docs:
        title = (d.metadata or {}).get("Title") or (d.metadata or {}).get("title") or "Unknown"
        parts.append(f"[{title}]\n{d.page_content}")
    return "\n\n".join(parts).strip()


def _docs_to_chunks(docs: list[Document]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in docs:
        md = d.metadata or {}
        chunk_id = md.get("chunk_id") or md.get("id") or ""
        title = md.get("Title") or md.get("title") or "Unknown"
        out.append(
            {
                "chunk_id": str(chunk_id or ""),
                "title": str(title),
                "metadata": dict(md),
                "text": d.page_content,
            }
        )
    return out


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for ch in chunks:
        cid = ch.get("chunk_id") or ""
        title = ch.get("title") or "Unknown"
        txt = ch.get("text") or ""
        parts.append(f"[CHUNK {cid} | {title}]\n{txt}")
    return "\n\n".join(parts).strip()


def _select_diverse_stories(
    docs: list[Document],
    *,
    n_stories: int = 3,
    chunks_per_story: int = 2,
) -> list[Document]:
    """Pick chunks from up to `n_stories` different titles so one book does not dominate."""
    by_title: dict[str, list[Document]] = {}
    for d in docs:
        md = d.metadata or {}
        title = str(md.get("Title") or md.get("title") or "Unknown").strip() or "Unknown"
        by_title.setdefault(title, []).append(d)

    picked: list[Document] = []
    picked_titles: set[str] = set()
    for title, group in by_title.items():
        if len(picked_titles) >= n_stories:
            break
        picked_titles.add(title)
        picked.extend(group[: max(1, chunks_per_story)])

    target = max(1, n_stories) * max(1, chunks_per_story)
    if len(picked) < min(target, len(docs)):
        seen = {id(d) for d in picked}
        for d in docs:
            if id(d) in seen:
                continue
            picked.append(d)
            if len(picked) >= target:
                break
    return picked


# ---------------------------------------------------------------------------
# BM25 hybrid fusion (RRF)
# ---------------------------------------------------------------------------

def _bm25_rank_docs(query: str, docs: list[Document]) -> list[Document]:
    """Re-rank docs by BM25 score over the candidate pool (requires rank-bm25)."""
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:
        LOG.warning(
            "rank-bm25 not installed — BM25 hybrid disabled. "
            "Run: pip install rank-bm25"
        )
        return docs

    tokenized_corpus = [d.page_content.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked]


def _rrf_fuse(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    *,
    bm25_weight: float = 0.3,
    rrf_k: int = 60,
) -> list[Document]:
    """Reciprocal Rank Fusion of dense-vector and BM25 ranked lists.

    Score = (1 - bm25_weight) / (dense_rank + rrf_k)
            + bm25_weight       / (bm25_rank  + rrf_k)

    A bm25_weight of 0.3 gives the dense signal 70 % of the influence.
    """
    all_docs: dict[str, Document] = {}
    for d in dense_docs + bm25_docs:
        key = d.page_content
        if key not in all_docs:
            all_docs[key] = d

    dense_rank: dict[str, int] = {d.page_content: i + 1 for i, d in enumerate(dense_docs)}
    bm25_rank:  dict[str, int] = {d.page_content: i + 1 for i, d in enumerate(bm25_docs)}

    dense_w = 1.0 - bm25_weight
    n = len(all_docs)
    scores: dict[str, float] = {}
    for key in all_docs:
        dr = dense_rank.get(key, n + rrf_k)
        br = bm25_rank.get(key, n + rrf_k)
        scores[key] = dense_w / (dr + rrf_k) + bm25_weight / (br + rrf_k)

    ranked = sorted(all_docs.keys(), key=lambda k: scores[k], reverse=True)
    return [all_docs[k] for k in ranked]


# ---------------------------------------------------------------------------
# Main retrieval entrypoint
# ---------------------------------------------------------------------------

def retrieve_docs(
    query: str,
    cfg: dict[str, Any],
    *,
    n_stories: int = 3,
    chunks_per_story: int = 2,
    k_boost: float = 1.0,
    use_reranker: Optional[bool] = None,
    filter_metadata: Optional[dict[str, Any]] = None,
) -> list[Document]:
    """Step 1: Chroma vector search, optionally fused with BM25 (hybrid).

    k_boost         — widen retrieval pool (agentic re-retrieve uses > 1.0).
    use_reranker    — override Reranker_enabled in setup.yaml when provided.
    filter_metadata — Chroma ``where`` filter, e.g. {"series_id": "series_01"}.
                      Enables targeted retrieval within a series or story type.
    """
    vectorstore = _build_vectorstore(cfg)
    base_k = int(cfg.get("Story_generation_n_results") or 3)
    k = max(base_k, n_stories * max(1, chunks_per_story) * 4)
    k = int(round(k * max(1.0, float(k_boost))))

    search_kwargs: dict[str, Any] = {"k": k}
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata

    docs = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(query)

    # Hybrid BM25 + dense fusion (RRF) when enabled.
    hybrid_on = str(cfg.get("Hybrid_search_enabled") or "").strip().lower() not in ("false", "0", "no", "")
    if hybrid_on and docs:
        bm25_weight = float(cfg.get("Hybrid_bm25_weight") or 0.3)
        bm25_ranked = _bm25_rank_docs(query, docs)
        docs = _rrf_fuse(docs, bm25_ranked, bm25_weight=bm25_weight)
        LOG.debug("Hybrid BM25+dense fusion applied (bm25_weight=%.2f)", bm25_weight)

    docs = _select_diverse_stories(docs, n_stories=n_stories, chunks_per_story=chunks_per_story)

    # Optional cross-encoder rerank (see Reranker_enabled / Reranker_model in setup.yaml).
    if use_reranker is None:
        reranker_on = str(cfg.get("Reranker_enabled") or "").strip().lower() not in ("false", "0", "no", "")
    else:
        reranker_on = bool(use_reranker)
    if reranker_on:
        reranker_model_id = str(cfg.get("Reranker_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        top_n = int(cfg.get("Story_generation_rerank_top_n") or 6)
        docs = _rerank_docs(query, docs, reranker_model_id=reranker_model_id, top_n=top_n)

    return docs
