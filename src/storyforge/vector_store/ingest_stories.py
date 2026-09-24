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
    """Embed a list of texts. Returns None when the model is unavailable.

    ``is_bge`` is accepted for call-site compatibility but no longer changes
    encoding: BGE's documented recipe (BAAI/bge-base-en-v1.5) only prefixes
    the *query* side ("Represent this sentence for searching relevant
    passages: ", applied at search time -- see
    storyforge.vector_store.embeddings.QUERY_PREFIX). Passages/documents are
    embedded with no instruction prefix. This function previously also
    prefixed passages with "Represent this passage for retrieval: ", which
    deviated from that recipe -- fixed here. Any collection ingested before
    this fix has vectors baked with the old (wrong) prefix and must be
    re-ingested; see docs/DATA_PREP.md.
    """
    if model is None:
        return None
    try:
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
    # How each .txt was ingested (see _story_json_plan):
    from_story_json: int = 0      # reviewed story_json chunks + metadata used
    stale_story_json: int = 0     # story_json exists but raw_text != .txt -> .txt chunks, story metadata only
    without_story_json: int = 0   # no story_json record -> .txt chunks, empty Author/Summary


def _read_text_keep_newlines(path: Path, *, errors: str = "strict") -> str:
    """Read UTF-8 text without newline translation (CRLF stays CRLF).

    Path.read_text(newline=...) only exists on Python 3.13+; open(newline="")
    gives the same result on the 3.10+ versions requirements.txt supports.
    """
    with open(path, encoding="utf-8", errors=errors, newline="") as fh:
        return fh.read()


def _norm_text(t: str) -> str:
    # Normalize CRLF and lone CR (Windows text-mode round-trips can leave \r).
    return (t or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _load_story_json(records_dir: Optional[Path], title: str) -> Optional[dict]:
    if records_dir is None:
        return None
    fp = records_dir / f"{title}.json"
    if not fp.is_file():
        return None
    try:
        import json

        rec = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:  # malformed JSON should not kill a whole ingest
        LOG.warning("Ignoring unreadable story_json %s: %s", fp, e)
        return None
    return rec if isinstance(rec, dict) else None


def _story_json_plan(rec: Optional[dict], txt: str, title: str) -> tuple[str, list[dict]]:
    """Decide how to ingest one story. Returns (mode, reviewed_chunks).

    mode = "story_json": the record's raw_text matches the .txt, so its
    (possibly hand-fixed) chunks, chunk_ids and section tags are what gets
    embedded -- matching docs/DATA_PREP.md ("chunks[].text: this is what gets
    embedded"). mode = "stale": record exists but the .txt changed since it was
    built, so chunks come from the .txt and only story-level metadata is taken.
    mode = "none": no record.
    """
    if rec is None:
        return "none", []
    raw = rec.get("raw_text")
    if raw is not None and _norm_text(str(raw)) != _norm_text(txt):
        return "stale", []
    out: list[dict] = []
    for i, ch in enumerate(rec.get("chunks") or [], start=1):
        if not isinstance(ch, dict):
            continue
        text = str(ch.get("text") or "").strip()
        if not text:
            continue  # DATA_PREP checklist: no empty chunks
        out.append(
            {
                "chunk_id": str(ch.get("chunk_id") or f"{title}_chunk_{i}"),
                "text": text,
                "section": str(ch.get("section") or ""),
            }
        )
    return ("story_json", out) if out else ("stale", [])


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
    records_dir: str | Path | None = None,
    use_story_json: bool = True,
) -> IngestResult:
    """Read Story_input/*.txt, chunk, embed, upsert into Chroma.

    When ``data/story_json/<Title>.json`` exists (``records_dir`` overrides the
    location) its Author / Summary / Display_title / Is_series are written as
    metadata, and -- if its raw_text still matches the .txt -- its reviewed
    chunks and section tags are embedded instead of re-chunking the .txt.
    Previously this path (used by reset_and_ingest.py) hardcoded Author="" /
    Summary="" and ignored every manual fix made in story_json.
    """
    try:
        from storyforge.data.records_to_manifest import _as_chroma_metadata, story_level_metadata

        cfg = load_config()
        root = Path(base_path or cfg.get("BASE_PATH") or Path(__file__).resolve().parents[1])
        # Default must match story_records.py's, or an omitted Story_input
        # silently points ingest at a directory that doesn't exist.
        stories_path = Path(stories_dir) if stories_dir else root / (cfg.get("Story_input") or "data/stories")
        stories_path = stories_path.resolve()
        rec_dir: Optional[Path] = None
        if use_story_json:
            rec_dir = Path(records_dir) if records_dir else root / "data" / "story_json"
            rec_dir = rec_dir.resolve() if rec_dir.is_dir() else None
        counts = {"story_json": 0, "stale": 0, "none": 0}

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
        embed_model_name = str(cfg.get("Vector_store_model") or "BAAI/bge-base-en-v1.5")
        is_bge = "bge" in embed_model_name.lower()
        embed_model = _get_embed_model(embed_model_name)
        if embed_model is not None:
            LOG.info("Ingest using embedding model: %s", embed_model_name)
        else:
            LOG.warning("No embedding model loaded — chromadb will use its built-in default.")

        for f in files:
            try:
                # newline="" so Windows does not translate CRLF before we normalize
                # for story_json staleness checks.
                text = _read_text_keep_newlines(f)
            except UnicodeDecodeError:
                text = _read_text_keep_newlines(f, errors="replace")

            title = f.stem
            rec = _load_story_json(rec_dir, title)
            mode, reviewed = _story_json_plan(rec, text, title)
            if mode == "story_json":
                chunk_rows = reviewed
            else:
                chunk_rows = [
                    {"chunk_id": f"{title}_chunk_{i}", "text": ch, "section": ""}
                    for i, ch in enumerate(
                        _chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars), start=1
                    )
                ]
                if mode == "stale":
                    LOG.warning(
                        "story_json for %s is stale (raw_text differs from the .txt): ingesting .txt chunks "
                        "with story-level metadata only (no section tags / chunk edits). Rebuild it with "
                        "scripts/prepare_story_records.py --overwrite --only \"%s\" and re-review.",
                        title, title,
                    )
            if not chunk_rows:
                LOG.warning("Skipping empty story file: %s", f.name)
                continue
            counts[mode] += 1

            story_md = story_level_metadata(rec or {}, title)
            ids: list[str] = []
            metadatas: list[dict] = []
            documents: list[str] = []
            for row in chunk_rows:
                ids.append(row["chunk_id"])
                metadatas.append(
                    _as_chroma_metadata(
                        {
                            **story_md,
                            "chunk_id": row["chunk_id"],  # used by grounded-facts attribution
                            "section": row["section"],
                        }
                    )
                )
                documents.append(row["text"])

            embeddings = _embed_chunks(embed_model, documents, is_bge=is_bge)
            if embeddings is not None:
                collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            else:
                # Chroma default embeddings — only OK if Vector_store_model is all-MiniLM-L6-v2.
                collection.upsert(ids=ids, metadatas=metadatas, documents=documents)
            chunks_written += len(ids)

        LOG.info(
            "Ingest sources: %d from story_json, %d stale story_json (.txt chunks), %d without story_json.",
            counts["story_json"], counts["stale"], counts["none"],
        )
        return IngestResult(
            success=True,
            files_seen=files_seen,
            chunks_written=chunks_written,
            collection_name=collection.name,
            error=None,
            from_story_json=counts["story_json"],
            stale_story_json=counts["stale"],
            without_story_json=counts["none"],
        )
    except Exception as e:
        LOG.exception("ingest_stories_dir failed")
        return IngestResult(success=False, error=str(e))
