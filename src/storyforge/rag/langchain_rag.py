"""
Three-step RAG for story generation — orchestrator.

This module was split in v2; heavy logic now lives in focused sub-modules:

  rag/retrieval.py   — Step 1: Chroma vector + hybrid BM25 retrieval
  rag/extraction.py  — Step 2: grounded facts extraction (HF API / local fallback)
  rag/generation.py  — Step 3: 5-section story generation / refinement

All original public symbols are re-exported here so that agentic_loop.py
and any other callers continue to work without source changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from storyforge.config.config import load_config
from storyforge.rag.attribution import build_debug_attribution_stub
from storyforge.rag.extraction import extract_grounded_facts
from storyforge.rag.generation import _sections_below_min_sentences, generate_from_facts
from storyforge.rag.retrieval import _docs_to_chunks, _docs_to_context, retrieve_docs

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAG3StepResult:
    content: str
    retrieval_context: str
    grounded_extraction: str
    grounded_facts: tuple = ()
    retrieval_chunks: tuple = ()
    debug_attribution: dict | None = None


def generate_story_3step_langchain(
    query: str,
    *,
    cfg: Optional[dict[str, Any]] = None,
    mode: Any = None,
    n_stories: int = 3,
    chunks_per_story: int = 2,
    show_progress: bool = True,
    debug: bool = False,
) -> RAG3StepResult:
    cfg = cfg or load_config()

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=3, desc="RAG 3-step", unit="step")
        except Exception:
            pbar = None

    docs = retrieve_docs(query, cfg, n_stories=n_stories, chunks_per_story=chunks_per_story)
    chunks = _docs_to_chunks(docs)
    retrieval_context = _docs_to_context(docs)
    if pbar:
        pbar.update(1)

    grounded_raw, parsed = extract_grounded_facts(query, chunks, cfg)
    if pbar:
        pbar.update(1)

    story = generate_from_facts(query, parsed, grounded_raw, cfg, mode=mode)
    min_sentences = int(cfg.get("Min_sentences_per_section") or 3)
    short_sections = _sections_below_min_sentences(story, min_sentences=min_sentences)
    if short_sections:
        LOG.info(
            "Section sentence guard triggered (min=%d). Short sections: %s.",
            min_sentences, short_sections,
        )
        feedback = (
            f"Section format fix required: each section must have at least {min_sentences} complete sentences. "
            f"Sections below minimum: {str(sorted(short_sections.items()))}. "
            "Expand only the short sections while preserving grounded facts."
        )
        refine_max = int(cfg.get("Single_pass_refine_max_tokens") or 0) or None
        story = generate_from_facts(
            query, parsed, grounded_raw, cfg,
            mode=mode, refine_feedback=feedback, prior_draft=story, max_new_tokens=refine_max,
        )
    if pbar:
        pbar.update(1)
        pbar.close()

    facts = parsed.facts
    debug_payload = None
    if debug:
        debug_payload = build_debug_attribution_stub(story=story, facts=facts)

    return RAG3StepResult(
        content=story,
        retrieval_context=retrieval_context,
        grounded_extraction=grounded_raw,
        grounded_facts=tuple([f.__dict__ for f in facts]),
        retrieval_chunks=tuple(chunks),
        debug_attribution=debug_payload,
    )


# ---------------------------------------------------------------------------
# Backward-compat re-exports (agentic_loop.py imports these by name)
# ---------------------------------------------------------------------------
__all__ = [
    "RAG3StepResult",
    "generate_story_3step_langchain",
    "_docs_to_chunks",
    "_docs_to_context",
    "extract_grounded_facts",
    "generate_from_facts",
    "retrieve_docs",
]
