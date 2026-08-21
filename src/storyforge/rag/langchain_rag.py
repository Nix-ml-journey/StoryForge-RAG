"""
Three-step RAG for story generation — orchestrator.

Heavy logic lives in focused sub-modules:

  rag/retrieval.py   — Step 1: Chroma vector + hybrid BM25 retrieval
  rag/extraction.py  — Step 2: grounded facts extraction (HF API / local fallback)
  rag/generation.py  — Step 3: 5-section story generation / refinement

Public entrypoint: ``generate_story_3step_langchain`` / ``RAG3StepResult``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from storyforge.config.config import load_config
from storyforge.rag.attribution import build_debug_attribution_stub
from storyforge.rag.extraction import extract_grounded_facts
from storyforge.rag.generation import _sections_below_min_sentences, generate_from_facts
from storyforge.rag.length_profile import resolve_length_profile
from storyforge.rag.retrieval import _docs_to_chunks, _docs_to_context, retrieve_docs

LOG = logging.getLogger(__name__)

__all__ = [
    "RAG3StepResult",
    "generate_story_3step_langchain",
]


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
    length: Any = None,
    n_stories: int = 3,
    chunks_per_story: int = 2,
    show_progress: bool = True,
    debug: bool = False,
) -> RAG3StepResult:
    cfg = cfg or load_config()
    profile = resolve_length_profile(cfg, length=length, mode=mode)

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

    story = generate_from_facts(query, parsed, grounded_raw, cfg, mode=mode, profile=profile)
    min_sentences = profile.min_sentences_per_section
    short_sections = _sections_below_min_sentences(story, min_sentences=min_sentences)
    word_count = len(story.split())
    if short_sections or word_count < profile.min_words:
        LOG.info(
            "Length guard triggered (target=%s, min_words=%d, min_sentences=%d). "
            "Draft had %d words; short sections: %s.",
            profile.name, profile.min_words, min_sentences, word_count, short_sections,
        )
        feedback_parts = [
            f"Length fix required: the story must reach about {profile.target_words} words "
            f"with roughly {profile.words_per_section} words per section."
        ]
        if short_sections:
            feedback_parts.append(
                f"Each section needs at least {min_sentences} complete sentences. "
                f"Sections below the minimum (section: sentence count): {sorted(short_sections.items())}."
            )
        if word_count < profile.min_words:
            feedback_parts.append(
                f"The draft is {word_count} words, under the {profile.min_words}-word minimum."
            )
        feedback_parts.append(
            "Expand the thin sections with grounded detail and dialogue; do not trim finished sections."
        )
        # Refine has to re-emit the whole story, so it needs at least the draft budget.
        refine_max = max(
            int(cfg.get("Single_pass_refine_max_tokens") or 0),
            profile.max_new_tokens,
        )
        story = generate_from_facts(
            query, parsed, grounded_raw, cfg,
            mode=mode, profile=profile, refine_feedback=" ".join(feedback_parts),
            prior_draft=story, max_new_tokens=refine_max,
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
