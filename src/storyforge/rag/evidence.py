"""Evidence for a story: retrieve chunks, extract grounded facts, and widen the search when facts are thin.

One interface for the agentic loops and the 3-step path (ADR-0011 Step 1). Widening merges facts and never
replaces them. Stage functions are imported at call time so tests can patch them on their source modules.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from storyforge.config.config import num

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Evidence:
    query: str  # the query that produced the latest chunks
    facts: Any  # ParsedFacts
    raw: str  # raw extraction text of the latest call
    chunks: tuple
    context: str


def reformulate_query(query: str, parsed) -> str:
    """
    Append a few character/place names from grounded facts to the query (no extra LLM).

    Used when RE_RETRIEVE runs so the next search stays on the same story.
    """
    terms: list[str] = []
    seen: set[str] = set()
    facts = getattr(parsed, "facts", ()) or ()
    for f in facts:
        if getattr(f, "type", "").lower() not in {"who", "where", "setting", "event"}:
            continue
        for token in re.findall(r"\b[A-Z][a-z]{2,}\b", getattr(f, "fact", "") or ""):
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(token)
        if len(terms) >= 5:
            break
    if not terms:
        return query
    return f"{query} {' '.join(terms[:5])}".strip()


class EvidenceSearch:
    """Holds the retrieval window (how many stories, how wide) so every gather and widen shares it."""

    def __init__(
        self, cfg: dict[str, Any], *, story_type: Any = None, n_stories: int = 3, chunks_per_story: int = 2
    ) -> None:
        self.cfg = cfg
        self.story_type = story_type
        self.n_stories = n_stories
        self.chunks_per_story = chunks_per_story
        self.k_boost = 1.0

    def gather(self, query: str, previous: Optional[Evidence] = None) -> Evidence:
        from storyforge.rag.attribution import merge_facts
        from storyforge.rag.extraction import extract_grounded_facts
        from storyforge.rag.retrieval import _docs_to_chunks, _docs_to_context, retrieve_docs

        docs = retrieve_docs(
            query, self.cfg, n_stories=self.n_stories, chunks_per_story=self.chunks_per_story,
            k_boost=self.k_boost, story_type=self.story_type,
        )
        chunks = _docs_to_chunks(docs)
        raw, facts = extract_grounded_facts(query, chunks, self.cfg)
        if previous is not None:
            facts = merge_facts(previous.facts, facts)  # widening adds evidence, never replaces it
        return Evidence(query=query, facts=facts, raw=raw, chunks=tuple(chunks), context=_docs_to_context(docs))

    def widen(self) -> None:
        """Search wider on the next gather (more candidates, more stories)."""
        self.k_boost *= float(self.cfg.get("Agentic_loop_reretrieve_k_boost") or 2.0)
        self.n_stories = max(self.n_stories, int(self.cfg.get("Agentic_loop_reretrieve_n_results") or 5))

    def preflight(self, query: str, evidence: Evidence) -> Evidence:
        """Fix thin evidence BEFORE any story is written.

        Tier 1 widens the search; tier 2 also reformulates the query with names from the facts found so far.
        """
        gate_min = int(num(self.cfg, "Agentic_loop_min_facts", 3))
        for tier in range(1, int(self.cfg.get("Agentic_loop_preflight_tiers", 2) or 0) + 1):
            if len(evidence.facts.facts) >= gate_min:
                break
            LOG.warning(
                "Pre-flight gate: %d facts < %d; widening retrieval (tier %d).", len(evidence.facts.facts), gate_min, tier
            )
            self.widen()
            q = reformulate_query(query, evidence.facts) if tier >= 2 else evidence.query
            evidence = self.gather(q, evidence)
        return evidence
