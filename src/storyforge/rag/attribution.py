"""
Grounded facts parsing and light hallucination checks.

Step 2 returns JSON facts; this module parses them, formats bullets for the
generation prompt, and optionally flags capitalized names not in the fact list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GroundedFact:
    fact: str
    type: str = "fact"
    source_chunk_ids: tuple[str, ...] = ()
    quote: str = ""


@dataclass(frozen=True)
class ParsedFacts:
    facts: tuple[GroundedFact, ...]
    raw: dict[str, Any]


def _strip_json_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def repair_json(text: str) -> str:
    t = _strip_json_fences(text)
    # Keep only the outermost { ... } object.
    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last != -1 and last > first:
        t = t[first : last + 1]
    # Fix trailing commas before } or ]
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t.strip()


def parse_grounded_facts_json(text: str) -> ParsedFacts:
    repaired = repair_json(text)
    data = json.loads(repaired) if repaired else {}
    facts_raw = data.get("facts") or []

    facts: list[GroundedFact] = []
    if isinstance(facts_raw, list):
        for f in facts_raw:
            if not isinstance(f, dict):
                continue
            fact_text = str(f.get("fact") or "").strip()
            src = f.get("source_chunk_ids") or f.get("sources") or []
            if isinstance(src, str):
                src_list = [src]
            elif isinstance(src, list):
                src_list = [str(s).strip() for s in src if str(s).strip()]
            else:
                src_list = []
            if not fact_text or not src_list:
                continue  # skip facts with no source chunk
            facts.append(
                GroundedFact(
                    fact=fact_text,
                    type=str(f.get("type") or "fact"),
                    source_chunk_ids=tuple(src_list),
                    quote=str(f.get("quote") or "").strip(),
                )
            )
    return ParsedFacts(facts=tuple(facts), raw=data if isinstance(data, dict) else {})


def extract_named_entities_heuristic(text: str) -> set[str]:
    """Rough set of capitalized words treated as proper nouns (for attribution check)."""
    t = (text or "").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return set()

    stop = {
        "The",
        "A",
        "An",
        "And",
        "But",
        "Or",
        "In",
        "On",
        "At",
        "To",
        "For",
        "Of",
        "With",
        "As",
        "When",
        "Then",
        "After",
        "Before",
        "Once",
        "Section",
        "SECTION",
        "WHO",
        "WHERE",
        "WHEN",
        "WHAT",
        "HOW",
        "WHY",
        "OUTCOME",
    }

    phrases = set(re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", t))
    tokens = set(re.findall(r"\b[A-Z][a-z]+\b", t))
    out = set()
    for x in phrases | tokens:
        if x in stop:
            continue
        if len(x) <= 2:
            continue
        out.add(x)
    return out


def attribution_violations(
    *,
    story: str,
    facts: tuple[GroundedFact, ...],
) -> set[str]:
    allowed_text = " ".join([f.fact + " " + (f.quote or "") for f in facts]).strip()
    allowed = extract_named_entities_heuristic(allowed_text)
    found = extract_named_entities_heuristic(story)
    return {e for e in found if e not in allowed}  # names in story but not in facts


def build_debug_attribution_stub(
    *,
    story: str,
    facts: tuple[GroundedFact, ...],
) -> dict[str, Any]:
    """Debug payload: each sentence tagged with all source chunk ids from facts."""
    chunk_ids = sorted({cid for f in facts for cid in f.source_chunk_ids})
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (story or "").strip()) if s.strip()]
    return {"sentences": [{"text": s, "source_chunk_ids": chunk_ids} for s in sentences]}


def format_facts_for_prompt(parsed: "ParsedFacts") -> str:
    """
    Numbered bullet list for the generation prompt (empty if no facts).

    Example: 1. [who] Alana is a soldier  (source: chunk_1)
    """
    if not parsed.facts:
        return ""
    lines: list[str] = []
    for i, f in enumerate(parsed.facts, start=1):
        src = ", ".join(f.source_chunk_ids) if f.source_chunk_ids else "N/A"
        quote = f' ("{f.quote}")' if f.quote else ""
        lines.append(f"{i}. [{f.type}] {f.fact}{quote}  (source: {src})")
    return "\n".join(lines)
