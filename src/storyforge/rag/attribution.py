"""
Grounded facts parsing and light hallucination checks.

Step 2 returns JSON facts; this module parses them, formats bullets for the
generation prompt, and optionally flags capitalized names not in the fact list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GroundedFact",
    "ParsedFacts",
    "repair_json",
    "parse_grounded_facts_json",
    "salvage_grounded_facts_json",
    "extract_named_entities_heuristic",
    "attribution_violations",
    "build_debug_attribution_stub",
    "format_facts_for_prompt",
]



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


# ---------------------------------------------------------------------------
# Lenient parsing for LOCAL-model grounded-facts output
#
# The HF path gets token-level JSON enforcement (response_format). Local
# Ollama / vLLM / Transformers output is much messier: markdown fences, prose
# before/after the object, a bare list instead of {"facts": [...]}, trailing
# commas, <think> blocks, singular key variants, and -- most often -- a JSON
# object cut off mid-fact by num_predict. json.loads() rejects all of that and
# the old code silently returned 0 facts. salvage_grounded_facts_json() keeps
# every fact object that is itself complete and well-formed, so a truncated
# answer still yields the facts written before the cut.
# ---------------------------------------------------------------------------

_SOURCE_KEYS = ("source_chunk_ids", "sources", "source_chunk_id", "chunk_ids", "chunk_id", "source")


def _strip_think_blocks(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    # An unclosed <think> means the model never left its reasoning (usually cut
    # off by the token budget) -- nothing after it is answer text.
    if re.search(r"<think>", t, flags=re.IGNORECASE):
        t = re.split(r"<think>", t, flags=re.IGNORECASE)[0]
    return t.strip()


def _norm_chunk_id(cid: str) -> str:
    return re.sub(r"\s+", "", str(cid or "")).strip().strip("[]").lower()


def _resolve_chunk_id(cid: str, known: dict[str, str]) -> str | None:
    """Map a model-cited id onto a real retrieved chunk id (or None).

    Accepts exact / case-insensitive matches, and an unambiguous suffix match
    (e.g. the model writes "chunk_3" for "Lovecraft__Cool_Air_chunk_3").
    """
    n = _norm_chunk_id(cid)
    if not n:
        return None
    if n in known:
        return known[n]
    hits = [orig for k, orig in known.items() if k.endswith("_" + n)]
    return hits[0] if len(hits) == 1 else None


def _lenient_fact(f: dict[str, Any], known: dict[str, str] | None) -> tuple[GroundedFact | None, str]:
    fact_text = str(f.get("fact") or f.get("text") or f.get("statement") or "").strip()
    if not fact_text:
        return None, "no_fact_text"
    src: Any = []
    for key in _SOURCE_KEYS:
        if f.get(key):
            src = f.get(key)
            break
    if isinstance(src, (str, int)):
        src_list = [str(src).strip()]
    elif isinstance(src, list):
        src_list = [str(x).strip() for x in src if str(x).strip()]
    else:
        src_list = []
    if known:
        resolved: list[str] = []
        for cid in src_list:
            r = _resolve_chunk_id(cid, known)
            if r and r not in resolved:
                resolved.append(r)
        src_list = resolved
    if not src_list:
        return None, "no_valid_source"
    return (
        GroundedFact(
            fact=fact_text,
            type=str(f.get("type") or "fact"),
            source_chunk_ids=tuple(src_list),
            quote=str(f.get("quote") or "").strip(),
        ),
        "",
    )


def _scan_fact_objects(text: str) -> list[dict[str, Any]]:
    """Decode every complete {...} object after the "facts" key (tolerates truncation)."""
    decoder = json.JSONDecoder()
    m = re.search(r'"facts"\s*:\s*\[', text)
    i = m.end() if m else 0
    out: list[dict[str, Any]] = []
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            if "fact" in obj or "text" in obj or "statement" in obj:
                out.append(obj)
            elif isinstance(obj.get("facts"), list):
                out.extend(x for x in obj["facts"] if isinstance(x, dict))
        i = end
    return out


def salvage_grounded_facts_json(
    text: str,
    *,
    known_chunk_ids: "list[str] | tuple[str, ...] | None" = None,
) -> tuple[ParsedFacts, dict[str, Any]]:
    """Best-effort parse of messy local-model facts output.

    Returns ``(ParsedFacts, diagnostics)``. ``diagnostics`` always has
    ``status`` (``ok`` / ``salvaged`` / ``empty_response`` / ``no_json`` /
    ``no_facts``), ``raw_fact_objects`` and ``dropped`` (reason -> count), so
    callers can log *why* extraction yielded nothing instead of failing silently.

    When ``known_chunk_ids`` is given, every cited id must resolve to a real
    retrieved chunk (see _resolve_chunk_id); facts citing none are dropped --
    a fact the model can't tie to a retrieved chunk isn't grounded.
    """
    diag: dict[str, Any] = {"status": "ok", "raw_fact_objects": 0, "dropped": {}}
    t = _strip_think_blocks(text)
    t = _strip_json_fences(t)
    if not t:
        diag["status"] = "empty_response"
        return ParsedFacts(facts=(), raw={}), diag

    known = {_norm_chunk_id(c): str(c) for c in (known_chunk_ids or []) if _norm_chunk_id(c)} or None

    data: Any = None
    first_obj, first_arr = t.find("{"), t.find("[")
    candidates: list[str] = []
    if first_arr != -1 and (first_obj == -1 or first_arr < first_obj):
        last = t.rfind("]")
        if last > first_arr:
            candidates.append(t[first_arr : last + 1])
    candidates.append(repair_json(t))
    for cand in candidates:
        try:
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
            break
        except ValueError:
            continue

    if isinstance(data, list):
        facts_raw = [x for x in data if isinstance(x, dict)]
        raw = {"facts": facts_raw}
    elif isinstance(data, dict):
        facts_raw = data.get("facts")
        if isinstance(facts_raw, dict):
            facts_raw = [facts_raw]
        facts_raw = [x for x in (facts_raw or []) if isinstance(x, dict)]
        raw = data
    else:
        facts_raw = _scan_fact_objects(re.sub(r",\s*([}\]])", r"\1", t))
        raw = {"facts": facts_raw}
        diag["status"] = "salvaged" if facts_raw else "no_json"

    diag["raw_fact_objects"] = len(facts_raw)
    facts: list[GroundedFact] = []
    for f in facts_raw:
        gf, why = _lenient_fact(f, known)
        if gf is None:
            diag["dropped"][why] = diag["dropped"].get(why, 0) + 1
            continue
        facts.append(gf)
    if not facts and diag["status"] in ("ok", "salvaged"):
        diag["status"] = "no_facts"
    return ParsedFacts(facts=tuple(facts), raw=raw), diag


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
