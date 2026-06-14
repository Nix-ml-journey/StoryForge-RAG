"""
Agentic story loop: retrieve → generate → evaluate → decide → repeat.

Each iteration may ACCEPT the draft, REFINE it (same retrieval, better prompt),
or RE_RETRIEVE (wider search + new facts). Stops when quality and completeness
pass, or after Agentic_loop_max_iterations.

Functions like completeness_report and decide_action are pure (no API calls)
so they can be unit-tested without LangChain or GPU dependencies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from storyforge.config.config import load_config
from storyforge.rag.generative_ai import Gen_mode, StoryType

# langchain_rag and evaluation are imported inside run_agentic_story_loop only,
# so tests can import decide_action without heavy dependencies.

LOG = logging.getLogger(__name__)

ACCEPT = "accept"
RE_RETRIEVE = "re_retrieve"
REFINE = "refine"

# Evaluation JSON keys that are not numeric criteria.
_NON_CRITERION_KEYS = {
    "conclusion",
    "suggestions",
    "average_score",
    "scores",
    "summary",
    "metadata",
    "success",
    "provider",
    "model",
}

_TERMINAL_CHARS = '.!?"\u201d\u2019\')'
_EXPECTED_SECTIONS = 5

_SECTION_HEADER_RE = re.compile(r"^\[SECTION\s+(\d+).*?\]\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class CompletenessReport:
    ok: bool
    word_count: int
    missing_sections: tuple[int, ...]
    ends_clean: bool
    reasons: tuple[str, ...] = ()


def _split_section_bodies(text: str) -> dict[int, str]:
    matches = list(_SECTION_HEADER_RE.finditer(text or ""))
    if not matches:
        return {}
    sections: dict[int, str] = {}
    for i, m in enumerate(matches):
        try:
            section_idx = int(m.group(1))
        except Exception:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text or "")
        sections[section_idx] = (text or "")[start:end].strip()
    return sections


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[^.!?]+[.!?]", (text or "").strip()))


def completeness_report(
    story: str,
    *,
    min_words: int = 250,
    expected_sections: int = _EXPECTED_SECTIONS,
    min_sentences_per_section: int = 0,
) -> CompletenessReport:
    """
    Rule-based check: all [SECTION n] headers present, ends with . ! or ?,
    and at least min_words. No LLM involved.
    """
    text = (story or "").strip()
    word_count = len(text.split())

    found_sections = {int(n) for n in re.findall(r"\[SECTION\s+(\d+)", text, flags=re.IGNORECASE)}
    missing = tuple(i for i in range(1, expected_sections + 1) if i not in found_sections)
    short_sections: list[str] = []
    if min_sentences_per_section > 0 and not missing:
        bodies = _split_section_bodies(text)
        for i in range(1, expected_sections + 1):
            count = _sentence_count(bodies.get(i, ""))
            if count < min_sentences_per_section:
                short_sections.append(f"SECTION {i} has {count} sentence(s) < {min_sentences_per_section}")

    ends_clean = bool(text) and text[-1] in _TERMINAL_CHARS

    reasons: list[str] = []
    if missing:
        reasons.append(f"missing sections: {list(missing)}")
    if not ends_clean:
        reasons.append("does not end on terminal punctuation")
    if word_count < min_words:
        reasons.append(f"too short ({word_count} < {min_words} words)")
    if short_sections:
        reasons.append("sections below sentence minimum: " + "; ".join(short_sections))

    ok = not missing and ends_clean and word_count >= min_words and not short_sections
    return CompletenessReport(
        ok=ok,
        word_count=word_count,
        missing_sections=missing,
        ends_clean=ends_clean,
        reasons=tuple(reasons),
    )


def _score_of(value: Any) -> Optional[float]:
    if isinstance(value, dict) and "score" in value:
        try:
            return float(value["score"])
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def criterion_score(eval_data: Optional[dict[str, Any]], name: str) -> Optional[float]:
    """One rubric score, e.g. criterion_score(data, 'faithfulness')."""
    if not eval_data:
        return None
    source = eval_data.get("scores") if isinstance(eval_data.get("scores"), dict) and eval_data.get("scores") else eval_data
    return _score_of(source.get(name)) if isinstance(source, dict) else None


def average_score(eval_data: Optional[dict[str, Any]]) -> float:
    """Mean of all numeric criteria in the evaluation payload (0.0 if none)."""
    if not eval_data:
        return 0.0

    explicit = eval_data.get("average_score")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool) and explicit:
        return float(explicit)

    if isinstance(eval_data.get("scores"), dict) and eval_data.get("scores"):
        source = eval_data["scores"]
    else:
        source = eval_data

    scores: list[float] = []
    for key, value in source.items():
        if key in _NON_CRITERION_KEYS:
            continue
        s = _score_of(value)
        if s is not None:
            scores.append(s)
    return round(sum(scores) / len(scores), 2) if scores else 0.0


@dataclass(frozen=True)
class Decision:
    action: str
    reasons: tuple[str, ...]
    avg: float
    faithfulness: Optional[float]


def decide_action(
    eval_data: Optional[dict[str, Any]],
    completeness: CompletenessReport,
    facts_count: int,
    cfg: dict[str, Any],
    *,
    has_eval: bool = True,
) -> Decision:
    """
    Next loop action: ACCEPT, REFINE, or RE_RETRIEVE.

    Policy (simplified):
      - Bad grounding (low faithfulness or zero facts) → RE_RETRIEVE
      - Incomplete but grounded → REFINE (finish the draft, don't restart)
      - Good scores + complete → ACCEPT
    """
    accept_score = float(cfg.get("Agentic_loop_accept_score") or 7.0)
    min_faith = float(cfg.get("Agentic_loop_min_faithfulness") or 6)
    min_facts = int(cfg.get("Agentic_loop_min_facts") or 3)

    avg = average_score(eval_data)
    faith = criterion_score(eval_data, "faithfulness")

    if not has_eval or not eval_data:
        if completeness.ok:
            return Decision(ACCEPT, ("complete (no eval provider)",), avg, faith)
        if facts_count <= 0:
            return Decision(RE_RETRIEVE, ("no grounded facts to refine from (no eval provider)",), avg, faith)
        return Decision(REFINE, ("incomplete (no eval provider): " + "; ".join(completeness.reasons),), avg, faith)

    if faith is not None and faith < min_faith:
        return Decision(RE_RETRIEVE, (f"faithfulness {faith} < {min_faith}",), avg, faith)
    if facts_count <= 0:
        return Decision(RE_RETRIEVE, ("no grounded facts extracted",), avg, faith)

    if avg >= accept_score and completeness.ok:
        return Decision(ACCEPT, (f"avg {avg} >= {accept_score}, complete, grounded",), avg, faith)

    if not completeness.ok:
        reasons = tuple(completeness.reasons) or ("incomplete",)
        return Decision(REFINE, reasons, avg, faith)

    if facts_count < min_facts:
        return Decision(
            RE_RETRIEVE,
            (f"low quality (avg {avg}) with thin grounding ({facts_count} < {min_facts} facts)",),
            avg,
            faith,
        )
    return Decision(REFINE, (f"avg {avg} < {accept_score}",), avg, faith)


def build_feedback(eval_data: Optional[dict[str, Any]], completeness: CompletenessReport) -> str:
    """Merge evaluator notes and completeness gaps into text for the refine prompt."""
    parts: list[str] = []
    if eval_data:
        conclusion = str(eval_data.get("conclusion") or "").strip()
        if conclusion:
            parts.append(f"Reviewer conclusion: {conclusion}")
        suggestions = eval_data.get("suggestions") or []
        if isinstance(suggestions, list):
            for s in suggestions:
                s = str(s).strip()
                if s:
                    parts.append(f"- {s}")
    if not completeness.ok:
        parts.append("Completeness issues to fix: " + "; ".join(completeness.reasons))
    parts.append("Keep every section, finish the final sentence, and stay strictly grounded in the facts.")
    return "\n".join(parts).strip()


def reformulate_query(query: str, parsed, eval_data: Optional[dict[str, Any]]) -> str:
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


@dataclass
class AgenticLoopResult:
    content: str
    accepted: bool
    stop_reason: str
    iterations: list[dict[str, Any]] = field(default_factory=list)
    final_scores: dict[str, Any] = field(default_factory=dict)
    final_average: float = 0.0
    retrieval_context: str = ""
    grounded_extraction: str = ""
    grounded_facts: tuple[dict[str, Any], ...] = ()
    retrieval_chunks: tuple[dict[str, Any], ...] = ()


def run_agentic_story_loop(
    query: str,
    *,
    cfg: Optional[dict[str, Any]] = None,
    mode: Gen_mode = Gen_mode.FAST,
    story_type: StoryType = StoryType.MIX,
    debug: bool = False,
    show_progress: bool = True,
) -> AgenticLoopResult:
    from storyforge.rag.langchain_rag import (
        _docs_to_chunks,
        _docs_to_context,
        extract_grounded_facts,
        generate_from_facts,
        retrieve_docs,
    )

    cfg = cfg or load_config()

    max_iter = max(1, int(cfg.get("Agentic_loop_max_iterations") or 3))
    mode_name = mode.value if isinstance(mode, Gen_mode) else str(mode or "").strip().lower()
    is_thinking = mode_name in {Gen_mode.THINKING.value, "thinking", "think", "slow", "medium"}

    min_words = int(
        cfg.get("Agentic_loop_min_words_thinking" if is_thinking else "Agentic_loop_min_words")
        or cfg.get("Agentic_loop_min_words")
        or 250
    )
    min_sentences_per_section = int(cfg.get("Min_sentences_per_section") or 3)
    k_boost_step = float(cfg.get("Agentic_loop_reretrieve_k_boost") or 2.0)
    reretrieve_n = int(cfg.get("Agentic_loop_reretrieve_n_results") or 5)
    refine_token_boost = int(
        cfg.get("Agentic_loop_refine_token_boost_thinking" if is_thinking else "Agentic_loop_refine_token_boost")
        or cfg.get("Agentic_loop_refine_token_boost")
        or 600
    )
    base_max_tokens = int(
        cfg.get("Single_pass_thinking_max_tokens" if is_thinking else "Single_pass_fast_max_tokens")
        or cfg.get("Single_pass_fast_max_tokens")
        or 768
    )

    eval_model = None
    has_eval = False
    eval_mod = None
    try:
        from storyforge.evaluation import evaluation as eval_mod

        eval_model = eval_mod.evaluate_model()
        has_eval = True
    except Exception as e:
        LOG.warning(
            "Agentic loop: no evaluation provider available (%s). Using completeness-only signals.",
            e,
        )

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(total=max_iter, desc="Agentic RAG", unit="iter")
        except Exception:
            pbar = None

    n_stories = 3
    chunks_per_story = 2
    k_boost = 1.0
    use_reranker: Optional[bool] = None
    current_query = query

    docs = retrieve_docs(
        current_query,
        cfg,
        n_stories=n_stories,
        chunks_per_story=chunks_per_story,
        k_boost=k_boost,
        use_reranker=use_reranker,
    )
    chunks = _docs_to_chunks(docs)
    retrieval_context = _docs_to_context(docs)
    grounded_raw, parsed = extract_grounded_facts(current_query, chunks, cfg)

    iterations: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    stop_reason = "max_iterations"

    refine_feedback: Optional[str] = None
    prior_draft: Optional[str] = None
    refine_max_new: Optional[int] = None

    for i in range(1, max_iter + 1):
        story = generate_from_facts(
            query,
            parsed,
            grounded_raw,
            cfg,
            mode=mode,
            refine_feedback=refine_feedback,
            prior_draft=prior_draft,
            max_new_tokens=refine_max_new,
        )

        comp = completeness_report(
            story,
            min_words=min_words,
            min_sentences_per_section=min_sentences_per_section,
        )

        eval_data: dict[str, Any] = {}
        if has_eval and eval_mod is not None:
            try:
                eval_data = eval_mod.evaluate_story_text(eval_model, story) or {}
            except Exception as e:
                LOG.warning("Agentic loop: evaluation call failed on iteration %d (%s).", i, e)
                eval_data = {}

        iter_has_eval = has_eval and bool(eval_data)
        decision = decide_action(eval_data, comp, len(parsed.facts), cfg, has_eval=iter_has_eval)

        iter_record = {
            "iteration": i,
            "action": decision.action,
            "average_score": decision.avg,
            "faithfulness": decision.faithfulness,
            "completeness_ok": comp.ok,
            "word_count": comp.word_count,
            "missing_sections": list(comp.missing_sections),
            "reasons": list(decision.reasons),
            "query": current_query,
            "facts_count": len(parsed.facts),
            "scores": eval_data,
        }
        iterations.append(iter_record)

        candidate = {
            "story": story,
            "complete": comp.ok,
            "avg": decision.avg,
            "eval_data": eval_data,
        }
        if best is None or (candidate["complete"], candidate["avg"]) > (best["complete"], best["avg"]):
            best = candidate

        if pbar:
            pbar.update(1)

        if decision.action == ACCEPT:
            stop_reason = "accepted"
            best = candidate
            break

        if i == max_iter:
            stop_reason = "max_iterations"
            break

        if decision.action == RE_RETRIEVE:
            k_boost *= k_boost_step
            n_stories = max(n_stories, reretrieve_n)
            use_reranker = False
            current_query = reformulate_query(query, parsed, eval_data)
            docs = retrieve_docs(
                current_query,
                cfg,
                n_stories=n_stories,
                chunks_per_story=chunks_per_story,
                k_boost=k_boost,
                use_reranker=use_reranker,
            )
            chunks = _docs_to_chunks(docs)
            retrieval_context = _docs_to_context(docs)
            grounded_raw, parsed = extract_grounded_facts(current_query, chunks, cfg)
            refine_feedback = None
            prior_draft = None
            refine_max_new = None
        else:
            refine_feedback = build_feedback(eval_data, comp)
            prior_draft = story
            refine_max_new = base_max_tokens + refine_token_boost if not comp.ok else None

    if pbar:
        pbar.close()

    best = best or {"story": "", "complete": False, "avg": 0.0, "eval_data": {}}
    return AgenticLoopResult(
        content=best["story"],
        accepted=(stop_reason == "accepted"),
        stop_reason=stop_reason,
        iterations=iterations,
        final_scores=best["eval_data"],
        final_average=best["avg"],
        retrieval_context=retrieval_context,
        grounded_extraction=grounded_raw,
        grounded_facts=tuple(f.__dict__ for f in parsed.facts),
        retrieval_chunks=tuple(chunks),
    )


__all__ = [
    "ACCEPT",
    "RE_RETRIEVE",
    "REFINE",
    "CompletenessReport",
    "Decision",
    "AgenticLoopResult",
    "completeness_report",
    "average_score",
    "criterion_score",
    "decide_action",
    "build_feedback",
    "reformulate_query",
    "run_agentic_story_loop",
]
