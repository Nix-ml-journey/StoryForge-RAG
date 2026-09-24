"""Agentic story loop: retrieve → generate → evaluate → decide → repeat.

ACCEPT / REFINE / RE_RETRIEVE until quality + completeness pass, or max iterations.
Pure helpers (completeness_report, decide_action) are unit-testable without GPU.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from storyforge.config.config import load_config
from storyforge.rag.generative_ai import Gen_mode, StoryType
from storyforge.rag.length_profile import sentence_count as _sentence_count
from storyforge.rag.length_profile import split_section_bodies as _split_section_bodies

LOG = logging.getLogger(__name__)

__all__ = [
    "ACCEPT",
    "REFINE",
    "RE_RETRIEVE",
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

ACCEPT = "accept"
RE_RETRIEVE = "re_retrieve"
REFINE = "refine"

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

# Include closing quotes so dialogue endings count as terminal.
_TERMINAL_CHARS = frozenset('.!?"\u201d\u2019\'')
_EXPECTED_SECTIONS = 5


@dataclass(frozen=True)
class CompletenessReport:
    ok: bool
    word_count: int
    missing_sections: tuple[int, ...]
    ends_clean: bool
    reasons: tuple[str, ...] = ()


def completeness_report(
    story: str,
    *,
    min_words: int = 250,
    expected_sections: int = _EXPECTED_SECTIONS,
    min_sentences_per_section: int = 0,
) -> CompletenessReport:
    """Rule check: all sections present, terminal ending, min words / sentences."""
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
    """Choose ACCEPT, REFINE, or RE_RETRIEVE from scores + completeness."""
    accept_score = float(cfg.get("Agentic_loop_accept_score") or 7.0)
    min_faith = float(cfg.get("Agentic_loop_min_faithfulness") or 6)
    min_facts = int(cfg.get("Agentic_loop_min_facts") or 3)

    avg = average_score(eval_data)
    faith = criterion_score(eval_data, "faithfulness")

    if not has_eval or not eval_data:
        # Grounding contract: without an evaluator, completeness is the only
        # quality signal left, so facts_count is the only evidence the draft is
        # grounded at all. Check it FIRST -- a complete-looking draft written
        # from zero extracted facts is retrieval-only / ungrounded prose and must
        # never be ACCEPTed (it previously was, which let an HF/eval outage
        # inflate accept_rate with ungrounded stories).
        if facts_count <= 0:
            return Decision(RE_RETRIEVE, ("no grounded facts extracted (no eval provider)",), avg, faith)
        if completeness.ok:
            return Decision(ACCEPT, ("complete (no eval provider)",), avg, faith)
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
    length: Any = None,
    story_type: StoryType = StoryType.MIX,
    debug: bool = False,
    show_progress: bool = True,
) -> AgenticLoopResult:
    """Retrieve → generate → evaluate → ACCEPT / REFINE / RE_RETRIEVE."""
    from storyforge.rag.extraction import extract_grounded_facts
    from storyforge.rag.generation import generate_from_facts
    from storyforge.rag.length_profile import is_thinking_mode, resolve_length_profile
    from storyforge.rag.retrieval import _docs_to_chunks, _docs_to_context, retrieve_docs

    cfg = cfg or load_config()

    max_iter = max(1, int(cfg.get("Agentic_loop_max_iterations") or 3))
    is_thinking = is_thinking_mode(mode)

    profile = resolve_length_profile(cfg, length=length, mode=mode)
    min_words = profile.min_words
    min_sentences_per_section = profile.min_sentences_per_section
    base_max_tokens = profile.max_new_tokens
    LOG.info(
        "Agentic loop length target '%s': %d words (~%s min), min %d words, "
        "min %d sentences/section, %d max new tokens.",
        profile.name, profile.target_words, profile.estimated_minutes,
        min_words, min_sentences_per_section, base_max_tokens,
    )

    k_boost_step = float(cfg.get("Agentic_loop_reretrieve_k_boost") or 2.0)
    reretrieve_n = int(cfg.get("Agentic_loop_reretrieve_n_results") or 5)
    refine_token_boost = int(
        cfg.get("Agentic_loop_refine_token_boost_thinking" if is_thinking else "Agentic_loop_refine_token_boost")
        or cfg.get("Agentic_loop_refine_token_boost")
        or 600
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
        try:
            story = generate_from_facts(
                query,
                parsed,
                grounded_raw,
                cfg,
                mode=mode,
                profile=profile,
                refine_feedback=refine_feedback,
                prior_draft=prior_draft,
                max_new_tokens=refine_max_new,
            )
        except RuntimeError as e:
            # generate_from_facts already retried thinking -> fast once and still
            # came back empty. Unlike the length guard (which has a single prior
            # draft to fall back to), the loop may have several iterations' worth
            # of history -- stop here and return the best draft seen so far
            # instead of losing the whole run to one bad generation call.
            LOG.warning(
                "Agentic loop: generation failed on iteration %d (%s). "
                "Stopping with the best draft found so far.", i, e,
            )
            iterations.append(
                {
                    "iteration": i,
                    "action": "generation_failed",
                    "average_score": None,
                    "faithfulness": None,
                    "completeness_ok": False,
                    "word_count": 0,
                    "missing_sections": [],
                    "reasons": [str(e)],
                    "query": current_query,
                    "facts_count": len(parsed.facts),
                    "scores": {},
                }
            )
            stop_reason = "generation_failed" if best is None else "generation_failed_using_best_so_far"
            break

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
            "has_eval": iter_has_eval,
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
            # Distinguish "ran out of iterations with nothing grounded to write
            # from" from an ordinary non-converging loop -- the first is a Step 2
            # (facts extraction) failure, not a generation-quality one.
            stop_reason = "max_iterations_no_grounded_facts" if not parsed.facts else "max_iterations"
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
