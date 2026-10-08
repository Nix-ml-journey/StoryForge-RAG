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
    "refine_regressed",
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
    iteration: int = 1,
) -> Decision:
    """Choose ACCEPT, REFINE, or RE_RETRIEVE from scores + completeness.

    From the 2nd iteration on, the accept bar drops by ``Agentic_loop_late_accept_slack``
    (default 0.5): a complete, grounded draft that is only marginally under the
    bar is accepted instead of paying another ~100 s refine pass for noise-level gain.
    """
    accept_score = float(cfg.get("Agentic_loop_accept_score") or 7.0)
    if iteration >= 2:
        accept_score -= float(cfg.get("Agentic_loop_late_accept_slack", 0.5) or 0.0)
    min_faith = float(cfg.get("Agentic_loop_min_faithfulness") or 6)
    min_facts = int(cfg.get("Agentic_loop_min_facts") or 3)
    rich_facts = int(cfg.get("Agentic_loop_rich_facts") or 8)

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

    if facts_count <= 0:
        return Decision(RE_RETRIEVE, ("no grounded facts extracted",), avg, faith)
    # Finish an incomplete/short draft BEFORE judging faithfulness: a truncated draft
    # scores noisy faithfulness, and re-retrieving would throw away its good sections.
    if not completeness.ok:
        reasons = tuple(completeness.reasons) or ("incomplete",)
        return Decision(REFINE, reasons, avg, faith)
    if faith is not None and faith < min_faith:
        if (iteration == 1 and facts_count >= min_facts) or facts_count >= rich_facts:
            # A 4B judge swings 4 -> 8 on the same facts, and with plenty of facts the problem is
            # the writer's invented details, not missing evidence (huntress 17-30 facts, doctor 11-17:
            # three re-retrieves all stayed at 4.0). Rewrite the offending sections instead.
            return Decision(REFINE, (f"faithfulness {faith} < {min_faith} (refine unsupported details)",), avg, faith)
        return Decision(RE_RETRIEVE, (f"faithfulness {faith} < {min_faith}",), avg, faith)

    # Thin grounding blocks ACCEPT: high scores on 1-3 facts mostly reward invented filler.
    if facts_count < min_facts:
        return Decision(
            RE_RETRIEVE,
            (f"thin grounding ({facts_count} < {min_facts} facts), avg {avg}",),
            avg,
            faith,
        )
    if avg >= accept_score:
        return Decision(ACCEPT, (f"avg {avg} >= {accept_score}, complete, grounded",), avg, faith)
    return Decision(REFINE, (f"avg {avg} < {accept_score}",), avg, faith)


def refine_regressed(prior: str, new: str, *, min_ratio: float = 0.8, min_words: int = 0) -> bool:
    """True when a refine pass made the draft worse: much shorter, or fewer sections.

    With ``min_words`` a shorter draft is fine as long as it still meets the minimum
    (tightening an over-long, cut-off draft is exactly what the refine asks for).
    """
    count = lambda s: len({int(n) for n in re.findall(r"\[SECTION\s+(\d+)", s or "", flags=re.IGNORECASE)})  # noqa: E731
    new_words = len((new or "").split())
    shrank = new_words < min_ratio * len((prior or "").split())
    if min_words:
        shrank = shrank and new_words < min_words
    return shrank or count(new) < count(prior)


def build_feedback(
    eval_data: Optional[dict[str, Any]],
    completeness: CompletenessReport,
    *,
    words_per_section: int = 0,
    expand_by_words: int = 0,
) -> str:
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
        if (completeness.missing_sections or not completeness.ends_clean) and words_per_section > 0:
            # Draft ran out of tokens before the ending: the fix is tighter earlier
            # sections, not a longer story.
            parts.append(
                f"The previous draft was cut off. Keep each section to about {words_per_section} "
                "words so all five sections fit, and end SECTION 5 with a complete sentence."
            )
    if expand_by_words > 0:
        # The last rewrite came back shorter and was discarded; repeating the same
        # feedback would reproduce it, so ask for explicit growth instead.
        parts.append(
            f"The previous rewrite was shorter and was discarded. Do NOT shorten: add about "
            f"{expand_by_words} words of grounded detail and dialogue across the sections, "
            "keeping every existing section."
        )
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


def _iteration_record(
    iteration: int,
    action: str,
    query: str,
    facts_count: int,
    *,
    avg: Optional[float] = None,
    faith: Optional[float] = None,
    comp: Optional[CompletenessReport] = None,
    words: int = 0,
    reasons: Any = (),
    has_eval: Optional[bool] = None,
    scores: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """One row of ``AgenticLoopResult.iterations`` (same shape for every outcome)."""
    rec: dict[str, Any] = {
        "iteration": iteration,
        "action": action,
        "average_score": avg,
        "faithfulness": faith,
        "completeness_ok": comp.ok if comp else False,
        "word_count": comp.word_count if comp else words,
        "missing_sections": list(comp.missing_sections) if comp else [],
        "reasons": list(reasons),
        "query": query,
        "facts_count": facts_count,
        "scores": scores or {},
    }
    if has_eval is not None:
        rec["has_eval"] = has_eval
    return rec


def _rule_driven_enabled(cfg: dict[str, Any]) -> bool:
    """Rule-driven loop needs a section-by-section writer (arc, or 5W1H sectioned)."""
    if str(cfg.get("Agentic_loop_rule_driven") or "").strip().lower() not in ("true", "1", "yes"):
        return False
    method = str(cfg.get("Story_generation_method") or "arc").strip().lower()
    sectioned = str(cfg.get("Story_generation_mode") or "single").strip().lower() == "sectioned"
    if method == "arc" or sectioned:
        return True
    LOG.warning("Agentic_loop_rule_driven needs Story_generation_method 'arc' or mode 'sectioned'; using the judge-driven loop.")
    return False


def _run_rule_driven(
    query: str, parsed, grounded_raw: str, cfg: dict[str, Any], *, mode, profile,
    current_query: str, eval_mod, eval_model, retrieval_context: str, chunks,
) -> "AgenticLoopResult":
    """ADR-0011 Step 2: write all sections, then rewrite only the sections that fail a rule.

    Rule checks (length, finished sentence, novel names) decide; the judge runs once at the end and
    only reports a score. At most ``Agentic_loop_rule_rounds`` rewrite rounds (default 2).
    """
    from storyforge.rag.attribution import format_facts_for_prompt
    from storyforge.rag.generation import generate_from_facts
    from storyforge.rag.section_rules import failure_feedback, section_failures

    rounds = max(0, int(cfg.get("Agentic_loop_rule_rounds", 2) or 0))
    max_novel = int(cfg.get("Agentic_loop_rule_max_novel_names", 3) or 3)
    facts_n = len(parsed.facts)
    iterations: list[dict[str, Any]] = []
    story = ""
    failures: dict[int, list[str]] = {}
    stop_reason = "rules_max_rounds"
    prior: Optional[str] = None
    feedback: Optional[str] = None
    rewrite: Optional[frozenset[int]] = None

    for rnd in range(rounds + 1):
        try:
            draft = generate_from_facts(
                query, parsed, grounded_raw, cfg, mode=mode, profile=profile,
                refine_feedback=feedback, prior_draft=prior, rewrite_sections=rewrite,
            )
        except RuntimeError as e:
            LOG.warning("Rule loop: generation failed in round %d (%s).", rnd, e)
            iterations.append(_iteration_record(rnd + 1, "generation_failed", current_query, facts_n, reasons=[str(e)]))
            stop_reason = "generation_failed" if not story else "generation_failed_using_best_so_far"
            break
        new_failures = section_failures(
            draft, tuple(parsed.facts), profile, max_novel_names=max_novel
        )
        if story and len(new_failures) > len(failures):
            LOG.warning("Rule loop: round %d left more failing sections (%d > %d); keeping the previous draft.",
                        rnd, len(new_failures), len(failures))
            iterations.append(_iteration_record(rnd + 1, "rewrite_rejected", current_query, facts_n,
                                                words=len(story.split()), reasons=["rewrite worsened the draft"]))
            break
        story, failures = draft, new_failures
        comp = completeness_report(story, min_words=profile.min_words,
                                   min_sentences_per_section=profile.min_sentences_per_section)
        rec = _iteration_record(
            rnd + 1, "write" if rnd == 0 else "rewrite", current_query, facts_n, comp=comp,
            reasons=[f"SECTION {i}: {'; '.join(w)}" for i, w in sorted(failures.items())],
        )
        iterations.append(rec)
        if not failures:
            stop_reason = "accepted"
            break
        prior = story
        rewrite = frozenset(failures)
        feedback = failure_feedback(failures, profile.words_per_section)

    eval_data: dict[str, Any] = {}
    if story and eval_mod is not None:
        try:
            eval_data = eval_mod.evaluate_story_text(eval_model, story, facts=format_facts_for_prompt(parsed)) or {}
        except Exception as e:  # noqa: BLE001 - a judge failure must not discard a finished draft
            LOG.error("Rule loop: judge failed (%s: %s); score not reported.", type(e).__name__, e)
    avg = average_score(eval_data) if eval_data else 0.0
    if iterations:
        iterations[-1].update(
            average_score=avg, faithfulness=criterion_score(eval_data, "faithfulness"),
            scores=eval_data, has_eval=bool(eval_data),
        )
    return AgenticLoopResult(
        content=story,
        accepted=(stop_reason == "accepted"),
        stop_reason=stop_reason,
        iterations=iterations,
        final_scores=eval_data,
        final_average=avg,
        retrieval_context=retrieval_context,
        grounded_extraction=grounded_raw,
        grounded_facts=tuple(f.__dict__ for f in parsed.facts),
        retrieval_chunks=tuple(chunks),
    )


def run_agentic_story_loop(
    query: str,
    *,
    cfg: Optional[dict[str, Any]] = None,
    mode: Gen_mode = Gen_mode.FAST,
    length: Any = None,
    story_type: StoryType = StoryType.MIX,
    debug: bool = False,
) -> AgenticLoopResult:
    """Retrieve → generate → evaluate → ACCEPT / REFINE / RE_RETRIEVE."""
    from storyforge.rag.attribution import format_facts_for_prompt, merge_facts
    from storyforge.rag.extraction import extract_grounded_facts
    from storyforge.rag.generation import generate_from_facts
    from storyforge.rag.length_profile import is_thinking_mode, resolve_length_profile
    from storyforge.rag.retrieval import _docs_to_chunks, _docs_to_context, retrieve_docs
    from storyforge.rag.section_rules import failure_feedback, section_failures

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
        or 900
    )

    eval_model = None
    has_eval = False
    eval_mod = None
    try:
        from storyforge.evaluation import evaluation as eval_mod

        eval_model = eval_mod.evaluate_model()
        has_eval = True
    except (RuntimeError, ValueError, OSError) as e:
        LOG.error(
            "Agentic loop: evaluator unavailable (%s). Using completeness-only signals; "
            "scores will be 0.0 for this run.",
            e,
        )

    n_stories = 3
    chunks_per_story = 2
    k_boost = 1.0
    current_query = query

    def retrieve_and_extract(q: str, previous=None):
        docs = retrieve_docs(
            q, cfg, n_stories=n_stories, chunks_per_story=chunks_per_story,
            k_boost=k_boost, story_type=story_type,
        )
        chunks = _docs_to_chunks(docs)
        raw, facts = extract_grounded_facts(q, chunks, cfg)
        if previous is not None:
            facts = merge_facts(previous, facts)  # widening adds evidence, never replaces it
        return _docs_to_context(docs), chunks, raw, facts

    retrieval_context, chunks, grounded_raw, parsed = retrieve_and_extract(current_query)

    # Pre-flight gate: thin evidence is fixed BEFORE any story is written. Tier 1 widens the
    # search; tier 2 also reformulates the query with names from the facts found so far.
    gate_min = int(cfg.get("Agentic_loop_min_facts") or 3)
    for tier in range(1, int(cfg.get("Agentic_loop_preflight_tiers", 2) or 0) + 1):
        if len(parsed.facts) >= gate_min:
            break
        LOG.warning(
            "Pre-flight gate: %d facts < %d; widening retrieval (tier %d).", len(parsed.facts), gate_min, tier
        )
        k_boost *= k_boost_step
        n_stories = max(n_stories, reretrieve_n)
        if tier >= 2:
            current_query = reformulate_query(query, parsed, {})
        retrieval_context, chunks, grounded_raw, parsed = retrieve_and_extract(current_query, parsed)

    if _rule_driven_enabled(cfg):
        return _run_rule_driven(
            query, parsed, grounded_raw, cfg, mode=mode, profile=profile, current_query=current_query,
            eval_mod=eval_mod if has_eval else None, eval_model=eval_model,
            retrieval_context=retrieval_context, chunks=chunks,
        )

    iterations: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    stop_reason = "max_iterations"

    refine_feedback: Optional[str] = None
    prior_draft: Optional[str] = None
    refine_max_new: Optional[int] = None
    rewrite_only: Optional[frozenset[int]] = None
    prev_regressed = False
    last_action = ""
    stop_on_stall = str(cfg.get("Agentic_loop_stop_on_no_progress", True)).strip().lower() not in ("false", "0", "no")
    max_novel = int(cfg.get("Agentic_loop_rule_max_novel_names", 3) or 3)

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
                rewrite_sections=rewrite_only,
            )
        except RuntimeError as e:
            # generate_from_facts already retried thinking -> fast once and still
            # came back empty: stop with the best draft seen so far rather than
            # losing the whole run to one bad generation call.
            LOG.warning(
                "Agentic loop: generation failed on iteration %d (%s). "
                "Stopping with the best draft found so far.", i, e,
            )
            iterations.append(_iteration_record(i, "generation_failed", current_query, len(parsed.facts), reasons=[str(e)]))
            stop_reason = "generation_failed" if best is None else "generation_failed_using_best_so_far"
            break

        regressed = bool(prior_draft) and refine_regressed(prior_draft, story, min_words=min_words)
        if regressed and prev_regressed:
            LOG.warning("Agentic loop: refine rejected twice in a row on iteration %d; stopping.", i)
            iterations.append(
                _iteration_record(
                    i, "refine_rejected", current_query, len(parsed.facts),
                    words=len(prior_draft.split()), reasons=["refine regressed twice; keeping the prior draft"],
                )
            )
            stop_reason = "refine_rejected"
            break
        if regressed:
            LOG.warning(
                "Agentic loop: refine on iteration %d regressed the draft (%d -> %d words); keeping the prior draft.",
                i, len(prior_draft.split()), len(story.split()),
            )
            story = prior_draft
        prev_regressed = regressed

        comp = completeness_report(
            story, min_words=min_words, min_sentences_per_section=min_sentences_per_section
        )

        eval_data: dict[str, Any] = {}
        if has_eval and eval_mod is not None:
            try:
                eval_data = eval_mod.evaluate_story_text(
                    eval_model, story, facts=format_facts_for_prompt(parsed)
                ) or {}
            except Exception as e:  # noqa: BLE001 - any provider failure must not discard a finished draft
                LOG.error("Agentic loop: evaluation failed on iteration %d (%s: %s).", i, type(e).__name__, e)
            if not eval_data:
                LOG.warning(
                    "Agentic loop: judge produced no usable scores on iteration %d; "
                    "decision falls back to completeness only.", i,
                )

        iter_has_eval = has_eval and bool(eval_data)
        decision = decide_action(
            eval_data, comp, len(parsed.facts), cfg, has_eval=iter_has_eval, iteration=i
        )
        iterations.append(
            _iteration_record(
                i, decision.action, current_query, len(parsed.facts),
                avg=decision.avg, faith=decision.faithfulness, comp=comp,
                reasons=list(decision.reasons), has_eval=iter_has_eval, scores=eval_data,
            )
        )

        candidate = {"story": story, "complete": comp.ok, "avg": decision.avg, "eval_data": eval_data}
        prior_best_avg = best["avg"] if best else 0.0
        if best is None or (candidate["complete"], candidate["avg"]) > (best["complete"], best["avg"]):
            best = candidate

        if decision.action == ACCEPT:
            stop_reason = "accepted"
            best = candidate
            break

        if (
            stop_on_stall and iter_has_eval and i >= 2 and decision.action == REFINE and last_action == REFINE
            and decision.avg < prior_best_avg + 0.5
        ):
            # Two refines in a row that did not beat the best draft: another pass is a coin flip on judge
            # noise (lawyer: 5.4 -> 4.6 -> 5.0 over 186 s). Keep the best draft and stop.
            LOG.warning("Agentic loop: refine made no progress (avg %.1f vs best %.1f); stopping.", decision.avg, prior_best_avg)
            stop_reason = "no_progress"
            break

        last_action = decision.action

        if i == max_iter:
            # "Ran out of iterations with nothing grounded to write from" is a Step 2
            # (facts extraction) failure, not a generation-quality one.
            stop_reason = "max_iterations_no_grounded_facts" if not parsed.facts else "max_iterations"
            break

        if decision.action == RE_RETRIEVE:
            prev_regressed = False
            k_boost *= k_boost_step
            n_stories = max(n_stories, reretrieve_n)
            current_query = reformulate_query(query, parsed, eval_data)
            retrieval_context, chunks, grounded_raw, parsed = retrieve_and_extract(current_query, parsed)
            refine_feedback = prior_draft = refine_max_new = rewrite_only = None
        else:
            expand = max(100, profile.target_words - comp.word_count) if regressed else 0
            refine_feedback = build_feedback(
                eval_data, comp, words_per_section=profile.words_per_section, expand_by_words=expand
            )
            # Name the unsupported details instead of trusting the judge's vague advice, and rewrite only
            # the sections that carry them.
            failures = section_failures(story, tuple(parsed.facts), profile, max_novel_names=max_novel)
            rewrite_only = frozenset(failures) or None
            if failures:
                refine_feedback += "\n" + failure_feedback(failures, profile.words_per_section)
            prior_draft = story
            refine_max_new = base_max_tokens + refine_token_boost if not comp.ok else None

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
