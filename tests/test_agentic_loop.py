from __future__ import annotations

from storyforge.rag.agentic_loop import (
    ACCEPT,
    RE_RETRIEVE,
    REFINE,
    CompletenessReport,
    average_score,
    build_feedback,
    completeness_report,
    criterion_score,
    decide_action,
    reformulate_query,
)
from storyforge.rag.attribution import parse_grounded_facts_json

# Pure decision helpers only — no GPU or API.

CFG = {
    "Agentic_loop_accept_score": 7.0,
    "Agentic_loop_min_faithfulness": 6,
    "Agentic_loop_min_facts": 6,
}


def _five_section_story(extra_words: int = 60) -> str:
    body = " ".join(["word"] * extra_words)
    return (
        "[SECTION 1: WHO]\n" + body + ".\n\n"
        "[SECTION 2: WHAT]\n" + body + ".\n\n"
        "[SECTION 3: TWIST]\n" + body + ".\n\n"
        "[SECTION 4: HOW]\n" + body + ".\n\n"
        "[SECTION 5: WHY]\n" + body + " the end."
    )


# completeness_report
def test_completeness_report_complete_story():
    rep = completeness_report(_five_section_story(), min_words=50)
    assert rep.ok is True
    assert rep.missing_sections == ()
    assert rep.ends_clean is True


def test_completeness_report_missing_sections():
    story = "[SECTION 1: WHO]\nOnce upon a time there was a hero who saved the day."
    rep = completeness_report(story, min_words=5)
    assert rep.ok is False
    assert rep.missing_sections == (2, 3, 4, 5)


def test_completeness_report_unfinished_ending():
    story = _five_section_story().rstrip(" the end.") + " and then they"
    rep = completeness_report(story, min_words=5)
    assert rep.ends_clean is False
    assert rep.ok is False


def test_completeness_report_too_short():
    story = "[SECTION 1: A][SECTION 2: B][SECTION 3: C][SECTION 4: D][SECTION 5: E] short."
    rep = completeness_report(story, min_words=250)
    assert rep.ok is False
    assert any("too short" in r for r in rep.reasons)


# average_score / criterion_score
def test_average_score_from_nested_criteria():
    data = {
        "coherence": {"score": 8, "feedback": "x"},
        "grammar": {"score": 6, "feedback": "x"},
        "creativity": {"score": 7, "feedback": "x"},
        "faithfulness": {"score": 9, "feedback": "x"},
        "overall": {"score": 10, "feedback": "x"},
        "conclusion": "ignored",
        "suggestions": ["ignored"],
    }
    assert average_score(data) == 8.0


def test_average_score_empty_is_zero():
    assert average_score({}) == 0.0
    assert average_score(None) == 0.0


def test_average_score_prefers_explicit_value():
    assert average_score({"average_score": 5.5, "coherence": {"score": 1}}) == 5.5


def test_criterion_score_reads_faithfulness():
    data = {"faithfulness": {"score": 4}}
    assert criterion_score(data, "faithfulness") == 4.0
    assert criterion_score(data, "coherence") is None


# decide_action (with eval)
def _good_eval(faith: int = 9, others: int = 9) -> dict:
    return {
        "coherence": {"score": others},
        "grammar": {"score": others},
        "creativity": {"score": others},
        "faithfulness": {"score": faith},
        "overall": {"score": others},
    }


def _complete() -> CompletenessReport:
    return completeness_report(_five_section_story(), min_words=50)


def _incomplete() -> CompletenessReport:
    return completeness_report("[SECTION 1: WHO]\nUnfinished story that just stops", min_words=50)


def test_decide_accept():
    d = decide_action(_good_eval(), _complete(), facts_count=10, cfg=CFG, has_eval=True)
    assert d.action == ACCEPT


def test_decide_re_retrieve_on_low_faithfulness():
    d = decide_action(_good_eval(faith=3), _complete(), facts_count=10, cfg=CFG, has_eval=True)
    assert d.action == RE_RETRIEVE


def test_decide_re_retrieve_on_thin_facts():
    # Not good enough to accept (avg < 7) AND too few grounded facts -> re-retrieve,
    # even though faithfulness alone is fine. (ACCEPT correctly takes precedence
    # when scores are high, so we use a non-accepting eval here.)
    d = decide_action(_good_eval(faith=8, others=4), _complete(), facts_count=2, cfg=CFG, has_eval=True)
    assert d.action == RE_RETRIEVE


def test_decide_refine_on_low_quality_but_grounded():
    # Faithful and enough facts, but low overall scores + incomplete -> refine.
    d = decide_action(_good_eval(faith=8, others=3), _incomplete(), facts_count=10, cfg=CFG, has_eval=True)
    assert d.action == REFINE


# decide_action (no eval provider)
def test_decide_no_eval_complete_accepts():
    d = decide_action({}, _complete(), facts_count=10, cfg=CFG, has_eval=False)
    assert d.action == ACCEPT


def test_decide_no_eval_incomplete_refines():
    d = decide_action({}, _incomplete(), facts_count=10, cfg=CFG, has_eval=False)
    assert d.action == REFINE


def test_decide_no_eval_thin_facts_still_refines():
    # With some grounded facts to work from, an incomplete draft is refined (not
    # re-retrieved) even without an eval provider.
    d = decide_action({}, _incomplete(), facts_count=2, cfg=CFG, has_eval=False)
    assert d.action == REFINE


def test_decide_no_eval_zero_facts_re_retrieves():
    # Nothing grounded to refine from -> re-retrieve.
    d = decide_action({}, _incomplete(), facts_count=0, cfg=CFG, has_eval=False)
    assert d.action == RE_RETRIEVE


def test_decide_incomplete_but_grounded_refines_before_re_retrieve():
    # Key policy: an incomplete draft with acceptable grounding refines to completion
    # rather than re-retrieving, even when facts are below min_facts.
    d = decide_action(_good_eval(faith=8, others=8), _incomplete(), facts_count=2, cfg=CFG, has_eval=True)
    assert d.action == REFINE


# build_feedback / reformulate_query
def test_build_feedback_includes_suggestions_and_gaps():
    data = {"conclusion": "Story stops abruptly.", "suggestions": ["Finish the ending", "Reduce repetition"]}
    fb = build_feedback(data, _incomplete())
    assert "Finish the ending" in fb
    assert "Reduce repetition" in fb
    assert "Completeness issues" in fb


def test_reformulate_query_appends_entities():
    parsed = parse_grounded_facts_json(
        '{"facts":[{"type":"who","fact":"Alana fights Zoruk","source_chunk_ids":["c1"]},'
        '{"type":"where","fact":"In Eldoria","source_chunk_ids":["c2"]}]}'
    )
    out = reformulate_query("the duel", parsed, {})
    assert out.startswith("the duel")
    assert "Alana" in out or "Zoruk" in out or "Eldoria" in out


def test_reformulate_query_unchanged_without_entities():
    parsed = parse_grounded_facts_json("{}")
    assert reformulate_query("the duel", parsed, {}) == "the duel"
