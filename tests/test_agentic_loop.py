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


def test_completeness_report_flags_sections_below_sentence_minimum():
    rep = completeness_report(_five_section_story(extra_words=20), min_words=5, min_sentences_per_section=3)
    assert rep.ok is False
    assert any("sections below sentence minimum" in r for r in rep.reasons)


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


def test_decide_no_eval_complete_but_zero_facts_does_not_accept():
    # Grounding contract (P0.3): with no evaluator, a complete-looking draft
    # written from zero grounded facts must NOT be accepted.
    d = decide_action({}, _complete(), facts_count=0, cfg=CFG, has_eval=False)
    assert d.action == RE_RETRIEVE
    assert d.action != ACCEPT
    assert any("no grounded facts" in r for r in d.reasons)


def test_decide_no_eval_complete_negative_facts_does_not_accept():
    d = decide_action({}, _complete(), facts_count=-1, cfg=CFG, has_eval=False)
    assert d.action == RE_RETRIEVE


def test_decide_eval_dict_empty_treated_as_no_eval_and_requires_facts():
    # has_eval=True but the eval call returned nothing -> same no-eval branch.
    d = decide_action({}, _complete(), facts_count=0, cfg=CFG, has_eval=True)
    assert d.action == RE_RETRIEVE


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


# ---------------------------------------------------------------------------
# run_agentic_story_loop: generation-failure recovery
#
# generate_from_facts raises RuntimeError when both a thinking draft and its
# fast-mode retry come back empty (see generation.py). Unlike the 3-step
# length guard (which has exactly one prior draft to fall back to), the
# agentic loop may have several iterations of history -- it should stop with
# the best draft seen so far instead of letting the exception abort the whole
# run and lose that history. retrieve_docs / extract_grounded_facts /
# generate_from_facts / evaluate_model are all imported lazily inside
# run_agentic_story_loop, so they're patched on their source modules.
# ---------------------------------------------------------------------------


def _fake_docs_chain(monkeypatch, *, facts_json):
    import storyforge.rag.retrieval as retrieval_mod
    import storyforge.rag.extraction as extraction_mod
    import storyforge.evaluation.evaluation as evaluation_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_docs", lambda *a, **k: ["doc"])
    monkeypatch.setattr(retrieval_mod, "_docs_to_chunks", lambda docs: [{"chunk_id": "c1", "title": "T", "metadata": {}, "text": "hi"}])
    monkeypatch.setattr(retrieval_mod, "_docs_to_context", lambda docs: "context")

    parsed = parse_grounded_facts_json(facts_json)
    monkeypatch.setattr(extraction_mod, "extract_grounded_facts", lambda query, chunks, cfg: (facts_json, parsed))

    # Force has_eval=False so decide_action uses its simpler no-eval-provider
    # branch (ACCEPT if complete, else REFINE/RE_RETRIEVE on facts_count) --
    # the exact eval score isn't what this test is about.
    def _raise_no_eval(*a, **k):
        raise RuntimeError("no evaluation provider configured for this test")
    monkeypatch.setattr(evaluation_mod, "evaluate_model", _raise_no_eval)


def test_generation_failure_on_first_iteration_stops_gracefully(stub_heavy_deps, monkeypatch):
    from storyforge.rag.agentic_loop import run_agentic_story_loop
    import storyforge.rag.generation as generation_mod

    _fake_docs_chain(
        monkeypatch,
        facts_json='{"facts":[{"type":"who","fact":"Alana fights Zoruk","source_chunk_ids":["c1"]}]}',
    )

    def _raise_empty(*a, **k):
        raise RuntimeError("Story generation returned an empty draft.")
    monkeypatch.setattr(generation_mod, "generate_from_facts", _raise_empty)

    cfg = {"Agentic_loop_max_iterations": 3}
    result = run_agentic_story_loop(
        "a test query", cfg=cfg, debug=False, show_progress=False
    )

    assert result.content == ""
    assert result.accepted is False
    assert result.stop_reason == "generation_failed"
    assert len(result.iterations) == 1
    assert result.iterations[0]["action"] == "generation_failed"


def test_generation_failure_after_partial_progress_keeps_best_draft(stub_heavy_deps, monkeypatch):
    from storyforge.rag.agentic_loop import run_agentic_story_loop
    import storyforge.rag.generation as generation_mod

    _fake_docs_chain(
        monkeypatch,
        facts_json='{"facts":[{"type":"who","fact":"Alana fights Zoruk","source_chunk_ids":["c1"]}]}',
    )

    # Iteration 1: incomplete (missing sections) but non-empty -> REFINE, not
    # ACCEPT, so the loop continues to a second iteration and this draft
    # becomes `best`. Iteration 2: the refine call comes back empty twice
    # (thinking + fast retry) and raises.
    first_draft = "[SECTION 1: WHO]\nAn unfinished story that just stops."
    calls = {"n": 0}

    def _first_ok_then_raise(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return first_draft
        raise RuntimeError("Story generation returned an empty draft.")

    monkeypatch.setattr(generation_mod, "generate_from_facts", _first_ok_then_raise)

    cfg = {"Agentic_loop_max_iterations": 3}
    result = run_agentic_story_loop(
        "a test query", cfg=cfg, debug=False, show_progress=False
    )

    assert result.content == first_draft
    assert result.accepted is False
    assert result.stop_reason == "generation_failed_using_best_so_far"
    assert len(result.iterations) == 2
    assert result.iterations[0]["action"] == "refine"
    assert result.iterations[1]["action"] == "generation_failed"
    assert calls["n"] == 2


def _complete_multi_sentence_story(sentences_per_section: int = 6) -> str:
    """A draft that passes completeness_report for a small length target."""
    sentence = "Alana walked through the quiet village and listened to the wind in the old trees."
    body = " ".join([sentence] * sentences_per_section)
    return "\n\n".join(f"[SECTION {i}: PART]\n{body}" for i in range(1, 6))


def test_no_eval_zero_facts_complete_draft_is_never_accepted(stub_heavy_deps, monkeypatch):
    """End-to-end loop: eval down + facts extraction empty + complete draft.

    Before the P0.3 fix this returned accepted=True on iteration 1 with zero
    grounded facts behind the story.
    """
    from storyforge.rag.agentic_loop import run_agentic_story_loop
    import storyforge.rag.generation as generation_mod

    _fake_docs_chain(monkeypatch, facts_json='{"facts": []}')
    story = _complete_multi_sentence_story()
    monkeypatch.setattr(generation_mod, "generate_from_facts", lambda *a, **k: story)

    cfg = {"Agentic_loop_max_iterations": 2, "Story_length_presets": {"short": 100}}
    result = run_agentic_story_loop(
        "a test query", cfg=cfg, length="short", debug=False, show_progress=False
    )

    assert result.accepted is False
    assert result.stop_reason == "max_iterations_no_grounded_facts"
    assert [it["action"] for it in result.iterations] == [RE_RETRIEVE, RE_RETRIEVE]
    assert all(it["facts_count"] == 0 for it in result.iterations)
    assert all(it["has_eval"] is False for it in result.iterations)
    # Sanity: the draft itself was complete, so ONLY missing facts blocked ACCEPT.
    assert all(it["completeness_ok"] is True for it in result.iterations)
    # Best-effort draft is still returned for inspection, just not accepted.
    assert result.content == story


def test_no_eval_with_facts_complete_draft_accepts(stub_heavy_deps, monkeypatch):
    from storyforge.rag.agentic_loop import run_agentic_story_loop
    import storyforge.rag.generation as generation_mod

    _fake_docs_chain(
        monkeypatch,
        facts_json='{"facts":[{"type":"who","fact":"Alana fights Zoruk","source_chunk_ids":["c1"]}]}',
    )
    story = _complete_multi_sentence_story()
    monkeypatch.setattr(generation_mod, "generate_from_facts", lambda *a, **k: story)

    cfg = {"Agentic_loop_max_iterations": 2, "Story_length_presets": {"short": 100}}
    result = run_agentic_story_loop(
        "a test query", cfg=cfg, length="short", debug=False, show_progress=False
    )

    assert result.accepted is True
    assert result.stop_reason == "accepted"
    assert result.iterations[0]["has_eval"] is False
