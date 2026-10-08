"""ADR-0011 Step 2: per-section rule checks and the rule-driven loop."""
from __future__ import annotations

import pytest

from storyforge.rag.attribution import parse_grounded_facts_json
from storyforge.rag.length_profile import resolve_length_profile
from storyforge.rag.section_rules import failure_feedback, section_failures

SENTENCE = "Alana crossed the quiet village and listened to the wind in the old trees."
FACTS_JSON = '{"facts":[{"type":"who","fact":"Alana walks through the village","source_chunk_ids":["c1"]}]}'
PROFILE = resolve_length_profile({"Story_length_presets": {}}, length=450)


@pytest.fixture(autouse=True)
def _auto_stub(stub_heavy_deps):
    """langchain / HF / torch stubs for every test here."""


def _body(n=6):
    return " ".join([SENTENCE] * n)


def _story(bodies):
    return "\n\n".join(f"[SECTION {i}: X]\n{b}" for i, b in enumerate(bodies, start=1))


def _facts():
    return parse_grounded_facts_json(FACTS_JSON).facts


def test_clean_story_has_no_failures():
    assert section_failures(_story([_body()] * 5), _facts(), PROFILE) == {}


def test_missing_cut_off_short_and_long_sections_are_flagged():
    bodies = [_body(), _body() + " And then the", _body(1), _body(40)]
    failures = section_failures(_story(bodies), _facts(), PROFILE)  # section 5 absent
    assert 1 not in failures
    assert failures[5] == ["missing"]
    assert any("finished sentence" in w for w in failures[2])
    assert any("sentences" in w or "under" in w for w in failures[3])
    assert any("over the" in w for w in failures[4])


def test_novel_names_over_limit_are_flagged():
    novel = _body() + " Then Zorbax met Quillon and Mara near Dren."
    failures = section_failures(_story([_body(), novel, _body(), _body(), _body()]), _facts(), PROFILE)
    assert list(failures) == [2]
    assert "names not in the facts" in failures[2][0]


def test_failure_feedback_names_each_section():
    text = failure_feedback({2: ["too long"], 4: ["missing"]}, 90)
    assert "SECTION 2: too long" in text and "SECTION 4: missing" in text and "90 words" in text


def _loop_cfg(**extra):
    return {
        "Agentic_loop_rule_driven": True,
        "Story_generation_method": "arc",
        "Agentic_loop_preflight_tiers": 0,
        "Story_length_presets": {},
        **extra,
    }


def _patch_chain(monkeypatch):
    import storyforge.evaluation.evaluation as evaluation_mod
    import storyforge.rag.extraction as extraction_mod
    import storyforge.rag.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_docs", lambda *a, **k: ["doc"])
    monkeypatch.setattr(retrieval_mod, "_docs_to_chunks", lambda docs: [{"chunk_id": "c1", "title": "T", "metadata": {}, "text": "hi"}])
    monkeypatch.setattr(retrieval_mod, "_docs_to_context", lambda docs: "context")
    parsed = parse_grounded_facts_json(FACTS_JSON)
    monkeypatch.setattr(extraction_mod, "extract_grounded_facts", lambda q, c, cfg: (FACTS_JSON, parsed))
    monkeypatch.setattr(evaluation_mod, "evaluate_model", lambda *a, **k: "judge")
    return evaluation_mod


def test_rule_loop_rewrites_only_failing_sections_and_judges_once(monkeypatch):
    import storyforge.rag.generation as generation_mod
    from storyforge.rag.agentic_loop import run_agentic_story_loop

    evaluation_mod = _patch_chain(monkeypatch)
    judged = []
    monkeypatch.setattr(
        evaluation_mod, "evaluate_story_text",
        lambda model, story, facts="": judged.append(story) or {"faithfulness": 8, "conclusion": "ok"},
    )
    calls = []

    def fake_generate(query, parsed, raw, cfg, **kw):
        calls.append(kw.get("rewrite_sections"))
        if len(calls) == 1:
            return _story([_body(), _body(), _body(), _body(), _body(1)])  # section 5 too short
        return _story([_body()] * 5)

    monkeypatch.setattr(generation_mod, "generate_from_facts", fake_generate)
    result = run_agentic_story_loop("q", cfg=_loop_cfg())

    assert result.accepted and result.stop_reason == "accepted"
    assert calls == [None, frozenset({5})]
    assert len(judged) == 1
    assert [i["action"] for i in result.iterations] == ["write", "rewrite"]
    assert result.iterations[-1]["faithfulness"] == 8


def test_rule_loop_stops_after_max_rounds_and_keeps_best(monkeypatch):
    import storyforge.rag.generation as generation_mod
    from storyforge.rag.agentic_loop import run_agentic_story_loop

    evaluation_mod = _patch_chain(monkeypatch)
    monkeypatch.setattr(evaluation_mod, "evaluate_story_text", lambda *a, **k: {})
    bad = _story([_body(), _body(), _body(), _body(), _body(1)])
    calls = []
    monkeypatch.setattr(generation_mod, "generate_from_facts", lambda *a, **k: calls.append(1) or bad)

    result = run_agentic_story_loop("q", cfg=_loop_cfg(Agentic_loop_rule_rounds=2))

    assert not result.accepted and result.stop_reason == "rules_max_rounds"
    assert len(calls) == 3  # one write + two rewrite rounds
    assert result.content == bad


def test_rule_driven_flag_off_or_wrong_method_uses_old_loop():
    from storyforge.rag.agentic_loop import _rule_driven_enabled

    assert not _rule_driven_enabled({"Agentic_loop_rule_driven": False, "Story_generation_method": "arc"})
    assert not _rule_driven_enabled({"Agentic_loop_rule_driven": True, "Story_generation_method": "5w1h"})
    assert _rule_driven_enabled({"Agentic_loop_rule_driven": True, "Story_generation_mode": "sectioned"})


def test_sentence_starting_words_are_not_novel_names():
    starters = "Every night Alana walked home. Yet the village slept. Now the wind rose. They listened. You could hear it."
    story = _story([_body(), starters + " " + _body(), _body(), _body(), _body()])
    assert 2 not in section_failures(story, _facts(), PROFILE)


def test_trim_overlong_cuts_to_whole_sentences_and_keeps_other_sections():
    from storyforge.rag.section_rules import trim_overlong

    story = _story([_body(), _body(40), _body()])
    out = trim_overlong(story, PROFILE)
    from storyforge.rag.length_profile import split_section_bodies

    bodies = split_section_bodies(out)
    assert len(bodies[2].split()) <= PROFILE.words_per_section * 1.5
    assert bodies[2].endswith(".") and bodies[1] == _body() and bodies[3] == _body()
    assert section_failures(out, _facts(), PROFILE, sections=3) == {}
