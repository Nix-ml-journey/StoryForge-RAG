"""Opt-in sectioned writer (Story_generation_mode: sectioned): one call per section."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from storyforge.rag.agentic_loop import completeness_report
from storyforge.rag.attribution import ParsedFacts
from storyforge.rag.length_profile import resolve_length_profile


@pytest.fixture(autouse=True)
def _auto_stub(stub_heavy_deps):
    """langchain / HF / torch stubs for every test here."""


CFG = {"Generation_provider": "ollama", "Story_generation_mode": "sectioned", "Story_length_presets": {}}
SENTENCE = "Alana crossed the quiet village and listened to the wind in the old trees."


class _Scripted:
    """Fake LLM: returns the next scripted reply and records the prompt."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, cfg, **kw):
        outer = self

        class _L:
            def invoke(self, prompt):
                outer.prompts.append(prompt)
                return outer.replies.pop(0)

        return _L()


def _run(replies, **kw):
    from storyforge.rag import generation

    llm = _Scripted(replies)
    profile = resolve_length_profile(CFG, length=450)
    with patch.object(generation, "_load_generation_llm", side_effect=llm), \
         patch.object(generation, "_apply_attribution_gate", side_effect=lambda s, *a, **k: s):
        story = generation.generate_from_facts(
            "q", ParsedFacts(facts=(), raw={}), "1. [who] Alana", CFG, mode="fast", profile=profile, **kw
        )
    return story, llm, profile


def _body(n=6):
    return " ".join([SENTENCE] * n)


def test_sectioned_writes_five_sections_each_with_its_own_call():
    story, llm, profile = _run([_body() for _ in range(5)])
    assert len(llm.prompts) == 5
    assert completeness_report(story, min_words=50).ok
    assert "[SECTION 5" in story
    # later sections see the earlier ones; the final section gets the ending rule
    assert SENTENCE in llm.prompts[1] and "(nothing yet)" in llm.prompts[0]
    assert "FINAL section" in llm.prompts[4]
    assert f"about {profile.words_per_section} words" in llm.prompts[0]


def test_sectioned_strips_echoed_header_and_trims_cut_off_fragment():
    replies = ["[SECTION 1: WHO]\n" + _body(), _body(), _body(), _body(), _body() + " And then the"]
    story, _, _ = _run(replies)
    assert story.count("[SECTION 1") == 1
    assert story.endswith("old trees.")


def test_sectioned_refine_rewrites_only_missing_or_short_sections():
    from storyforge.rag.generation import _SECTION_HEADERS

    prior = "\n\n".join(f"{h}\n{_body()}" for h in _SECTION_HEADERS[:4])  # SECTION 5 missing
    story, llm, _ = _run(["The ending arrives. " + _body(5)], refine_feedback="finish it", prior_draft=prior)
    assert len(llm.prompts) == 1
    assert "[SECTION 5" in llm.prompts[0] and "finish it" in llm.prompts[0]
    assert completeness_report(story, min_words=50).ok


def test_sectioned_refine_with_all_sections_fine_rewrites_everything_with_feedback():
    from storyforge.rag.generation import _SECTION_HEADERS

    prior = "\n\n".join(f"{h}\n{_body()}" for h in _SECTION_HEADERS)
    _, llm, _ = _run([_body() for _ in range(5)], refine_feedback="add tension", prior_draft=prior)
    assert len(llm.prompts) == 5 and all("add tension" in p for p in llm.prompts)


def test_default_mode_is_single_pass(monkeypatch):
    from storyforge.rag import generation

    called = []
    monkeypatch.setattr(generation, "_generate_sectioned", lambda *a, **k: called.append(1) or "x")
    monkeypatch.setattr(generation, "_invoke_nonempty", lambda *a, **k: "single")
    monkeypatch.setattr(generation, "_apply_attribution_gate", lambda s, *a, **k: s)
    out = generation.generate_from_facts("q", ParsedFacts(facts=(), raw={}), "", {"Generation_provider": "ollama"}, mode="fast")
    assert out == "single" and not called


# ---------------------------------------------------------------------------
# Story_generation_method: arc (story-arc outline, own prompt set)
# ---------------------------------------------------------------------------

def _run_arc(replies, **kw):
    from storyforge.rag import generation

    llm = _Scripted(replies)
    cfg = {**CFG, "Story_generation_method": "arc"}
    profile = resolve_length_profile(cfg, length=450)
    with patch.object(generation, "_load_generation_llm", side_effect=llm), \
         patch.object(generation, "_apply_attribution_gate", side_effect=lambda s, *a, **k: s):
        story = generation.generate_from_facts(
            "q", ParsedFacts(facts=(), raw={}), "1. [who] Alana", cfg, mode="fast", profile=profile, **kw
        )
    return story, llm


def test_arc_method_uses_its_own_outline_prompts_and_section_roles():
    from storyforge.rag.generation_arc import ARC_HEADERS

    story, llm = _run_arc([_body() for _ in range(5)])
    assert len(llm.prompts) == 5
    assert all(h in story for h in ARC_HEADERS) and "WHO, WHERE, WHEN" not in story
    assert completeness_report(story, min_words=50).ok  # same [SECTION n: ...] shape downstream
    assert "ordinary world" in llm.prompts[0] and "turning point" in llm.prompts[3]
    assert "FINAL section" in llm.prompts[4]
    assert "five-part story arc" in llm.prompts[0]  # arc system prompt, not the 5W1H one


def test_arc_refine_rewrites_only_weak_sections():
    from storyforge.rag.generation_arc import ARC_HEADERS

    prior = "\n\n".join(f"{h}\n{_body()}" for h in ARC_HEADERS[:3])  # sections 4 and 5 missing
    _, llm = _run_arc([_body(), _body()], refine_feedback="raise the stakes", prior_draft=prior)
    assert len(llm.prompts) == 2 and "[SECTION 4: CLIMAX" in llm.prompts[0]


def test_5w1h_stays_the_default_method():
    story, _, _ = _run([_body() for _ in range(5)])
    assert "WHO, WHERE, WHEN" in story and "SETUP (The Ordinary World)" not in story


def test_arc_prompt_set_matches_the_section_placeholders():
    from storyforge.config.config import load_prompts

    arc = load_prompts()["generation_arc"]
    assert len(arc["roles"]) == 5
    needed = ("{query}", "{grounded_facts}", "{story_so_far}", "{header}", "{role}", "{words}",
              "{max_words}", "{min_sentences}", "{ending_rule}", "{feedback_block}")
    assert all(p in arc["section_user"] for p in needed)
