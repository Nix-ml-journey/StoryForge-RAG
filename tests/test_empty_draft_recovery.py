"""Tests for empty-draft recovery in generation.py and langchain_rag.py.

Covers:
  - thinking mode returning empty → automatic fast-mode retry
  - still empty after retry → RuntimeError with clear message
  - length guard refine returning empty → falls back to original draft
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Auto-stub heavy deps (same pattern as test_extraction.py)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _auto_stub(stub_heavy_deps):
    """Ensure langchain / HF / torch stubs are active for every test here."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> dict:
    base = {
        "Generation_provider": "ollama",
        "Generative_model": "qwen3.5:9b",
        "Ollama_base_url": "http://localhost:11434",
        "Generation_fast_temperature": 0.4,
        "Generation_fast_top_p": 0.8,
        "Generation_thinking_temperature": 0.35,
        "Generation_thinking_top_p": 0.85,
        "Single_pass_thinking_max_tokens": 4000,
        "Single_pass_fast_max_tokens": 3200,
        "Generation_repetition_penalty": 1.08,
        "Story_length_presets": {},
    }
    base.update(overrides)
    return base


_GOOD_STORY = """[SECTION 1: WHO, WHERE, WHEN (The Setup)]
The warrior stood at the gate. It was a cold morning. He had walked far.

[SECTION 2: WHAT (The Problem Starts)]
The gate was locked. No one answered. He knocked three times with no response.

[SECTION 3: TWIST/COMPLICATION (The Challenge)]
A shadow moved beyond the wall. It was not what he expected to find here.

[SECTION 4: HOW (The Big Action/Climax)]
He climbed over. The courtyard was empty but warm fires burned inside.

[SECTION 5: WHY/OUTCOME (The Moral and Conclusion)]
He found what he sought. The journey had changed him. He walked back a different man."""


# ---------------------------------------------------------------------------
# generate_from_facts — empty-draft recovery
# ---------------------------------------------------------------------------

def test_thinking_mode_empty_retries_with_fast(monkeypatch):
    """When thinking mode returns empty, generate_from_facts retries once with fast mode."""
    call_log = []

    class _FakeLLM:
        def __init__(self, mode_label):
            self._label = mode_label
        def invoke(self, prompt):
            call_log.append(self._label)
            return "" if self._label == "thinking" else _GOOD_STORY

    def _fake_load_llm(cfg, *, mode=None, max_new_tokens=None, profile=None):
        from storyforge.rag.length_profile import is_thinking_mode as itm
        label = "thinking" if itm(mode) else "fast"
        return _FakeLLM(label)

    from storyforge.rag.attribution import ParsedFacts
    parsed = ParsedFacts(facts=(), raw={})

    with patch("storyforge.rag.generation._load_generation_llm", side_effect=_fake_load_llm), \
         patch("storyforge.rag.generation._apply_attribution_gate", side_effect=lambda s, *a, **k: s), \
         patch("storyforge.rag.generation.resolve_length_profile") as mock_profile:
        mock_profile.return_value = MagicMock(
            max_new_tokens=3200, min_words=100, target_words=450,
            words_per_section=90, min_sentences_per_section=2, name="short",
        )
        from storyforge.rag.generation import generate_from_facts
        result = generate_from_facts("A warrior", parsed, "{}", _cfg(), mode="thinking")

    assert call_log == ["thinking", "fast"], f"Expected thinking then fast retry, got {call_log}"
    assert "SECTION 1" in result


def test_both_attempts_empty_raises_runtime_error(monkeypatch):
    """If both thinking and fast-mode attempts return empty, RuntimeError is raised."""

    class _EmptyLLM:
        def invoke(self, prompt):
            return ""

    from storyforge.rag.attribution import ParsedFacts
    parsed = ParsedFacts(facts=(), raw={})

    with patch("storyforge.rag.generation._load_generation_llm", return_value=_EmptyLLM()), \
         patch("storyforge.rag.generation.resolve_length_profile") as mock_profile:
        mock_profile.return_value = MagicMock(
            max_new_tokens=3200, min_words=100, target_words=450,
            words_per_section=90, min_sentences_per_section=2, name="short",
        )
        from storyforge.rag.generation import generate_from_facts
        with pytest.raises(RuntimeError, match="empty draft"):
            generate_from_facts("A warrior", parsed, "{}", _cfg(), mode="thinking")


def test_fast_mode_empty_raises_runtime_error(monkeypatch):
    """Fast mode returning empty also raises RuntimeError (no retry loop for fast)."""

    class _EmptyLLM:
        def invoke(self, prompt):
            return ""

    from storyforge.rag.attribution import ParsedFacts
    parsed = ParsedFacts(facts=(), raw={})

    with patch("storyforge.rag.generation._load_generation_llm", return_value=_EmptyLLM()), \
         patch("storyforge.rag.generation.resolve_length_profile") as mock_profile:
        mock_profile.return_value = MagicMock(
            max_new_tokens=3200, min_words=100, target_words=450,
            words_per_section=90, min_sentences_per_section=2, name="short",
        )
        from storyforge.rag.generation import generate_from_facts
        with pytest.raises(RuntimeError, match="empty draft"):
            generate_from_facts("A warrior", parsed, "{}", _cfg(), mode="fast")


# ---------------------------------------------------------------------------
# langchain_rag.py — length guard fallback
# ---------------------------------------------------------------------------

def test_length_guard_falls_back_to_original_when_refine_raises(monkeypatch):
    """If the length guard refine call raises RuntimeError, the original draft is returned."""
    short_story = "Too short."
    call_count = [0]

    def _fake_generate_from_facts(query, parsed, grounded_raw, cfg, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: return a story that is too short to pass the length guard
            return short_story
        # Second call (refine): raise as if empty draft
        raise RuntimeError("Story generation returned an empty draft.")

    from storyforge.rag.attribution import ParsedFacts
    parsed = ParsedFacts(facts=(), raw={})

    profile_mock = MagicMock(
        max_new_tokens=3200, min_words=300, target_words=450,
        words_per_section=90, min_sentences_per_section=3, name="short",
    )

    with patch("storyforge.rag.langchain_rag.generate_from_facts", side_effect=_fake_generate_from_facts), \
         patch("storyforge.rag.langchain_rag.retrieve_docs", return_value=[]), \
         patch("storyforge.rag.langchain_rag.extract_grounded_facts", return_value=("{}", parsed)), \
         patch("storyforge.rag.langchain_rag.resolve_length_profile", return_value=profile_mock), \
         patch("storyforge.rag.langchain_rag._docs_to_chunks", return_value=[]), \
         patch("storyforge.rag.langchain_rag._docs_to_context", return_value=""):
        from storyforge.rag.langchain_rag import generate_story_3step_langchain
        result = generate_story_3step_langchain("A warrior", cfg=_cfg(), show_progress=False)

    assert result.content == short_story, (
        f"Expected fallback to original short story, got: {result.content!r}"
    )
    assert call_count[0] == 2, f"Expected 2 calls (initial + refine), got {call_count[0]}"
