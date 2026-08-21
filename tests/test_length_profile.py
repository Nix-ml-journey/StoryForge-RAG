"""Tests for the story length target.

Pure arithmetic and config resolution — no GPU, no API, no LangChain.
"""
from __future__ import annotations

from storyforge.rag.length_profile import (
    BUILTIN_LENGTH_PRESETS,
    is_thinking_mode,
    length_token_cap,
    resolve_length_profile,
)

# Presets small enough to reason about, plus a token floor that never interferes.
CFG = {
    "Story_length_presets": {"short": 450, "long": 1500, "epic": 2200},
    "Story_length_default_fast": "short",
    "Story_length_default_thinking": "long",
    "Story_length_words_per_minute": 140,
    "Story_length_max_new_tokens_cap": 6000,
    "Single_pass_fast_max_tokens": 0,
    "Single_pass_thinking_max_tokens": 0,
}


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

def test_explicit_length_wins_over_mode_default():
    profile = resolve_length_profile(CFG, length="epic", mode="fast")
    assert profile.name == "epic"
    assert profile.target_words == 2200


def test_mode_default_applies_when_no_length_requested():
    assert resolve_length_profile(CFG, mode="fast").target_words == 450
    assert resolve_length_profile(CFG, mode="thinking").target_words == 1500


def test_duration_target_converts_narration_minutes_to_words():
    """A 12-minute script at 140 wpm needs ~1680 words."""
    profile = resolve_length_profile(CFG, length="12min", mode="thinking")
    assert profile.target_words == 1680
    assert 11.5 <= profile.estimated_minutes <= 12.5


def test_word_count_target_accepts_string_and_int():
    assert resolve_length_profile(CFG, length="1800").target_words == 1800
    assert resolve_length_profile(CFG, length=1800).target_words == 1800


def test_preset_aliases_resolve_to_canonical_presets():
    assert resolve_length_profile(CFG, length="xl").target_words == 2200
    assert resolve_length_profile(CFG, length="  LONG  ").target_words == 1500


def test_unparseable_target_falls_back_to_mode_default():
    profile = resolve_length_profile(CFG, length="enormous", mode="thinking")
    assert profile.target_words == 1500


def test_empty_config_still_resolves_a_usable_target():
    profile = resolve_length_profile({}, mode="fast")
    assert profile.target_words == BUILTIN_LENGTH_PRESETS["short"]
    assert profile.max_new_tokens > 0
    assert profile.min_sentences_per_section >= 3


def test_config_presets_override_builtins():
    cfg = dict(CFG, Story_length_presets={"long": 5000})
    assert resolve_length_profile(cfg, length="long").target_words == 5000


# ---------------------------------------------------------------------------
# Derived numbers stay mutually consistent
# ---------------------------------------------------------------------------

def test_longer_target_raises_prompt_guidance_tokens_and_gates_together():
    """The whole point: one target moves all three levers in the same direction.

    Raising only the token budget was what previously left output stuck at ~450
    words, because the prompt and the accept gate never moved with it.
    """
    short = resolve_length_profile(CFG, length="short")
    epic = resolve_length_profile(CFG, length="epic")

    assert epic.words_per_section > short.words_per_section
    assert epic.sentences_low > short.sentences_low
    assert epic.min_sentences_per_section > short.min_sentences_per_section
    assert epic.min_words > short.min_words
    assert epic.max_new_tokens > short.max_new_tokens


def test_accept_gate_sits_below_the_target_but_above_half():
    """A near-target draft must pass; a half-length draft must not."""
    profile = resolve_length_profile(CFG, length="long")
    assert profile.min_words < profile.target_words
    assert profile.min_words > profile.target_words // 2


def test_token_budget_covers_the_word_target_with_headroom():
    profile = resolve_length_profile(CFG, length="epic")
    # English prose runs ~1.35 tokens/word; the budget must clear that.
    assert profile.max_new_tokens > profile.target_words * 1.35


def test_token_budget_never_drops_below_the_configured_floor():
    """Single_pass_*_max_tokens stays a floor so manual tuning is not lost."""
    cfg = dict(CFG, Single_pass_fast_max_tokens=3200)
    assert resolve_length_profile(cfg, length="short", mode="fast").max_new_tokens == 3200


def test_token_budget_is_clamped_by_the_cap():
    cfg = dict(CFG, Story_length_max_new_tokens_cap=2000)
    assert resolve_length_profile(cfg, length="epic").max_new_tokens == 2000


def test_sentence_range_is_ordered_and_never_below_three():
    for length in ("short", "long", "epic", "100"):
        profile = resolve_length_profile(CFG, length=length)
        assert 3 <= profile.min_sentences_per_section <= profile.sentences_low
        assert profile.sentences_low < profile.sentences_high


# ---------------------------------------------------------------------------
# Prompt guidance
# ---------------------------------------------------------------------------

def test_guidance_text_states_the_resolved_numbers():
    profile = resolve_length_profile(CFG, length="long")
    guidance = profile.guidance_text()

    assert str(profile.words_per_section) in guidance
    assert str(profile.target_words) in guidance
    assert str(profile.min_sentences_per_section) in guidance
    # Rendered as prompt bullets so it drops into the existing Rules block.
    assert all(line.startswith("- ") for line in guidance.splitlines())


def test_as_dict_reports_the_target_for_api_responses():
    payload = resolve_length_profile(CFG, length="12min").as_dict()
    assert payload["name"] == "12min"
    assert payload["target_words"] == 1680
    assert payload["estimated_minutes"] == 12.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_is_thinking_mode_accepts_enum_values_and_aliases():
    from storyforge.rag.generative_ai import Gen_mode

    assert is_thinking_mode(Gen_mode.THINKING) is True
    assert is_thinking_mode("medium") is True
    assert is_thinking_mode(Gen_mode.FAST) is False
    assert is_thinking_mode(None) is False


def test_length_token_cap_ignores_missing_and_invalid_values():
    assert length_token_cap({}) > 0
    assert length_token_cap({"Story_length_max_new_tokens_cap": "oops"}) > 0
