"""Resolve a story length target into prompt guidance, tokens, and accept gates.

One word target keeps those three in sync. Accepts a preset (``"long"``),
narration duration (``"12min"``), or word count (``1800``).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "BUILTIN_LENGTH_PRESETS",
    "LengthProfile",
    "is_thinking_mode",
    "length_presets",
    "length_token_cap",
    "resolve_length_profile",
]

_SECTIONS = 5
_WORDS_PER_SENTENCE = 18
_SENTENCE_LOW_RATIO = 0.85
_SENTENCE_HIGH_RATIO = 1.2
_MIN_WORDS_RATIO = 0.85  # accept gate sits just under the target
_MIN_SENTENCE_RATIO = 0.7
_ABSOLUTE_MIN_SENTENCES = 3
_TOKENS_PER_WORD = 1.35
_TOKEN_HEADROOM = 1.35
_DEFAULT_WORDS_PER_MINUTE = 140
_DEFAULT_TOKEN_CAP = 6000
_MIN_TOKEN_CAP = 256
_MIN_TARGET_WORDS = 100

BUILTIN_LENGTH_PRESETS: dict[str, int] = {
    "short": 450,     # ~3 min
    "medium": 900,    # ~6 min
    "long": 1500,     # ~11 min
    "epic": 2200,     # ~16 min
}

_PRESET_ALIASES = {
    "s": "short",
    "brief": "short",
    "tiny": "short",
    "m": "medium",
    "mid": "medium",
    "standard": "medium",
    "l": "long",
    "xl": "epic",
    "xlong": "epic",
    "verylong": "epic",
    "very_long": "epic",
}

_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)$")

# mode="medium" means thinking; length="medium" means the ~900-word preset.
_THINKING_MODE_NAMES = frozenset({"thinking", "think", "slow", "medium"})


def is_thinking_mode(mode: Any) -> bool:
    """True for thinking / medium / slow mode values."""
    return str(getattr(mode, "value", mode) or "").strip().lower() in _THINKING_MODE_NAMES


@dataclass(frozen=True)
class LengthProfile:
    """Every length-related number for one generation request."""

    name: str
    target_words: int
    words_per_section: int
    sentences_low: int
    sentences_high: int
    min_sentences_per_section: int
    min_words: int
    max_new_tokens: int
    words_per_minute: int

    @property
    def estimated_minutes(self) -> float:
        """Narration time at ``words_per_minute``, for video-script targets."""
        return round(self.target_words / max(1, self.words_per_minute), 1)

    def guidance_text(self) -> str:
        """The ``{length_guidance}`` block injected into the story prompts."""
        return "\n".join(
            [
                f"- LENGTH TARGET: write about {self.words_per_section} words under EACH section"
                f" header, roughly {self.sentences_low}-{self.sentences_high} sentences.",
                f"- EACH section MUST contain at least {self.min_sentences_per_section} complete"
                " sentences before moving on to the next section.",
                f"- Whole-story target: about {self.target_words} words"
                f" (~{self.estimated_minutes} minutes read aloud). Do not stop early.",
                "- Reach the target with new grounded detail, action, and dialogue, never by"
                " repeating sentences or padding with filler.",
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        """Compact summary for API ``gen_params`` responses."""
        return {
            "name": self.name,
            "target_words": self.target_words,
            "min_words": self.min_words,
            "words_per_section": self.words_per_section,
            "sentences_per_section": f"{self.sentences_low}-{self.sentences_high}",
            "min_sentences_per_section": self.min_sentences_per_section,
            "max_new_tokens": self.max_new_tokens,
            "estimated_minutes": self.estimated_minutes,
        }


def length_presets(cfg: Optional[dict[str, Any]]) -> dict[str, int]:
    """Built-in presets overlaid with ``Story_length_presets`` from config."""
    presets = dict(BUILTIN_LENGTH_PRESETS)
    configured = (cfg or {}).get("Story_length_presets")
    if isinstance(configured, dict):
        for raw_name, raw_words in configured.items():
            try:
                words = int(raw_words)
            except (TypeError, ValueError):
                continue
            if words > 0:
                presets[str(raw_name).strip().lower()] = words
    return presets


def length_token_cap(cfg: Optional[dict[str, Any]]) -> int:
    """Hard ceiling on max_new_tokens (including refine boosts)."""
    try:
        cap = int((cfg or {}).get("Story_length_max_new_tokens_cap") or _DEFAULT_TOKEN_CAP)
    except (TypeError, ValueError):
        cap = _DEFAULT_TOKEN_CAP
    return max(_MIN_TOKEN_CAP, cap)


def _words_per_minute(cfg: Optional[dict[str, Any]]) -> int:
    try:
        wpm = int((cfg or {}).get("Story_length_words_per_minute") or _DEFAULT_WORDS_PER_MINUTE)
    except (TypeError, ValueError):
        wpm = _DEFAULT_WORDS_PER_MINUTE
    return max(1, wpm)


def _parse_target(
    value: Any,
    *,
    presets: dict[str, int],
    words_per_minute: int,
) -> Optional[tuple[str, int]]:
    """Parse preset / ``12min`` / word count into ``(name, words)``, or None."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        words = int(value)
        return (f"{words}w", words) if words > 0 else None

    text = str(value).strip().lower()
    if not text:
        return None

    preset_name = _PRESET_ALIASES.get(text, text)
    if preset_name in presets:
        return preset_name, int(presets[preset_name])

    duration = _DURATION_RE.match(text)
    if duration:
        words = int(round(float(duration.group(1)) * words_per_minute))
        return (f"{duration.group(1)}min", words) if words > 0 else None

    if text.replace(".", "", 1).isdigit():
        words = int(round(float(text)))
        return (f"{words}w", words) if words > 0 else None

    return None


def _build_profile(
    name: str,
    target_words: int,
    *,
    words_per_minute: int,
    token_cap: int,
    token_floor: int = 0,
) -> LengthProfile:
    target = max(_MIN_TARGET_WORDS, int(target_words))
    words_per_section = max(20, round(target / _SECTIONS))

    sentences = words_per_section / _WORDS_PER_SENTENCE
    low = max(_ABSOLUTE_MIN_SENTENCES, math.floor(sentences * _SENTENCE_LOW_RATIO))
    high = max(low + 2, math.ceil(sentences * _SENTENCE_HIGH_RATIO))
    min_sentences = max(_ABSOLUTE_MIN_SENTENCES, round(low * _MIN_SENTENCE_RATIO))

    # Prefer the higher of derived budget vs configured floor.
    derived_tokens = math.ceil(target * _TOKENS_PER_WORD * _TOKEN_HEADROOM)
    tokens = min(max(derived_tokens, max(0, int(token_floor))), token_cap)

    return LengthProfile(
        name=str(name),
        target_words=target,
        words_per_section=words_per_section,
        sentences_low=low,
        sentences_high=high,
        min_sentences_per_section=min_sentences,
        min_words=int(round(target * _MIN_WORDS_RATIO)),
        max_new_tokens=tokens,
        words_per_minute=words_per_minute,
    )


def resolve_length_profile(
    cfg: Optional[dict[str, Any]] = None,
    *,
    length: Any = None,
    mode: Any = None,
) -> LengthProfile:
    """Resolve length → mode default → Story_length_default → built-in preset."""
    cfg = cfg or {}
    presets = length_presets(cfg)
    words_per_minute = _words_per_minute(cfg)
    token_cap = length_token_cap(cfg)
    thinking = is_thinking_mode(mode)

    token_floor_key = "Single_pass_thinking_max_tokens" if thinking else "Single_pass_fast_max_tokens"
    try:
        token_floor = int(cfg.get(token_floor_key) or cfg.get("Single_pass_fast_max_tokens") or 0)
    except (TypeError, ValueError):
        token_floor = 0

    candidates = (
        length,
        cfg.get("Story_length_default_thinking" if thinking else "Story_length_default_fast"),
        cfg.get("Story_length_default"),
        "long" if thinking else "short",
    )
    for candidate in candidates:
        parsed = _parse_target(candidate, presets=presets, words_per_minute=words_per_minute)
        if parsed:
            name, words = parsed
            return _build_profile(
                name,
                words,
                words_per_minute=words_per_minute,
                token_cap=token_cap,
                token_floor=token_floor,
            )

    return _build_profile(
        "short",
        BUILTIN_LENGTH_PRESETS["short"],
        words_per_minute=words_per_minute,
        token_cap=token_cap,
        token_floor=token_floor,
    )
