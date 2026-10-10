"""Per-section rule checks for the rule-driven loop (ADR-0011 Step 2).

No LLM and no GPU: a section fails on length, an unfinished sentence, or too many names that are
not in the grounded facts. The loop rewrites only the failing sections.
"""
from __future__ import annotations

import re

from storyforge.rag.attribution import attribution_violations
from storyforge.rag.length_profile import LengthProfile, sentence_count, split_section_bodies

TERMINAL = ('.', '!', '?', '"', "\u201d", "\u2019", "'")
OVERSHOOT = 1.6  # a section may run this far over its word budget
_UNDER = 0.5  # ... and must reach this share of it
_SECTIONS = 5
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def without_sentence_starts(text: str) -> str:
    """Drop each sentence's first word: 'Every', 'Now', 'They' are capitalised there, not names."""
    return " ".join(" ".join(sent.split()[1:]) for sent in _SENTENCE_END.split(text or ""))


def trim_overlong(story: str, profile: LengthProfile, factor: float = 1.5) -> str:
    """Cut a section that runs past ``factor`` x its word budget back to whole sentences (no LLM).

    Rewriting an over-long section rarely shortens it (486 -> 498 words in the 2026-10-08 run), so the
    loop trims instead. At least ``min_sentences_per_section`` sentences are always kept.
    """
    limit = int(max(1, profile.words_per_section) * factor)
    for body in split_section_bodies(story or "").values():
        if len(body.split()) <= limit:
            continue
        kept: list[str] = []
        for sent in _SENTENCE_END.split(body.strip()):
            if kept and len(kept) >= profile.min_sentences_per_section and len(" ".join(kept + [sent]).split()) > limit:
                break
            kept.append(sent)
        story = story.replace(body, " ".join(kept), 1)
    return story


def section_failures(
    story: str,
    facts: tuple,
    profile: LengthProfile,
    *,
    max_novel_names: int = 3,
    sections: int = _SECTIONS,
) -> dict[int, list[str]]:
    """Return ``{section_number: [reasons]}`` for sections that break a rule (empty = all pass)."""
    present = {int(n) for n in re.findall(r"\[SECTION\s+(\d+)", story or "", flags=re.IGNORECASE)}
    bodies = split_section_bodies(story or "")
    budget = max(1, profile.words_per_section)
    failures: dict[int, list[str]] = {}
    for i in range(1, sections + 1):
        why: list[str] = []
        body = (bodies.get(i) or "").strip()
        if i not in present or not body:
            failures[i] = ["missing"]
            continue
        words = len(body.split())
        if not body.endswith(TERMINAL):
            why.append("does not end on a finished sentence")
        if sentence_count(body) < profile.min_sentences_per_section:
            why.append(f"only {sentence_count(body)} sentences (need {profile.min_sentences_per_section})")
        if words > OVERSHOOT * budget:
            why.append(f"{words} words, over the {budget}-word budget")
        elif words < _UNDER * budget:
            why.append(f"{words} words, under the {budget}-word budget")
        novel = sorted(attribution_violations(story=without_sentence_starts(body), facts=facts))
        if len(novel) > max_novel_names:
            why.append("names not in the facts: " + ", ".join(novel[:6]))
        if why:
            failures[i] = why
    return failures


def failure_feedback(failures: dict[int, list[str]], words_per_section: int) -> str:
    """Rewrite instructions naming each failing section and why."""
    lines = [f"SECTION {i}: " + "; ".join(why) for i, why in sorted(failures.items())]
    lines.append(
        f"Aim for about {words_per_section} words per section, finish every sentence, "
        "and use only people, places and events from the GROUNDED_FACTS."
    )
    return "\n".join(lines)
