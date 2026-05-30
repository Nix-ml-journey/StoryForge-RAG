from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from storyforge.config.config import load_config


class Gen_mode(str, Enum):
    FAST = "fast"
    THINKING = "thinking"


class StoryType(str, Enum):
    SINGLE = "single"
    SERIES = "series"
    MIX = "mix"


def parse_gen_mode(value: Optional[str]) -> Gen_mode:
    v = (value or "").strip().lower()
    if v in {"thinking", "think", "slow"}:
        return Gen_mode.THINKING
    return Gen_mode.FAST


def parse_story_type(value: Optional[str]) -> StoryType:
    v = (value or "").strip().lower()
    if v in {"single", "standalone"}:
        return StoryType.SINGLE
    if v in {"series", "chapter", "chapters"}:
        return StoryType.SERIES
    return StoryType.MIX


def get_mode_sampling(mode: Gen_mode) -> Tuple[float, float]:
    """Default (temperature, top_p) for API responses (generation uses setup.yaml instead)."""
    if mode == Gen_mode.THINKING:
        return (0.6, 0.9)
    return (0.7, 0.95)


# Dialogue quote characters (straight and curly Unicode).
_DOUBLE_OPEN = ('"', "\u201c")
_DOUBLE_CLOSE = ('"', "\u201d")


def _close_quote_for(para: str) -> str:
    """Return the matching close-double-quote for the opening in para."""
    return "\u201d" if para and para[0] == "\u201c" else '"'


def clean_story_output(text: str) -> str:
    """
    Light cleanup applied before saving a generated story.

    - Drop short instruction text before the first [SECTION header
    - Add a closing quote on speech paragraphs that opened " but never closed
    - Insert a space after a closing quote when it touches the next word (e.g. "Leo)
    """
    text = (text or "").strip()

    # Remove leaked prompt lines above [SECTION 1
    first_section = re.search(r"\[SECTION", text, re.IGNORECASE)
    if first_section and first_section.start() > 0:
        prefix = text[: first_section.start()].strip()
        if len(re.findall(r"[.!?]", prefix)) <= 2:
            text = text[first_section.start() :]

    # Close speech lines that start with " but have no closing " on the same paragraph.
    paras = re.split(r"\n{2,}", text)
    fixed: list[str] = []
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if (
            para[0] in _DOUBLE_OPEN                          # starts with open "
            and not any(c in _DOUBLE_CLOSE for c in para[1:])  # no close " inside
            and para[-1] in ".!?,"                           # ends with punctuation
        ):
            para = para + _close_quote_for(para)
        fixed.append(para)
    text = "\n\n".join(fixed)

    text = re.sub(r'(?<=[\u201d"])(?=[A-Za-z])', " ", text)

    # Collapse double spaces from the steps above.
    text = re.sub(r"  +", " ", text)

    return text


def save_generated_story(content: str, *, filename_stem: str = "generated_story") -> Optional[Path]:
    text = clean_story_output(content)
    if not text:
        return None
    cfg = load_config()
    base = Path(str(cfg.get("BASE_PATH") or Path(__file__).resolve().parents[3]))
    out_dir = Path(str(cfg.get("Generated_story_output") or "Generated_Stories"))
    out = (base / out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"{ts}_{filename_stem}.txt"
    path.write_text(text, encoding="utf-8")
    return path


__all__ = [
    "Gen_mode",
    "StoryType",
    "parse_gen_mode",
    "parse_story_type",
    "get_mode_sampling",
    "clean_story_output",
    "save_generated_story",
]
