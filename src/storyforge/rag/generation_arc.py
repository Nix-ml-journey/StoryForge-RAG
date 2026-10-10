"""Alternative Step 3 method: classic story-arc outline, written one section per call.

``Story_generation_method: arc`` (default; ``5w1h`` is the older outline) swaps the WHO/WHAT/TWIST/HOW/WHY outline for
setup -> inciting incident -> rising action -> climax -> resolution, with its own prompt set
(``generation_arc`` in prompts.yaml). The section loop, length budgets, and refine rule (rewrite only
missing/short sections) are shared with the sectioned 5W1H writer in ``generation.py``;
everything downstream (completeness check, judge, loop) sees the same ``[SECTION n: ...]`` shape.
"""
from __future__ import annotations

from typing import Any, Optional

from storyforge.config.config import load_prompts
from storyforge.rag.generation import _generate_sectioned
from storyforge.rag.length_profile import LengthProfile

ARC_HEADERS = (
    "[SECTION 1: SETUP (The Ordinary World)]",
    "[SECTION 2: INCITING INCIDENT (The Disruption)]",
    "[SECTION 3: RISING ACTION (The Escalation)]",
    "[SECTION 4: CLIMAX (The Turning Point)]",
    "[SECTION 5: RESOLUTION (The Aftermath)]",
)


def _arc_prompts() -> tuple[dict[str, str], tuple[str, ...]]:
    p = (load_prompts() or {}).get("generation_arc") or {}
    prompts = {
        "section_system": p.get("section_system") or "Write one section of a grounded story arc.",
        "section_user": p.get("section_user")
        or "QUERY:\n{query}\n\nGROUNDED_FACTS:\n{grounded_facts}\n\nSTORY SO FAR:\n{story_so_far}\n\n"
        "Write ONLY this section: {header}\nIts job: {role}\nAbout {words} words. {ending_rule}\n{feedback_block}",
    }
    return prompts, tuple(str(r) for r in (p.get("roles") or ()))


def generate_arc(
    query: str,
    facts_for_prompt: str,
    cfg: dict[str, Any],
    *,
    mode: Any,
    profile: LengthProfile,
    prior_draft: Optional[str] = None,
    feedback: Optional[str] = None,
    rewrite_sections: Optional[frozenset[int]] = None,
) -> str:
    """Write (or partially rewrite) the story with the story-arc outline."""
    prompts, roles = _arc_prompts()
    return _generate_sectioned(
        query, facts_for_prompt, cfg, mode=mode, profile=profile,
        prior_draft=prior_draft, feedback=feedback,
        headers=ARC_HEADERS, prompts=prompts, roles=roles, rewrite_sections=rewrite_sections,
    )
