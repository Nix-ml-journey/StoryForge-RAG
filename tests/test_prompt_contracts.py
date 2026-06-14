from __future__ import annotations

from storyforge.config.config import load_prompts


def _generation_prompts() -> dict:
    prompts = load_prompts()
    generation = prompts.get("generation") or {}
    assert generation, "generation prompts section is required"
    return generation


def test_grounded_story_prompt_requires_full_five_section_completion():
    generation = _generation_prompts()
    story_user = str(generation.get("grounded_story_user") or "")

    assert "You MUST write all five sections." in story_user
    assert "EACH section MUST contain at least 3 complete sentences" in story_user
    assert "Do not end the story until SECTION 5 reaches a" in story_user
    assert "finished, resolved final sentence." in story_user
    assert "Do NOT introduce new named characters, places, or events not in the facts above." in story_user


def test_refine_prompt_prefers_continuation_over_restart():
    generation = _generation_prompts()
    refine_system = str(generation.get("grounded_story_refine_system") or "")
    refine_user = str(generation.get("grounded_story_refine_user") or "")

    assert "continue from the earliest incomplete section instead of restarting from scratch" in refine_system
    assert "prioritize completing missing/weak sections while preserving valid grounded content" in refine_user
