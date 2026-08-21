"""Tests for config loading and prompt YAML contracts."""
from __future__ import annotations

from storyforge.config.config import load_prompts, load_config, resolve_config_path


# ---------------------------------------------------------------------------
# Prompt YAML contracts (merged from test_prompt_contracts.py)
# ---------------------------------------------------------------------------

def _generation_prompts() -> dict:
    prompts = load_prompts()
    generation = prompts.get("generation") or {}
    assert generation, "generation prompts section is required"
    return generation


def test_grounded_story_prompt_requires_full_five_section_completion():
    generation = _generation_prompts()
    story_user = str(generation.get("grounded_story_user") or "")

    assert "You MUST write all five sections." in story_user
    assert "Do not end the story until SECTION 5 reaches a" in story_user
    assert "finished, resolved final sentence." in story_user
    assert "Do NOT introduce new named characters, places, or events not in the facts above." in story_user


def test_story_prompts_take_length_guidance_from_the_length_profile():
    """Per-section length must come from the resolved target, never be hardcoded.

    A literal "3-6 sentences per section" in the prompt caps output regardless of
    the token budget, which is the bug the length profile exists to prevent.
    """
    generation = _generation_prompts()

    for key in ("grounded_story_user", "grounded_story_refine_user"):
        template = str(generation.get(key) or "")
        assert "{length_guidance}" in template, f"{key} must accept length guidance"
        assert "3-6 sentences" not in template, f"{key} must not hardcode a sentence range"
        assert "at least 3 complete sentences" not in template, (
            f"{key} must not hardcode a sentence minimum"
        )


def test_refine_prompt_prefers_continuation_over_restart():
    generation = _generation_prompts()
    refine_system = str(generation.get("grounded_story_refine_system") or "")
    refine_user = str(generation.get("grounded_story_refine_user") or "")

    assert "continue from the earliest incomplete section instead of restarting from scratch" in refine_system
    assert "prioritize completing missing/weak sections while preserving valid grounded content" in refine_user


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_resolve_config_path_honors_explicit_path(tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("BASE_PATH: custom\n", encoding="utf-8")

    assert resolve_config_path(config_path) == config_path


def test_load_config_returns_independent_copies(tmp_path):
    config_path = tmp_path / "setup.yaml"
    config_path.write_text(
        "BASE_PATH: first\nStory_input: Stories\n",
        encoding="utf-8",
    )

    first = load_config(config_path, overlay_keys=False)
    first["BASE_PATH"] = "mutated"
    second = load_config(config_path, overlay_keys=False)

    assert second["BASE_PATH"] == "first"


def test_load_config_normalizes_example_base_path_for_fresh_clone(tmp_path):
    example_path = tmp_path / "setup.example.yaml"
    example_path.write_text(
        'BASE_PATH: "C:/path/to/your/project"\nStory_input: Stories\n',
        encoding="utf-8",
    )

    config = load_config(example_path, overlay_keys=False)

    assert config["BASE_PATH"]
    assert "path/to/your/project" not in config["BASE_PATH"].replace("\\", "/")


def test_example_config_exposes_generation_precision_flag():
    config = load_config("setup.example.yaml", overlay_keys=False)
    assert "Generation_precision" in config
    assert str(config["Generation_precision"]).lower() in {"auto", "bf16", "fp16"}


def test_example_config_exposes_ollama_generation_settings():
    config = load_config("setup.example.yaml", overlay_keys=False)
    assert config.get("Generation_provider") == "ollama"
    assert config.get("Generative_model") == "qwen3.5:9b"
    assert config.get("Ollama_base_url") == "http://localhost:11434"


def test_example_config_exposes_vllm_generation_settings():
    config = load_config("setup.example.yaml", overlay_keys=False)
    assert config.get("vLLM_base_url") == "http://localhost:8001/v1"
    assert config.get("vLLM_model") == "Qwen/Qwen2.5-7B-Instruct"


def test_example_config_exposes_hf_grounded_facts_json_mode_flag():
    config = load_config("setup.example.yaml", overlay_keys=False)
    assert "HF_grounded_facts_json_mode" in config
    assert isinstance(config["HF_grounded_facts_json_mode"], bool)


def test_example_config_exposes_thinking_mode_generation_flags():
    config = load_config("setup.example.yaml", overlay_keys=False)
    for key in (
        "Generation_thinking_temperature",
        "Generation_thinking_top_p",
        "Single_pass_thinking_max_tokens",
        "Generation_repetition_penalty",
        "Generation_no_repeat_ngram_size",
        "Single_pass_refine_max_tokens",
        "Agentic_loop_refine_token_boost_thinking",
    ):
        assert key in config


def test_example_config_exposes_story_length_target_settings():
    config = load_config("setup.example.yaml", overlay_keys=False)
    for key in (
        "Story_length_presets",
        "Story_length_default_fast",
        "Story_length_default_thinking",
        "Story_length_words_per_minute",
        "Story_length_max_new_tokens_cap",
    ):
        assert key in config

    presets = config["Story_length_presets"]
    assert isinstance(presets, dict)
    # Defaults must be resolvable preset names, or every request falls back.
    assert config["Story_length_default_fast"] in presets
    assert config["Story_length_default_thinking"] in presets


def test_example_config_no_longer_hardcodes_length_gates():
    """Word/sentence minimums are derived from the length target.

    Leaving them in config would let them contradict the target, which is how
    the loop used to demand 1100 words from a prompt asking for ~450.
    """
    config = load_config("setup.example.yaml", overlay_keys=False)
    assert "Min_sentences_per_section" not in config
    assert "Agentic_loop_min_words" not in config
    assert "Agentic_loop_min_words_thinking" not in config
