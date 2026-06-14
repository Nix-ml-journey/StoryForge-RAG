from storyforge_config import load_config, resolve_config_path


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
        "Min_sentences_per_section",
        "Single_pass_refine_max_tokens",
        "Agentic_loop_refine_token_boost_thinking",
        "Agentic_loop_min_words_thinking",
    ):
        assert key in config
