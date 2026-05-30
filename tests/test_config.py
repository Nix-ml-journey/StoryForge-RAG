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
