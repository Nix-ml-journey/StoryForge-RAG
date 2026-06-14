from storyforge.rag.generation_backend import (
    generation_provider,
    ollama_base_url,
    ollama_model_id,
    use_ollama_for_generation,
)


def test_generation_provider_defaults_to_ollama():
    assert generation_provider({}) == "ollama"
    assert use_ollama_for_generation({}) is True


def test_generation_provider_transformers_alias():
    cfg = {"Generation_provider": "transformers"}
    assert generation_provider(cfg) == "transformers"
    assert use_ollama_for_generation(cfg) is False


def test_ollama_model_and_base_url_from_config():
    cfg = {
        "Generative_model": "qwen3.5:9b",
        "Ollama_base_url": "http://127.0.0.1:11434/",
    }
    assert ollama_model_id(cfg) == "qwen3.5:9b"
    assert ollama_base_url(cfg) == "http://127.0.0.1:11434"
