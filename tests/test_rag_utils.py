"""Utility-level RAG tests: generation backend helpers and story cleanup.

Merged from:
  - test_generation_backend.py  (generation_provider, ollama_model_id, etc.)
  - test_story_cleanup.py       (clean_story_output)
"""
from __future__ import annotations

from storyforge.rag.generation_backend import (
    generation_provider,
    ollama_base_url,
    ollama_model_id,
    use_ollama_for_generation,
)
from storyforge.rag.generative_ai import clean_story_output


# ---------------------------------------------------------------------------
# generation_backend helpers
# ---------------------------------------------------------------------------

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


def test_ollama_model_id_falls_back_to_default():
    assert ollama_model_id({}) == "qwen3.5:9b"


def test_ollama_base_url_strips_trailing_slash():
    cfg = {"Ollama_base_url": "http://localhost:11434/"}
    assert not ollama_base_url(cfg).endswith("/")


# ---------------------------------------------------------------------------
# clean_story_output (generative_ai)
# ---------------------------------------------------------------------------

def test_clean_story_output_closes_quote_spacing_and_duplicate_words():
    raw = (
        '[SECTION 1: WHO]\n'
        '" Hello there," she said to them them.\n\n'
        '[SECTION 2: WHAT]\n'
        'He replied "Sure"and walked on.\n'
    )
    out = clean_story_output(raw)
    assert '"Hello there,"' in out
    assert "them them" not in out
    assert '"Sure" and' in out
