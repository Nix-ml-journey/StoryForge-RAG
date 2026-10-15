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
    vllm_base_url,
    vllm_model_id,
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


def test_generation_provider_vllm():
    cfg = {"Generation_provider": "vllm"}
    assert generation_provider(cfg) == "vllm"
    assert use_ollama_for_generation(cfg) is False


def test_vllm_model_and_base_url_from_config():
    cfg = {
        "vLLM_model": "Qwen/Qwen2.5-7B-Instruct",
        "vLLM_base_url": "http://127.0.0.1:8001/v1/",
    }
    assert vllm_model_id(cfg) == "Qwen/Qwen2.5-7B-Instruct"
    assert vllm_base_url(cfg) == "http://127.0.0.1:8001/v1"


def test_vllm_model_id_falls_back_to_generative_model():
    cfg = {"Generative_model": "Qwen/Qwen3-8B"}
    assert vllm_model_id(cfg) == "Qwen/Qwen3-8B"


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


def test_ollama_client_really_gets_num_ctx_and_repeat_penalty():
    # langchain-ollama ignores a constructor `options=` dict; both must be real fields or Ollama stays at 4096.
    from storyforge.rag.generation_backend import build_chat_ollama, ollama_num_ctx

    cfg = {"Model_max_prompt_tokens": 12288, "Generation_repetition_penalty": 1.1}
    llm = build_chat_ollama(cfg, max_new_tokens=50, temperature=0.5, top_p=0.9)
    assert llm.num_ctx == ollama_num_ctx(cfg) == 12288 and llm.repeat_penalty == 1.1


def test_prompt_longer_than_num_ctx_is_logged_not_silent(caplog):
    from storyforge.rag.generation_backend import invoke_combined_prompt

    class Inner:
        num_ctx, num_predict = 1000, 200

        def invoke(self, messages):
            return type("R", (), {"content": "ok"})()

    with caplog.at_level("WARNING"):
        assert invoke_combined_prompt(Inner(), "x" * 5000) == "ok"
        assert "may not fit num_ctx=1000" in caplog.text
        caplog.clear()
        invoke_combined_prompt(Inner(), "x" * 100)
        assert "may not fit" not in caplog.text
