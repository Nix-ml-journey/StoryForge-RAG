"""Tests for Step 2 grounded-facts extraction (extraction.py).

Covers:
  - JSON mode adds response_format to HF chat_completion kwargs
  - JSON mode gracefully falls back when backend raises TypeError
  - JSON mode disabled by config
  - extract_grounded_facts returns (raw_str, ParsedFacts) with a stub HF client
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Auto-stub all heavy ML/GPU deps for every test in this module.
# stub_heavy_deps is defined in conftest.py.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _auto_stub(stub_heavy_deps):
    """Ensure langchain / HF / torch stubs are active for every test here."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> dict:
    base = {
        "HF_grounded_facts_model": "Qwen/Qwen3-8B",
        "HF_grounded_facts_max_new_tokens": 128,
        "HF_grounded_facts_temperature": 0.1,
        "HF_grounded_facts_json_mode": True,
        "facehugging_api": "hf-test-token",
    }
    base.update(overrides)
    return base


def _fake_hf_response(content: str):
    """Build a minimal InferenceClient response object."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# _hf_chat_extract_json --- JSON mode
# ---------------------------------------------------------------------------

def test_json_mode_adds_response_format_to_request(monkeypatch):
    """When HF_grounded_facts_json_mode is true, response_format must be sent."""
    captured: dict = {}

    def _fake_chat_completion(**kwargs):
        captured.update(kwargs)
        return _fake_hf_response('{"facts":[]}')

    fake_client = MagicMock()
    fake_client.chat_completion.side_effect = _fake_chat_completion

    with patch("storyforge.rag.extraction.InferenceClient", return_value=fake_client):
        from storyforge.rag.extraction import _hf_chat_extract_json
        _hf_chat_extract_json(cfg=_cfg(), system="sys", user="usr")

    assert captured.get("response_format") == {"type": "json_object"}


def test_json_mode_disabled_by_config(monkeypatch):
    """When HF_grounded_facts_json_mode is false, no response_format is sent."""
    captured: dict = {}

    def _fake_chat_completion(**kwargs):
        captured.update(kwargs)
        return _fake_hf_response('{"facts":[]}')

    fake_client = MagicMock()
    fake_client.chat_completion.side_effect = _fake_chat_completion

    with patch("storyforge.rag.extraction.InferenceClient", return_value=fake_client):
        from storyforge.rag.extraction import _hf_chat_extract_json
        _hf_chat_extract_json(cfg=_cfg(HF_grounded_facts_json_mode=False), system="sys", user="usr")

    assert "response_format" not in captured


def test_json_mode_falls_back_on_type_error(monkeypatch):
    """If the backend rejects response_format with TypeError, retry without it."""
    call_count = 0

    def _fake_chat_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if "response_format" in kwargs:
            raise TypeError("unexpected keyword argument 'response_format'")
        return _fake_hf_response('{"facts":[]}')

    fake_client = MagicMock()
    fake_client.chat_completion.side_effect = _fake_chat_completion

    with patch("storyforge.rag.extraction.InferenceClient", return_value=fake_client):
        from storyforge.rag.extraction import _hf_chat_extract_json
        result = _hf_chat_extract_json(cfg=_cfg(), system="sys", user="usr")

    assert call_count == 2                      # first attempt + fallback
    assert '"facts"' in result                  # raw JSON contains facts key
    assert result.strip().startswith("{")       # raw is a JSON object, not an error string
    assert result.strip().endswith("}")


def test_hf_chat_extract_json_raises_without_token():
    """Missing HF token must raise ValueError immediately."""
    from storyforge.rag.extraction import _hf_chat_extract_json
    with pytest.raises(ValueError, match="Missing Hugging Face token"):
        _hf_chat_extract_json(cfg={"facehugging_api": ""}, system="sys", user="usr")


# ---------------------------------------------------------------------------
# extract_grounded_facts --- public API
# ---------------------------------------------------------------------------

def test_extract_grounded_facts_returns_raw_and_parsed(monkeypatch):
    """extract_grounded_facts returns (raw_str, ParsedFacts) with at least one fact."""
    facts_payload = json.dumps({
        "facts": [
            {
                "type": "who",
                "fact": "Alana is a warrior",
                "source_chunk_ids": ["c1"],
                "quote": "Alana, a warrior",
            }
        ]
    })

    def _fake_hf(**kwargs):
        return _fake_hf_response(facts_payload)

    fake_client = MagicMock()
    fake_client.chat_completion.side_effect = _fake_hf

    with patch("storyforge.rag.extraction.InferenceClient", return_value=fake_client):
        from storyforge.rag.extraction import extract_grounded_facts
        raw, parsed = extract_grounded_facts(
            "Who is Alana?",
            [{"chunk_id": "c1", "content": "Alana, a warrior, stood tall.", "Title": "Saga"}],
            _cfg(),
        )

    assert "Alana" in raw
    assert len(parsed.facts) == 1
    assert parsed.facts[0].fact == "Alana is a warrior"
    assert parsed.facts[0].source_chunk_ids == ("c1",)


def test_extract_grounded_facts_falls_back_on_hf_error(monkeypatch):
    """When the HF call fails, the function should fall back to local extraction."""
    from storyforge.rag.extraction import extract_grounded_facts

    # Make HF always raise
    with patch("storyforge.rag.extraction.InferenceClient", side_effect=RuntimeError("HF down")):
        # Mock the local fallback (load_ollama_llm path)
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = '{"facts":[]}'

        with patch("storyforge.rag.extraction._load_facts_llm", return_value=fake_llm):
            raw, parsed = extract_grounded_facts(
                "Who is Alana?",
                [{"chunk_id": "c1", "content": "Alana walked.", "Title": "Test"}],
                _cfg(),
            )

    # raw should be the JSON string returned by the fallback LLM
    assert isinstance(raw, str)
    assert raw.strip().startswith("{")
    assert raw.strip().endswith("}")
    assert '"facts"' in raw
