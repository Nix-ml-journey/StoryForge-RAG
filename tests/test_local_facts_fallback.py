"""Tests for the LOCAL grounded-facts path (no HF): salvage parser + extraction fallback.

The HF path gets token-level JSON enforcement; the local Ollama / vLLM path
used to do a bare invoke + strict json.loads and silently return 0 facts on
any formatting slip. These tests pin the hardened behaviour with mocked local
LLMs -- no Ollama, GPU, or network needed.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from storyforge.rag.attribution import salvage_grounded_facts_json

KNOWN = ["Lovecraft__Cool_Air_chunk_3", "Lovecraft__Cool_Air_chunk_4"]
GOOD_FACT = {
    "type": "who",
    "fact": "Dr. Munoz keeps his room cold",
    "source_chunk_ids": ["Lovecraft__Cool_Air_chunk_3"],
    "quote": "the room was cold",
}


# ---------------------------------------------------------------------------
# salvage_grounded_facts_json
# ---------------------------------------------------------------------------

def test_salvage_clean_json():
    parsed, diag = salvage_grounded_facts_json(json.dumps({"facts": [GOOD_FACT]}), known_chunk_ids=KNOWN)
    assert diag["status"] == "ok"
    assert len(parsed.facts) == 1


def test_salvage_prose_and_fences_around_object():
    raw = "Sure! Here are the facts:\n```json\n" + json.dumps({"facts": [GOOD_FACT]}) + "\n```\nHope this helps."
    parsed, _ = salvage_grounded_facts_json(raw, known_chunk_ids=KNOWN)
    assert [f.fact for f in parsed.facts] == ["Dr. Munoz keeps his room cold"]


def test_salvage_bare_list():
    parsed, _ = salvage_grounded_facts_json(json.dumps([GOOD_FACT, GOOD_FACT]), known_chunk_ids=KNOWN)
    assert len(parsed.facts) == 2


def test_salvage_trailing_commas():
    raw = '{"facts": [{"type": "who", "fact": "A", "source_chunk_ids": ["Lovecraft__Cool_Air_chunk_3",],},],}'
    parsed, _ = salvage_grounded_facts_json(raw, known_chunk_ids=KNOWN)
    assert len(parsed.facts) == 1


def test_salvage_truncated_mid_object_keeps_complete_facts():
    second = dict(GOOD_FACT, fact="He dies when the cooling fails", source_chunk_ids=["Lovecraft__Cool_Air_chunk_4"])
    full = json.dumps({"facts": [GOOD_FACT, second, GOOD_FACT]})
    truncated = full[: full.rfind('"quote"') + 12]  # cut inside the 3rd object
    with pytest.raises(ValueError):
        json.loads(truncated)
    parsed, diag = salvage_grounded_facts_json(truncated, known_chunk_ids=KNOWN)
    assert diag["status"] == "salvaged"
    assert [f.fact for f in parsed.facts] == [GOOD_FACT["fact"], second["fact"]]


def test_salvage_strips_think_block():
    raw = "<think>let me reason about chunks...</think>" + json.dumps({"facts": [GOOD_FACT]})
    parsed, _ = salvage_grounded_facts_json(raw, known_chunk_ids=KNOWN)
    assert len(parsed.facts) == 1


def test_salvage_unclosed_think_is_empty_response():
    parsed, diag = salvage_grounded_facts_json("<think>reasoning that never ends because the budget ran out")
    assert parsed.facts == ()
    assert diag["status"] == "empty_response"


def test_salvage_empty_string():
    parsed, diag = salvage_grounded_facts_json("")
    assert parsed.facts == () and diag["status"] == "empty_response"


def test_salvage_resolves_abbreviated_chunk_id_by_unique_suffix():
    fact = dict(GOOD_FACT, source_chunk_ids=["chunk_4"])
    parsed, _ = salvage_grounded_facts_json(json.dumps({"facts": [fact]}), known_chunk_ids=KNOWN)
    assert parsed.facts[0].source_chunk_ids == ("Lovecraft__Cool_Air_chunk_4",)


def test_salvage_drops_facts_citing_unknown_chunks():
    bad = dict(GOOD_FACT, source_chunk_ids=["Some_Other_Book_chunk_9"])
    parsed, diag = salvage_grounded_facts_json(json.dumps({"facts": [bad, GOOD_FACT]}), known_chunk_ids=KNOWN)
    assert len(parsed.facts) == 1
    assert diag["dropped"] == {"no_valid_source": 1}


def test_salvage_accepts_singular_source_key():
    fact = {"type": "who", "fact": "X", "source_chunk_id": "Lovecraft__Cool_Air_chunk_3"}
    parsed, _ = salvage_grounded_facts_json(json.dumps({"facts": [fact]}), known_chunk_ids=KNOWN)
    assert parsed.facts[0].source_chunk_ids == ("Lovecraft__Cool_Air_chunk_3",)


def test_salvage_without_known_ids_keeps_any_cited_source():
    fact = dict(GOOD_FACT, source_chunk_ids=["anything"])
    parsed, _ = salvage_grounded_facts_json(json.dumps({"facts": [fact]}))
    assert len(parsed.facts) == 1


def test_salvage_pure_prose_reports_no_json():
    parsed, diag = salvage_grounded_facts_json("Dr. Munoz keeps his room cold. That is all.")
    assert parsed.facts == () and diag["status"] == "no_json"


# ---------------------------------------------------------------------------
# extract_grounded_facts -- local path
# ---------------------------------------------------------------------------

CHUNKS = [
    {"chunk_id": "Lovecraft__Cool_Air_chunk_3", "title": "Lovecraft__Cool_Air", "text": "the room was cold"},
    {"chunk_id": "Lovecraft__Cool_Air_chunk_4", "title": "Lovecraft__Cool_Air", "text": "the pump failed"},
]


def _local_cfg(**over):
    cfg = {
        "Grounded_facts_provider": "local",
        "Generation_provider": "ollama",
        "HF_grounded_facts_max_new_tokens": 3200,
    }
    cfg.update(over)
    return cfg


@pytest.fixture
def extraction(stub_heavy_deps):
    import storyforge.rag.extraction as extraction_mod
    return extraction_mod


def _fake_llm(*responses):
    llm = MagicMock()
    llm.invoke.side_effect = list(responses)
    return llm


def test_local_provider_never_calls_hf(extraction):
    llm = _fake_llm(json.dumps({"facts": [GOOD_FACT]}))
    with patch.object(extraction, "InferenceClient", side_effect=AssertionError("HF must not be called")), \
         patch.object(extraction, "_load_facts_llm", return_value=llm):
        raw, parsed = extraction.extract_grounded_facts("cold room", CHUNKS, _local_cfg())
    assert len(parsed.facts) == 1
    prompt = llm.invoke.call_args[0][0]
    assert "OUTPUT FORMAT (strict)" in prompt
    assert "[CHUNK Lovecraft__Cool_Air_chunk_3 | Lovecraft__Cool_Air]" in prompt


def test_hf_failure_falls_back_to_hardened_local_path(extraction):
    messy = "Here you go:\n```json\n" + json.dumps({"facts": [GOOD_FACT]}) + "\n```"
    llm = _fake_llm(messy)
    with patch.object(extraction, "InferenceClient", side_effect=RuntimeError("402 Payment Required")), \
         patch.object(extraction, "_load_facts_llm", return_value=llm):
        raw, parsed = extraction.extract_grounded_facts(
            "cold room", CHUNKS, _local_cfg(Grounded_facts_provider="hf", facehugging_api="tok")
        )
    assert len(parsed.facts) == 1
    assert llm.invoke.call_count == 1  # HF error goes straight to the local path, no extra local retry


def test_local_retries_once_with_compact_prompt_after_zero_facts(extraction):
    llm = _fake_llm('{"facts": [{"type": "who", "fact": "cut off', json.dumps({"facts": [GOOD_FACT]}))
    with patch.object(extraction, "_load_facts_llm", return_value=llm):
        raw, parsed = extraction.extract_grounded_facts("cold room", CHUNKS, _local_cfg())
    assert len(parsed.facts) == 1
    assert llm.invoke.call_count == 2
    second_prompt = llm.invoke.call_args_list[1][0][0]
    assert "AT MOST 12 facts" in second_prompt


def test_local_failure_logs_error_and_returns_empty(extraction, caplog):
    llm = _fake_llm("I cannot find facts.", "Still prose.")
    with patch.object(extraction, "_load_facts_llm", return_value=llm), \
         caplog.at_level(logging.ERROR, logger="storyforge.rag.extraction"):
        raw, parsed = extraction.extract_grounded_facts("cold room", CHUNKS, _local_cfg())
    assert parsed.facts == ()
    assert raw == "Still prose."
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and "LOCAL grounded-facts extraction FAILED" in errors[0].getMessage()
    assert "no_json" in errors[0].getMessage()


def test_local_retries_can_be_disabled(extraction):
    llm = _fake_llm("nope")
    with patch.object(extraction, "_load_facts_llm", return_value=llm):
        _, parsed = extraction.extract_grounded_facts("q", CHUNKS, _local_cfg(Local_grounded_facts_retries=0))
    assert parsed.facts == () and llm.invoke.call_count == 1


def test_local_json_format_defaults_to_schema_and_falls_back_when_rejected(extraction):
    calls = []

    def _loader(cfg, *, json_format=None):
        calls.append(json_format)
        if json_format:
            broken = MagicMock()
            broken.invoke.side_effect = RuntimeError("unsupported format")
            return broken
        return _fake_llm(json.dumps({"facts": [GOOD_FACT]}))

    with patch.object(extraction, "_load_facts_llm", side_effect=_loader):
        _, parsed = extraction.extract_grounded_facts("q", CHUNKS, _local_cfg())
    assert len(parsed.facts) == 1
    assert calls[0] == extraction.FACTS_JSON_SCHEMA
    assert calls[1] is None


@pytest.mark.parametrize("setting, expected", [("json", "json"), ("off", None), (None, "SCHEMA")])
def test_local_json_format_config(extraction, setting, expected):
    got = extraction._local_json_format({"Local_grounded_facts_json_format": setting})
    assert got == (extraction.FACTS_JSON_SCHEMA if expected == "SCHEMA" else expected)


def test_load_facts_llm_passes_schema_to_ollama(extraction):
    with patch.object(extraction, "load_ollama_llm", return_value="LLM") as fake:
        extraction._load_facts_llm(_local_cfg(), json_format=extraction.FACTS_JSON_SCHEMA)
    kwargs = fake.call_args.kwargs
    assert kwargs["json_format"] == extraction.FACTS_JSON_SCHEMA
    assert kwargs["thinking"] is False
    assert kwargs["max_new_tokens"] == 3200


def test_build_chat_ollama_forwards_format(stub_heavy_deps):
    import langchain_ollama
    import storyforge.rag.generation_backend as gb

    captured = {}
    with patch.object(langchain_ollama, "ChatOllama", side_effect=lambda **kw: captured.update(kw) or "x", create=True):
        gb.build_chat_ollama({}, max_new_tokens=10, temperature=0.1, top_p=0.8, json_format="json")
        assert captured["format"] == "json"
        captured.clear()
        gb.build_chat_ollama({}, max_new_tokens=10, temperature=0.1, top_p=0.8)
        assert "format" not in captured
