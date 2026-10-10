from storyforge.evaluation import evaluation


class _FakeGemini:
    def __init__(self, content="{}"):
        self.content = content

    def invoke(self, _prompt):
        return self.content


def test_evaluate_model_prefers_huggingface_when_token_is_available(monkeypatch):
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "facehugging_api": "hf-token",
        "HF_evaluation_model": "test-hf-eval-model",
        "Evaluation_provider_priority": ["huggingface", "gemini"],
    })

    model = evaluation.evaluate_model()

    assert model["provider"] == "huggingface"
    assert model["model"] == "test-hf-eval-model"


def test_evaluate_model_reads_hf_model_from_config(monkeypatch):
    # HF model must come from config when no model_name override is passed.
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "facehugging_api": "hf-token",
        "HF_evaluation_model": "Qwen/Qwen2.5-7B-Instruct",
        "Evaluation_provider_priority": ["huggingface", "gemini"],
    })

    model = evaluation.evaluate_model()

    assert model["provider"] == "huggingface"
    assert model["model"] == "Qwen/Qwen2.5-7B-Instruct"


def test_evaluate_model_falls_back_to_gemini_when_hf_unavailable(monkeypatch):
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "facehugging_api": "",
        "Gemini_api_key": "gemini-token",
        "Gemini_evaluation_model": "gemini-2.0-flash",
        "Evaluation_provider_priority": ["huggingface", "gemini"],
    })
    monkeypatch.setattr(evaluation, "ChatGoogleGenerativeAI", lambda **_kwargs: _FakeGemini())

    model = evaluation.evaluate_model()

    assert isinstance(model, _FakeGemini)


def test_invoke_with_retry_uses_gemini_fallback_after_hf_transient_error(monkeypatch):
    hf_model = {"provider": "huggingface", "model": "test-model", "api_key": "hf-token"}

    def _raise_transient(_model, _prompt):
        raise RuntimeError("503 service unavailable")

    monkeypatch.setattr(evaluation, "_invoke_hf_with_retry", _raise_transient)
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "Gemini_evaluation_model": "gemini-2.0-flash",
        "Gemini_evaluation_fallback_model": "gemini-2.5-flash",
    })
    monkeypatch.setattr(
        evaluation, "_build_gemini_evaluator", lambda **_kwargs: _FakeGemini('{"overall": {"score": 7}}')
    )

    response = evaluation._invoke_with_retry(hf_model, "evaluate this")

    assert response == '{"overall": {"score": 7}}'


def test_parse_json_response_extracts_json_from_model_chatter():
    response = 'Here is the evaluation:\n{"overall": {"score": 8}, "suggestions": []}\nThanks.'

    parsed = evaluation._parse_json_response(response)

    assert parsed["overall"]["score"] == 8


# ---------------------------------------------------------------------------
# Ollama evaluation backend (Evaluation_mode: "ollama", alias "local")
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.mark.parametrize("mode", ["ollama", "local"])
def test_evaluate_model_returns_ollama_evaluator(monkeypatch, mode):
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "Evaluation_mode": mode,
        "Generative_model": "qwen3.5:9b",
        "Ollama_base_url": "http://localhost:11434",
        "facehugging_api": "hf-token",
        "Gemini_api_key": "gemini-token",
    })
    monkeypatch.setattr(evaluation, "_check_ollama_ready", lambda *_a: None)

    model = evaluation.evaluate_model()

    assert model == {
        "provider": "ollama",
        "model": "qwen3.5:9b",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
    }


def test_ollama_evaluator_model_override(monkeypatch):
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "Evaluation_mode": "ollama",
        "Generative_model": "qwen3.5:9b",
        "Ollama_evaluation_model": "qwen3:4b",
    })
    monkeypatch.setattr(evaluation, "_check_ollama_ready", lambda *_a: None)

    assert evaluation.evaluate_model()["model"] == "qwen3:4b"


def test_ollama_unreachable_fails_explicitly_without_api_fallback(monkeypatch):
    import requests

    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "Evaluation_mode": "ollama",
        "facehugging_api": "hf-token",
        "Gemini_api_key": "gemini-token",
    })

    def _down(*_a, **_k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _down)
    monkeypatch.setattr(evaluation, "_build_huggingface_evaluator", lambda **_k: pytest.fail("API fallback used"))

    with pytest.raises(RuntimeError, match="Ollama is unreachable"):
        evaluation.evaluate_model()


def test_invoke_with_retry_routes_ollama_provider(monkeypatch):
    model = {"provider": "ollama", "model": "qwen3.5:9b", "base_url": "x", "temperature": 0.1}
    monkeypatch.setattr(evaluation, "_invoke_ollama_once", lambda m, p: '{"overall": {"score": 9}}')

    assert evaluation._invoke_with_retry(model, "evaluate this") == '{"overall": {"score": 9}}'

def test_story_eval_prompt_includes_facts_only_when_given():
    from storyforge.evaluation.evaluation import _story_eval_prompt

    with_facts = _story_eval_prompt("THE STORY", "1. [who] Alana is a soldier")
    without = _story_eval_prompt("THE STORY")
    assert "Grounded facts (source of truth)" in with_facts and "Alana is a soldier" in with_facts
    assert "Grounded facts (source of truth)" not in without and "{facts_section}" not in without


def test_evaluate_story_text_retries_once_on_unparseable_judge_output(monkeypatch):
    replies = iter(["I think the story is fine.", '{"overall": {"score": 8}}'])
    prompts = []

    def fake(model, prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(evaluation, "_invoke_with_retry", fake)
    data = evaluation.evaluate_story_text({"provider": "ollama"}, "story")
    assert data["overall"]["score"] == 8
    assert len(prompts) == 2 and "not valid JSON" in prompts[1]


def test_evaluate_story_text_gives_up_after_one_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation, "_invoke_with_retry", lambda m, p: calls.append(p) or "nope")
    assert evaluation.evaluate_story_text({"provider": "ollama"}, "story") == {}
    assert len(calls) == 2


def test_parse_json_response_salvages_scores_from_truncated_judge_json():
    cut = '{"faithfulness": 8, "coherence": 7, "grammar": 6, "creativity": 7, "overall": 7, "conclusion": "Good but the end'
    data = evaluation._parse_json_response(cut)
    assert data["faithfulness"]["score"] == 8 and data["overall"]["score"] == 7
    assert evaluation._parse_json_response("I cannot comply.") == {}


def test_loop_judge_uses_compact_prompt_and_carries_facts():
    from storyforge.evaluation.evaluation import _story_eval_prompt

    compact = _story_eval_prompt("THE STORY", "1. [who] Alana", compact=True)
    assert "Reply with ONLY this JSON object" in compact and "Alana" in compact and "{facts_section}" not in compact
    assert "Reply with ONLY this JSON object" not in _story_eval_prompt("THE STORY")


def test_audit_story_claims_parses_unsupported_sentences_and_keeps_none_distinct(monkeypatch):
    from storyforge.evaluation import evaluation as ev

    replies = iter([
        '{"unsupported": [{"section": 2, "sentence": "A dragon burned the mill.", "why": "no dragon in facts"}, {"section": 3}]}',
        '{"unsupported": []}',
        "not json at all",
    ])
    monkeypatch.setattr(ev, "_invoke_with_retry", lambda model, prompt: next(replies))
    found = ev.audit_story_claims("m", "story", "1. [fact] x")
    assert found == [{"section": 2, "sentence": "A dragon burned the mill.", "why": "no dragon in facts"}]
    assert ev.audit_story_claims("m", "story", "") == []
    assert ev.audit_story_claims("m", "story", "") is None
