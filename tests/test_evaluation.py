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
# Local evaluation backend (Evaluation_mode: "local")
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_local_evaluator_when_configured(monkeypatch):
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "Evaluation_mode": "local",
        "Local_evaluation_model": "test-local-eval-model",
        "Local_evaluation_device": "cpu",
    })

    model = evaluation.evaluate_model()

    assert model == {
        "provider": "local",
        "model": "test-local-eval-model",
        "device": "cpu",
        "temperature": 0.1,
    }


def test_evaluate_model_local_mode_ignores_api_priority(monkeypatch):
    # Even with HF/Gemini keys present, Evaluation_mode: local must win --
    # it's a separate backend, not another entry in the API priority list.
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "Evaluation_mode": "local",
        "facehugging_api": "hf-token",
        "Gemini_api_key": "gemini-token",
        "Evaluation_provider_priority": ["huggingface", "gemini"],
    })

    model = evaluation.evaluate_model()

    assert model["provider"] == "local"


def test_invoke_with_retry_routes_local_provider_to_local_invoker(monkeypatch):
    local_model = {"provider": "local", "model": "test-model", "device": "cpu", "temperature": 0.1}
    calls = []

    def _fake_invoke_local(model, prompt):
        calls.append((model, prompt))
        return '{"overall": {"score": 9}}'

    monkeypatch.setattr(evaluation, "_invoke_local_once", _fake_invoke_local)

    response = evaluation._invoke_with_retry(local_model, "evaluate this")

    assert response == '{"overall": {"score": 9}}'
    assert calls == [(local_model, "evaluate this")]


def test_local_evaluation_falls_back_to_api_chain_on_failure(monkeypatch):
    local_model = {"provider": "local", "model": "test-model", "device": "cpu", "temperature": 0.1}

    def _raise_oom(_model, _prompt):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(evaluation, "_invoke_local_once", _raise_oom)
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "Evaluation_provider_priority": ["huggingface", "gemini"],
    })
    monkeypatch.setattr(
        evaluation, "_build_huggingface_evaluator",
        lambda **_kwargs: {"provider": "huggingface", "model": "fallback-model", "api_key": "hf-token"},
    )
    monkeypatch.setattr(
        evaluation, "_invoke_hf_with_retry", lambda *_a, **_k: '{"overall": {"score": 6}}'
    )

    response = evaluation._invoke_with_retry(local_model, "evaluate this")

    assert response == '{"overall": {"score": 6}}'


def test_local_evaluation_raises_when_local_and_all_fallbacks_fail(monkeypatch):
    local_model = {"provider": "local", "model": "test-model", "device": "cpu", "temperature": 0.1}

    monkeypatch.setattr(
        evaluation, "_invoke_local_once",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
    )
    monkeypatch.setattr(evaluation, "_cfg", lambda: {
        "Evaluation_provider_priority": ["huggingface", "gemini"],
    })
    monkeypatch.setattr(
        evaluation, "_build_huggingface_evaluator",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("no HF token")),
    )
    monkeypatch.setattr(
        evaluation, "_build_gemini_evaluator",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("no Gemini key")),
    )

    try:
        evaluation._invoke_with_retry(local_model, "evaluate this")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "Local evaluation failed" in str(e)
