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
