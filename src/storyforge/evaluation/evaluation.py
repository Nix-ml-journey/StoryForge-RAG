import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]
except ImportError:
    ChatGoogleGenerativeAI = None

from storyforge.config.config import load_config, load_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Optional overrides for tests; None = read from setup.yaml.
evaluation_provider_priority: list[str] = ["huggingface", "gemini"]
hf_evaluation_model: str | None = None  # None → HF_evaluation_model in setup.yaml
hf_api_key: str | None = None
gemini_api_key: str | None = None
gemini_evaluation_model: str = ""
gemini_evaluation_fallback_model: str = ""


def _cfg() -> dict[str, Any]:
    return load_config()


def _eval_prompts() -> dict[str, str]:
    p = load_prompts() or {}
    raw_eval = p.get("evaluation")
    ev: dict[str, Any] = raw_eval if isinstance(raw_eval, dict) else {}
    return {
        "with_story": str(ev.get("with_story") or ""),
        "with_summary": str(ev.get("with_summary") or ""),
    }

EVAL_RETRY_MAX_ATTEMPTS = 6
EVAL_RETRY_BASE_DELAY_SEC = 2
EVAL_RETRY_BACKOFF_FACTOR = 2


def _is_retryable_api_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "503" in msg
        or "unavailable" in msg
        or "high demand" in msg
        or "429" in msg
        or "rate limit" in msg
        or "resource exhausted" in msg
    )


def _normalise_provider_priority(value) -> list[str]:
    if isinstance(value, str):
        raw = [p.strip() for p in value.split(",")]
    elif isinstance(value, (list, tuple)):
        raw = [str(p).strip() for p in value]
    else:
        raw = ["huggingface", "gemini"]
    providers = [p.lower() for p in raw if p]
    return providers or ["huggingface", "gemini"]


def _build_huggingface_evaluator(temperature: float = 0.3, model_name: Optional[str] = None) -> dict[str, Any]:
    cfg = _cfg()
    if hf_api_key is None:
        token = str(cfg.get("facehugging_api") or "").strip()
    else:
        token = str(hf_api_key).strip()
    if not token:
        raise ValueError("Hugging Face API key is not set for evaluation")
    model_id = (
        model_name
        or (hf_evaluation_model if hf_evaluation_model is not None else "")
        or str(cfg.get("HF_evaluation_model") or "").strip()
        or "Qwen/Qwen2.5-1.5B-Instruct"
    )
    if not model_id:
        raise ValueError("Hugging Face evaluation model is not set in the configuration")
    logging.info("Initialized StoryEvaluator with Hugging Face model: %s", model_id)
    return {
        "provider": "huggingface",
        "model": model_id,
        "api_key": token,
        "temperature": float(temperature if temperature is not None else cfg.get("HF_evaluation_temperature", 0.1)),
    }


def _build_gemini_evaluator(temperature: float = 0.3, model_name: Optional[str] = None):
    if ChatGoogleGenerativeAI is None:
        raise ImportError("langchain-google-genai is not installed")
    cfg = _cfg()
    if gemini_api_key is None:
        key = str(cfg.get("Gemini_api_key") or "").strip()
    else:
        key = str(gemini_api_key).strip()
    if not key:
        raise ValueError("Gemini API key is not set in the configuration")
    default_name = (gemini_evaluation_model or str(cfg.get("Gemini_evaluation_model") or "")).strip()
    name = model_name or default_name
    if not name:
        raise ValueError("Gemini evaluation model is not set in the configuration")
    llm = ChatGoogleGenerativeAI(
        model=name,
        temperature=temperature,
        api_key=key,
    )
    logging.info("Initialized StoryEvaluator with Gemini model: %s", name)
    return llm


def _invoke_huggingface_once(evaluator: dict[str, Any], prompt: str) -> str:
    cfg = _cfg()
    hf_evaluation_max_new_tokens = int(cfg.get("HF_evaluation_max_new_tokens", 700))
    hf_evaluation_temperature = float(cfg.get("HF_evaluation_temperature", 0.1))
    model_id = evaluator["model"]

    # Router chat_completion (same as Step 2 extraction). The old hf-inference
    # route returns 400 for models like Qwen2.5-7B-Instruct.
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=evaluator["api_key"])
    resp = client.chat_completion(
        model=str(model_id),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=hf_evaluation_max_new_tokens,
        temperature=evaluator.get("temperature", hf_evaluation_temperature),
    )
    try:
        return (resp.choices[0].message.content or "").strip()  # type: ignore[attr-defined]
    except Exception:
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg.get("content") or "").strip()
        return str(resp).strip()


def _invoke_hf_with_retry(evaluator: dict[str, Any], prompt: str) -> str:
    last_exc = None
    for attempt in range(EVAL_RETRY_MAX_ATTEMPTS):
        try:
            return _invoke_huggingface_once(evaluator, prompt)
        except Exception as e:
            last_exc = e
            if not _is_retryable_api_error(e) or attempt == EVAL_RETRY_MAX_ATTEMPTS - 1:
                break
            delay = EVAL_RETRY_BASE_DELAY_SEC * (EVAL_RETRY_BACKOFF_FACTOR**attempt)
            logging.warning(
                "HF evaluation transient error (attempt %s/%s), retrying in %.1fs: %s",
                attempt + 1,
                EVAL_RETRY_MAX_ATTEMPTS,
                delay,
                e,
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    return ""


def _invoke_gemini_with_retry(model, prompt: str):
    last_exc = None
    for attempt in range(EVAL_RETRY_MAX_ATTEMPTS):
        try:
            return model.invoke(prompt)
        except Exception as e:
            last_exc = e
            if not _is_retryable_api_error(e) or attempt == EVAL_RETRY_MAX_ATTEMPTS - 1:
                break
            delay = EVAL_RETRY_BASE_DELAY_SEC * (EVAL_RETRY_BACKOFF_FACTOR**attempt)
            logging.warning(
                "Gemini evaluation transient error (attempt %s/%s), retrying in %.1fs: %s",
                attempt + 1,
                EVAL_RETRY_MAX_ATTEMPTS,
                delay,
                e,
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    return ""


def _invoke_with_retry(model, prompt: str):
    cfg = _cfg()
    primary = (gemini_evaluation_model or str(cfg.get("Gemini_evaluation_model") or "")).strip()
    fallback = (gemini_evaluation_fallback_model or str(cfg.get("Gemini_evaluation_fallback_model") or "")).strip()
    if isinstance(model, dict) and model.get("provider") == "huggingface":
        try:
            return _invoke_hf_with_retry(model, prompt)
        except Exception as e:
            # Any HF failure → try Gemini so the agentic loop still has a judge.
            try:
                gemini = _build_gemini_evaluator(model_name=fallback or primary)
            except Exception:
                raise e
            logging.warning("HF evaluation failed (%s); falling back to Gemini.", e)
            return _invoke_gemini_with_retry(gemini, prompt)

    try:
        return _invoke_gemini_with_retry(model, prompt)
    except Exception as e:
        if not _is_retryable_api_error(e) or not fallback:
            raise
        logging.warning(
            "Primary evaluation model exhausted retries. Switching to fallback: %s",
            fallback,
        )
        fb = _build_gemini_evaluator(model_name=fallback)
        return _invoke_gemini_with_retry(fb, prompt)


def evaluate_model(
    temperature: float = 0.3,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
):
    cfg = _cfg()
    providers = [provider.lower()] if provider else _normalise_provider_priority(
        cfg.get("Evaluation_provider_priority") or evaluation_provider_priority
    )
    errors: list[str] = []
    for candidate in providers:
        try:
            if candidate in {"hf", "huggingface", "hugging_face"}:
                return _build_huggingface_evaluator(temperature=temperature, model_name=model_name)
            if candidate == "gemini":
                return _build_gemini_evaluator(temperature=temperature, model_name=model_name)
        except Exception as e:
            errors.append(f"{candidate}: {e}")
            logging.warning("Evaluation provider unavailable (%s): %s", candidate, e)
    raise ValueError("No evaluation provider could be initialized. " + " | ".join(errors))


def _parse_json_response(response) -> dict:
    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, list):
        parts = []
        for p in raw:
            if isinstance(p, dict):
                parts.append(p.get("text", ""))
            else:
                parts.append(str(p) if p is not None else "")
        text = "".join(parts)
    else:
        text = str(raw) if raw is not None else ""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if not text.startswith("{") and "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"Evaluation response was not valid JSON: {e}")
        logging.error(f"Raw response (first 500 chars): {text[:500]}")
        return {}


def evaluate_generated_story(model, file_path) -> dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            story = file.read()
        prompt = _eval_prompts()["with_story"].format(story=story)
        response = _invoke_with_retry(model, prompt)
        evaluation_data = _parse_json_response(response)
        return evaluation_data
    except Exception as e:
        logging.error(f"Error evaluating generated story: {e}")
        return {}


def evaluate_story_text(model, story_text: str) -> dict[str, Any]:
    """Score a story string in memory (used by the agentic loop each iteration)."""
    try:
        prompt = _eval_prompts()["with_story"].format(story=story_text)
        response = _invoke_with_retry(model, prompt)
        return _parse_json_response(response)
    except Exception as e:
        logging.error(f"Error evaluating story text: {e}")
        return {}


def evaluate_generated_summary(model, summary_path, story_path=None) -> dict[str, Any]:
    try:
        with open(summary_path, "r", encoding="utf-8") as file:
            summary = file.read()
        story_text = "Not provided."
        if story_path and Path(story_path).exists():
            with open(story_path, "r", encoding="utf-8") as f:
                story_text = f.read()
        prompt = _eval_prompts()["with_summary"].format(summary=summary, story=story_text)
        response = _invoke_with_retry(model, prompt)
        evaluation_data = _parse_json_response(response)
        return evaluation_data
    except Exception as e:
        logging.error(f"Error evaluating generated summary: {e}")
        return {}


def save_evaluation_results(evaluation_data: dict[str, Any], result_type: str, source_stem: Optional[str] = None):
    try:
        cfg = _cfg()
        base = Path(str(cfg.get("BASE_PATH") or "."))
        evaluated_stories_output = str(cfg.get("Evaluated_stories_output") or "data/outputs/evaluated_stories")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = source_stem if source_stem else datetime.now().strftime("%H%M%S")
        filename = f"{timestamp}_{result_type}_{unique}_evaluation_results.json"
        output_folder = base / evaluated_stories_output
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / filename
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(evaluation_data, file, indent=4)
        logging.info(f"Evaluation results saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving evaluation results: {e}")


