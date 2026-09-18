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

from storyforge.api_errors import is_retryable_api_error
from storyforge.config.config import load_config, load_prompts

# Root logging is configured once in storyforge/__init__.py.


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


# Shared with rag/extraction.py so both HF call sites classify errors identically.
# Aliased rather than imported under its own name to keep existing call sites intact.
_is_retryable_api_error = is_retryable_api_error


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
    token = str(cfg.get("facehugging_api") or "").strip()
    if not token:
        raise ValueError("Hugging Face API key is not set for evaluation")
    model_id = (
        model_name
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
    key = str(cfg.get("Gemini_api_key") or "").strip()
    if not key:
        raise ValueError("Gemini API key is not set in the configuration")
    default_name = str(cfg.get("Gemini_evaluation_model") or "").strip()
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


_LOCAL_EVAL_CACHE: dict[tuple, Any] = {}  # (model_id, device) -> (tokenizer, model)


def _local_evaluation_device(cfg: dict[str, Any]) -> str:
    device = str(cfg.get("Local_evaluation_device") or "cpu").strip().lower() or "cpu"
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logging.warning("Local_evaluation_device=cuda requested but unavailable; using cpu.")
                return "cpu"
        except Exception:
            return "cpu"
    return device


def _load_local_evaluator_model(model_id: str, device: str):
    """Load (or return cached) a small local causal LM used only for scoring drafts."""
    cache_key = (model_id, device)
    if cache_key in _LOCAL_EVAL_CACHE:
        return _LOCAL_EVAL_CACHE[cache_key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    logging.info("Loading local evaluation model (first use): %s on %s", model_id, device)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda" if device == "cuda" else None,
        torch_dtype=(torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None),
    )
    if device != "cuda":
        model = model.to(device)

    _LOCAL_EVAL_CACHE[cache_key] = (tok, model)
    return tok, model


def _build_local_evaluator(temperature: float = 0.1, model_name: Optional[str] = None) -> dict[str, Any]:
    cfg = _cfg()
    model_id = model_name or str(cfg.get("Local_evaluation_model") or "Qwen/Qwen2.5-3B-Instruct")
    device = _local_evaluation_device(cfg)
    logging.info("Initialized StoryEvaluator with local model: %s (%s)", model_id, device)
    return {"provider": "local", "model": model_id, "device": device, "temperature": float(temperature)}


def _invoke_local_once(evaluator: dict[str, Any], prompt: str) -> str:
    cfg = _cfg()
    max_new = int(cfg.get("Local_evaluation_max_new_tokens") or cfg.get("HF_evaluation_max_new_tokens") or 700)
    tok, model = _load_local_evaluator_model(evaluator["model"], evaluator["device"])

    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        # Tokenizer has no chat template — fall back to raw text.
        input_ids = tok(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    temperature = float(evaluator.get("temperature") or 0.1)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            pad_token_id=(tok.pad_token_id or tok.eos_token_id),
        )
    generated = out[0][input_ids.shape[-1] :]
    return tok.decode(generated, skip_special_tokens=True).strip()


def _invoke_local_with_fallback(evaluator: dict[str, Any], prompt: str) -> str:
    """Try the local model; on any failure (e.g. OOM), fall back to the API chain.

    Keeps the agentic loop moving even if the local model can't be loaded or
    run on this machine, rather than failing every evaluation for the rest of
    the run.
    """
    try:
        return _invoke_local_once(evaluator, prompt)
    except Exception as e:
        logging.warning(
            "Local evaluation model failed (%s); falling back to the API provider chain.", e
        )
        cfg = _cfg()
        providers = _normalise_provider_priority(
            cfg.get("Evaluation_provider_priority") or ["huggingface", "gemini"]
        )
        temperature = evaluator.get("temperature")
        errors = [f"local: {e}"]
        for candidate in providers:
            try:
                if candidate in {"hf", "huggingface", "hugging_face"}:
                    hf_evaluator = _build_huggingface_evaluator(temperature=temperature)
                    return _invoke_hf_with_retry(hf_evaluator, prompt)
                if candidate == "gemini":
                    gemini = _build_gemini_evaluator(temperature=temperature)
                    return _invoke_gemini_with_retry(gemini, prompt)
            except Exception as e2:
                errors.append(f"{candidate}: {e2}")
        raise RuntimeError(
            "Local evaluation failed and no API fallback succeeded: " + " | ".join(errors)
        ) from e


def _invoke_huggingface_once(evaluator: dict[str, Any], prompt: str) -> str:
    cfg = _cfg()
    hf_evaluation_max_new_tokens = int(cfg.get("HF_evaluation_max_new_tokens", 700))
    hf_evaluation_temperature = float(cfg.get("HF_evaluation_temperature", 0.1))
    model_id = evaluator["model"]

    from huggingface_hub import InferenceClient

    client = InferenceClient(token=evaluator["api_key"])
    resp = client.chat_completion(
        model=str(model_id),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=hf_evaluation_max_new_tokens,
        temperature=evaluator.get("temperature", hf_evaluation_temperature),
    )
    try:
        return (resp.choices[0].message.content or "").strip()
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
                attempt + 1, EVAL_RETRY_MAX_ATTEMPTS, delay, e,
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
                attempt + 1, EVAL_RETRY_MAX_ATTEMPTS, delay, e,
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    return ""


def _invoke_with_retry(model, prompt: str):
    cfg = _cfg()
    primary = str(cfg.get("Gemini_evaluation_model") or "").strip()
    fallback = str(cfg.get("Gemini_evaluation_fallback_model") or "").strip()
    if isinstance(model, dict) and model.get("provider") == "local":
        return _invoke_local_with_fallback(model, prompt)

    if isinstance(model, dict) and model.get("provider") == "huggingface":
        try:
            return _invoke_hf_with_retry(model, prompt)
        except Exception as e:
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
            "Primary evaluation model exhausted retries. Switching to fallback: %s", fallback,
        )
        fb = _build_gemini_evaluator(model_name=fallback)
        return _invoke_gemini_with_retry(fb, prompt)


def evaluate_model(
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
):
    """Build an evaluator for the first available provider.

    `temperature=None` (the default) means "use the configured value" --
    HF_evaluation_temperature for the HF provider. This previously defaulted to
    a hardcoded 0.3, and since every caller invokes evaluate_model() with no
    arguments, the configured value was unreachable and evaluation always ran
    at 0.3. That extra sampling noise feeds straight into the agentic loop's
    ACCEPT / REFINE / RE_RETRIEVE decisions.
    """
    cfg = _cfg()
    # Resolve once here so no provider builder ever receives None (Gemini would
    # forward it straight into ChatGoogleGenerativeAI). HF_evaluation_temperature
    # is the only evaluation-temperature knob in the config and applies to
    # whichever provider ends up serving the request.
    if temperature is None:
        try:
            temperature = float(cfg.get("HF_evaluation_temperature", 0.1))
        except (TypeError, ValueError):
            temperature = 0.1

    # Evaluation_mode: "local" short-circuits the API provider priority below --
    # it's a separate in-process backend (Transformers), not another HTTP
    # provider to race against huggingface/gemini. Removes the HF/Gemini API
    # round-trip (and rate-limit risk) from every agentic-loop iteration.
    eval_mode = str(cfg.get("Evaluation_mode") or "api").strip().lower()
    if eval_mode == "local":
        return _build_local_evaluator(temperature=temperature, model_name=model_name)

    providers = [provider.lower()] if provider else _normalise_provider_priority(
        cfg.get("Evaluation_provider_priority") or ["huggingface", "gemini"]
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
        return _parse_json_response(response)
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
        return _parse_json_response(response)
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
