"""Local story generation via Ollama, vLLM, or Hugging Face Transformers."""

from __future__ import annotations

import re
from typing import Any, Optional, Protocol

_OLLAMA_LLM_CACHE: dict[tuple, Any] = {}


class GenerationLLM(Protocol):
    def invoke(self, prompt: str) -> str: ...


class _PromptLLM:
    """Adapter so Ollama ChatOllama matches HuggingFacePipeline.invoke(prompt)."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def invoke(self, prompt: str) -> str:
        return invoke_combined_prompt(self._llm, prompt)


def generation_provider(cfg: dict[str, Any]) -> str:
    """Return ``ollama``, ``vllm``, or ``transformers``."""
    raw = str(cfg.get("Generation_provider") or "ollama").strip().lower()
    if raw in {"hf", "huggingface", "transformers", "local"}:
        return "transformers"
    if raw == "vllm":
        return "vllm"
    return "ollama"


def use_ollama_for_generation(cfg: dict[str, Any]) -> bool:
    return generation_provider(cfg) == "ollama"


def ollama_model_id(cfg: dict[str, Any]) -> str:
    return str(cfg.get("Generative_model") or cfg.get("Ollama_model") or "qwen3.5:9b").strip()


def ollama_base_url(cfg: dict[str, Any]) -> str:
    return str(cfg.get("Ollama_base_url") or "http://localhost:11434").strip().rstrip("/")


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3/Qwen3.5 may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ollama_num_ctx(cfg: dict[str, Any]) -> int:
    """Ollama context window; must be set or prompts silently truncate."""
    configured = int(cfg.get("Model_max_prompt_tokens") or 8192)
    return max(configured, 2048)


def build_chat_ollama(
    cfg: dict[str, Any],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    thinking: bool = False,
    model_override: Optional[str] = None,
    json_format: Any = None,
) -> Any:
    """Build ChatOllama. Use ``reasoning`` (not ``think``) or thinking stays on by default.

    ``json_format`` is passed through as Ollama's ``format``: ``"json"`` for
    JSON mode, or a JSON-schema dict for grammar-constrained structured output
    (Ollama >= 0.5). ``None`` (default) leaves free-form text generation alone.
    """
    from langchain_ollama import ChatOllama

    repeat_penalty = float(cfg.get("Generation_repetition_penalty") or 1.08)
    kwargs: dict[str, Any] = {}
    if json_format:
        kwargs["format"] = json_format
    return ChatOllama(
        model=str(model_override or "").strip() or ollama_model_id(cfg),
        base_url=ollama_base_url(cfg),
        temperature=temperature,
        top_p=top_p,
        num_predict=max_new_tokens,
        options={"repeat_penalty": repeat_penalty, "num_ctx": ollama_num_ctx(cfg)},
        reasoning=thinking,
        **kwargs,
    )


def load_ollama_llm(
    cfg: dict[str, Any],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    thinking: bool = False,
    model_override: Optional[str] = None,
    json_format: Any = None,
) -> GenerationLLM:
    """Cached ChatOllama instance for repeated pipeline calls."""
    import json as _json

    model = str(model_override or "").strip() or ollama_model_id(cfg)
    base_url = ollama_base_url(cfg)
    repeat_penalty = float(cfg.get("Generation_repetition_penalty") or 1.08)
    num_ctx = ollama_num_ctx(cfg)
    # A schema dict isn't hashable -- key the cache on its canonical JSON.
    fmt_key = _json.dumps(json_format, sort_keys=True) if json_format else ""
    cache_key = (model, base_url, max_new_tokens, temperature, top_p, repeat_penalty, thinking, num_ctx, fmt_key)
    if cache_key not in _OLLAMA_LLM_CACHE:
        _OLLAMA_LLM_CACHE[cache_key] = build_chat_ollama(
            cfg,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking=thinking,
            model_override=model_override,
            json_format=json_format,
        )
    return _PromptLLM(_OLLAMA_LLM_CACHE[cache_key])


def invoke_prompt(llm: Any, *, system: str, user: str) -> str:
    """Invoke ChatOllama with separate system and user messages."""
    system_text = (system or "").strip()
    user_text = (user or "").strip()
    from langchain_core.messages import HumanMessage, SystemMessage

    messages: list[Any] = []
    if system_text:
        messages.append(SystemMessage(content=system_text))
    messages.append(HumanMessage(content=user_text))
    inner = getattr(llm, "_llm", llm)
    response = inner.invoke(messages)
    result = str(getattr(response, "content", response) or "").strip()
    return strip_thinking_tags(result)


def invoke_combined_prompt(llm: Any, prompt: str) -> str:
    """Invoke when callers already merged system + user into one string."""
    text = (prompt or "").strip()
    if not text:
        return ""
    from langchain_core.messages import HumanMessage

    inner = getattr(llm, "_llm", llm)
    response = inner.invoke([HumanMessage(content=text)])
    result = str(getattr(response, "content", response) or "").strip()
    return strip_thinking_tags(result)


def vllm_base_url(cfg: dict) -> str:
    """Base URL for the vLLM OpenAI-compatible endpoint."""
    return str(cfg.get("vLLM_base_url") or "http://localhost:8001/v1").strip().rstrip("/")


def vllm_model_id(cfg: dict) -> str:
    """Model name to pass to the vLLM server (must match --model at launch)."""
    return str(
        cfg.get("vLLM_model") or cfg.get("Generative_model") or "Qwen/Qwen2.5-7B-Instruct"
    ).strip()


def build_chat_openai(
    cfg: dict[str, Any],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Any:
    """Build a raw ChatOpenAI client against the local vLLM OpenAI-compatible server.

    Shared by load_vllm_llm (non-streaming) and orchestration_routes.py's SSE
    stream (needs the raw client for .astream()) so the two call sites can't
    drift the way build_chat_ollama / an inline ChatOllama once did.
    """
    from langchain_openai import ChatOpenAI  # type: ignore

    repeat_penalty = float(cfg.get("Generation_repetition_penalty") or 1.08)
    return ChatOpenAI(
        model=vllm_model_id(cfg),
        base_url=vllm_base_url(cfg),
        api_key="EMPTY",  # required non-empty; vLLM ignores it
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        model_kwargs={"frequency_penalty": max(0.0, min(2.0, repeat_penalty - 1.0))},
    )


def load_vllm_llm(
    cfg: dict,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> GenerationLLM:
    """ChatOpenAI client for a local vLLM OpenAI-compatible server."""
    llm = build_chat_openai(cfg, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    return _PromptLLM(llm)
