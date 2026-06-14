"""Local story generation via Ollama or Hugging Face Transformers."""

from __future__ import annotations

import re
from typing import Any, Protocol

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
    """Return ``ollama`` or ``transformers``."""
    raw = str(cfg.get("Generation_provider") or "ollama").strip().lower()
    if raw in {"hf", "huggingface", "transformers", "local"}:
        return "transformers"
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


def load_ollama_llm(
    cfg: dict[str, Any],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    thinking: bool = False,
) -> GenerationLLM:
    """Cached ChatOllama instance for repeated pipeline calls."""
    from langchain_ollama import ChatOllama

    model = ollama_model_id(cfg)
    base_url = ollama_base_url(cfg)
    repeat_penalty = float(cfg.get("Generation_repetition_penalty") or 1.08)
    cache_key = (model, base_url, max_new_tokens, temperature, top_p, repeat_penalty, thinking)
    if cache_key not in _OLLAMA_LLM_CACHE:
        _OLLAMA_LLM_CACHE[cache_key] = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            num_predict=max_new_tokens,
            options={"repeat_penalty": repeat_penalty},
            think=thinking,
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
