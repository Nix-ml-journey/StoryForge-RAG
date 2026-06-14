"""Step 2 — Grounded fact extraction via HF Inference API (with local fallback).

Public API:
    extract_grounded_facts(query, chunks, cfg) -> tuple[str, ParsedFacts]
    _get_generation_prompts()                  -> dict[str, str]
"""
from __future__ import annotations

import logging
from typing import Any

from huggingface_hub import InferenceClient

from storyforge.config.config import load_prompts
from storyforge.rag.attribution import ParsedFacts, parse_grounded_facts_json
from storyforge.rag.generation_backend import load_ollama_llm, use_ollama_for_generation
from storyforge.rag.retrieval import _format_chunks_for_prompt

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf_token(cfg: dict[str, Any]) -> str:
    # secrets.py already overlays all HF env vars into facehugging_api at load time.
    return (cfg.get("facehugging_api") or "").strip()


def _get_generation_prompts() -> dict[str, str]:
    """Load prompt templates from prompts.yaml (generation section)."""
    p = (load_prompts() or {}).get("generation") or {}
    return {
        "facts_system": p.get("grounded_facts_system") or "Return grounded facts as JSON.",
        "facts_user": (
            p.get("grounded_facts_user")
            or "QUERY:\n{query}\n\nCHUNKS:\n{retrieval_chunks}\n\nReturn JSON facts."
        ),
        "story_system": p.get("grounded_story_system") or "Write a grounded story.",
        "story_user": (
            p.get("grounded_story_user")
            or "QUERY:\n{query}\n\nSECTION_HEADERS:\n{section_headers}\n\nGROUNDED_FACTS:\n{grounded_facts}"
        ),
        "refine_system": (
            p.get("grounded_story_refine_system")
            or "You revise a grounded story draft using reviewer feedback while staying strictly grounded."
        ),
        "refine_user": (
            p.get("grounded_story_refine_user")
            or (
                "QUERY:\n{query}\n\nSECTION_HEADERS:\n{section_headers}\n\n"
                "GROUNDED_FACTS:\n{grounded_facts}\n\nPREVIOUS_DRAFT:\n{prior_draft}\n\n"
                "REVIEWER_FEEDBACK:\n{feedback}\n\nRewrite the complete story."
            )
        ),
    }


def _hf_chat_extract_json(
    *,
    cfg: dict[str, Any],
    system: str,
    user: str,
) -> str:
    """Step 2 via Hugging Face Inference API (chat_completion router endpoint).

    Uses the router chat endpoint because many instruct models are only served there.
    Falls back gracefully when the model backend does not support response_format.
    """
    model_id = (
        cfg.get("HF_grounded_facts_model")
        or cfg.get("HF_structured_profile_model")
        or cfg.get("HF_evaluation_model")
        or "Qwen/Qwen2.5-7B-Instruct"
    )
    temperature = float(cfg.get("HF_grounded_facts_temperature") or cfg.get("Layer1_temperature") or 0.1)
    max_new = int(cfg.get("HF_grounded_facts_max_new_tokens") or cfg.get("Layer1_max_tokens") or 300)
    token = _hf_token(cfg)
    if not token:
        raise ValueError("Missing Hugging Face token for grounded facts extraction (facehugging_api / env).")

    client = InferenceClient(token=token)
    request_kwargs: dict[str, Any] = {
        "model": str(model_id),
        "messages": [
            {"role": "system", "content": str(system or "").strip()},
            {"role": "user",   "content": str(user   or "").strip()},
        ],
        "max_tokens": max_new,
        "temperature": temperature,
    }

    # Optional strict JSON mode (falls back for backends that do not accept response_format).
    json_mode_raw = str(cfg.get("HF_grounded_facts_json_mode") or "true").strip().lower()
    if json_mode_raw not in ("false", "0", "no"):
        request_kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat_completion(**request_kwargs)
    except TypeError:
        if "response_format" not in request_kwargs:
            raise
        LOG.info("HF grounded-facts response_format unsupported; retrying without JSON mode.")
        request_kwargs.pop("response_format", None)
        resp = client.chat_completion(**request_kwargs)

    try:
        return (resp.choices[0].message.content or "").strip()  # type: ignore[attr-defined]
    except Exception:
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg["content"]).strip()
        return str(resp).strip()


def _load_facts_llm(cfg: dict[str, Any]) -> Any:
    """Step 2 local fallback when the HF API is unavailable."""
    max_new = int(cfg.get("HF_grounded_facts_max_new_tokens") or cfg.get("Layer1_max_tokens") or 300)
    temperature = float(cfg.get("HF_grounded_facts_temperature") or cfg.get("Layer1_temperature") or 0.1)
    top_p = float(cfg.get("Generation_fast_top_p") or 0.8)

    if use_ollama_for_generation(cfg):
        LOG.info("Using Ollama for grounded-facts fallback")
        return load_ollama_llm(cfg, max_new_tokens=max_new, temperature=temperature, top_p=top_p, thinking=False)

    # Transformers path — lazy import to avoid loading heavy deps at module import time.
    # Uses the shared model cache from generation.py so GPU weights aren't loaded twice.
    from storyforge.rag.generation import _load_or_get_cached_local_model  # lazy: breaks circular
    from transformers import pipeline  # type: ignore
    from langchain_huggingface import HuggingFacePipeline

    model_id = cfg.get("Generative_model") or cfg.get("GENERATIVE_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
    tok, model = _load_or_get_cached_local_model(model_id, cfg)
    fact_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=fact_pipe)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_grounded_facts(
    query: str,
    chunks: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[str, ParsedFacts]:
    """Step 2: turn retrieved chunks into grounded facts (JSON).

    Tries the HF Inference API first; falls back to a local Ollama / Transformers
    model if the API call fails.  Returns (raw_text, parsed_facts).
    """
    prompts = _get_generation_prompts()
    try:
        grounded_raw = _hf_chat_extract_json(
            cfg=cfg,
            system=prompts["facts_system"],
            user=prompts["facts_user"].format(
                query=query,
                retrieval_chunks=_format_chunks_for_prompt(chunks),
            ),
        )
    except Exception as e:
        LOG.warning("HF routed extraction unavailable; falling back to local extraction. Error: %s", e)
        facts_llm = _load_facts_llm(cfg)
        facts_prompt = (
            f"{(prompts['facts_system'] or '').strip()}\n\n"
            + prompts["facts_user"].format(
                query=query,
                retrieval_chunks=_format_chunks_for_prompt(chunks),
            ).strip()
        )
        grounded_raw = str(facts_llm.invoke(facts_prompt) or "").strip()

    try:
        parsed = parse_grounded_facts_json(grounded_raw)
    except Exception:
        parsed = ParsedFacts(facts=(), raw={})
    return grounded_raw, parsed
