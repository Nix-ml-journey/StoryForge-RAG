"""Step 2 — Grounded fact extraction via HF Inference API (with local fallback).

Public API:
    extract_grounded_facts(query, chunks, cfg) -> tuple[str, ParsedFacts]
    _get_generation_prompts()                  -> dict[str, str]
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from huggingface_hub import InferenceClient

from storyforge.api_errors import is_retryable_api_error
from storyforge.config.config import load_prompts
from storyforge.rag.attribution import (
    ParsedFacts,
    parse_grounded_facts_json,
    salvage_grounded_facts_json,
)
from storyforge.rag.generation_backend import (
    generation_provider,
    load_ollama_llm,
    load_vllm_llm,
)
from storyforge.rag.retrieval import _format_chunks_for_prompt

LOG = logging.getLogger(__name__)

_EXTRACTION_RETRY_MAX_ATTEMPTS = 3
_EXTRACTION_RETRY_BASE_DELAY_SEC = 2
_EXTRACTION_RETRY_BACKOFF_FACTOR = 2


def _hf_token(cfg: dict[str, Any]) -> str:
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
        or cfg.get("HF_evaluation_model")
        or "Qwen/Qwen2.5-7B-Instruct"
    )
    temperature = float(cfg.get("HF_grounded_facts_temperature") or 0.1)
    max_new = int(cfg.get("HF_grounded_facts_max_new_tokens") or 300)
    token = _hf_token(cfg)
    if not token:
        raise ValueError("Missing Hugging Face token for grounded facts extraction (facehugging_api / env).")

    # Treat False as off — `False or "true"` would wrongly enable JSON mode.
    _jm_val = cfg.get("HF_grounded_facts_json_mode")
    if _jm_val is None:
        _json_mode_enabled = True
    elif isinstance(_jm_val, bool):
        _json_mode_enabled = _jm_val
    else:
        _json_mode_enabled = str(_jm_val).strip().lower() not in ("false", "0", "no")

    # Qwen3 models (e.g. the default HF_grounded_facts_model "Qwen/Qwen3-8B")
    # emit hidden chain-of-thought reasoning by default. That reasoning counts
    # against max_tokens, and for this structured-extraction call it can (and
    # in testing reliably did) consume the entire budget before the model ever
    # writes the JSON answer -- finish_reason "length", empty message.content,
    # 0 facts parsed. vLLM/SGLang-style OpenAI-compatible backends expose a
    # toggle via extra_body.chat_template_kwargs.enable_thinking, but testing
    # against the actual HF-routed backend for Qwen3-8B showed no measurable
    # effect from that flag (reasoning-token counts were statistically the
    # same with it on vs. off) -- this specific router/backend combination
    # appears to silently ignore it. Qwen3 is separately trained to honor a
    # literal "/no_think" suffix in the user turn regardless of API-level
    # support, so that is appended instead/as well. Default both off for
    # extraction specifically -- we need a short, deterministic JSON answer,
    # not a reasoning trace -- but keep it configurable/off-able in case a
    # given provider/model combination rejects the field outright.
    _dt_val = cfg.get("HF_grounded_facts_disable_thinking")
    if _dt_val is None:
        _disable_thinking = True
    elif isinstance(_dt_val, bool):
        _disable_thinking = _dt_val
    else:
        _disable_thinking = str(_dt_val).strip().lower() not in ("false", "0", "no")

    user_content = str(user or "").strip()
    if _disable_thinking:
        # Model-level soft switch (works regardless of whether the router/
        # backend wires up an API-level toggle -- see comment above).
        user_content = f"{user_content}\n\n/no_think"

    client = InferenceClient(token=token)
    request_kwargs: dict[str, Any] = {
        "model": str(model_id),
        "messages": [
            {"role": "system", "content": str(system or "").strip()},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": max_new,
        "temperature": temperature,
    }
    if _json_mode_enabled:
        request_kwargs["response_format"] = {"type": "json_object"}
    if _disable_thinking:
        # Kept in addition to /no_think: harmless if the backend ignores it,
        # and it may still be honored by other providers/models in the future.
        request_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    # extra_body and response_format above are both optional -- not every
    # backend the router lands on supports them. A bare TypeError doesn't say
    # which kwarg was rejected, so on failure drop both optional kwargs and
    # retry once rather than guessing; if a real (non-optional-kwarg) error
    # occurs, it propagates normally (extract_grounded_facts()'s caller falls
    # back to local extraction on any exception here).
    try:
        resp = client.chat_completion(**request_kwargs)
    except TypeError:
        dropped = [k for k in ("extra_body", "response_format") if k in request_kwargs]
        if not dropped:
            raise
        LOG.info("HF grounded-facts unsupported kwarg(s) %s; retrying without them.", dropped)
        for k in dropped:
            request_kwargs.pop(k, None)
        resp = client.chat_completion(**request_kwargs)

    # Log finish_reason/usage on every call at DEBUG (cheap, and priceless when
    # a JSON-parse failure shows up later): "length" here means the answer
    # itself -- not hidden reasoning, that's handled separately above -- got
    # cut off mid-JSON by max_tokens, which is a token-budget problem with a
    # known fix (raise HF_grounded_facts_max_new_tokens). Any other
    # finish_reason with a parse failure points to a genuine formatting bug
    # instead, which raising the budget would not fix.
    try:
        _choice = resp.choices[0]  # type: ignore[attr-defined]
        LOG.info(
            "HF grounded-facts call: finish_reason=%s usage=%s",
            getattr(_choice, "finish_reason", None), getattr(resp, "usage", None),
        )
        if getattr(_choice, "finish_reason", None) == "length":
            LOG.warning(
                "HF grounded-facts response was cut off by max_tokens (finish_reason=length, "
                "usage=%s). If this leads to a JSON parse failure below, the fix is to raise "
                "HF_grounded_facts_max_new_tokens (currently %s), not to change the prompt.",
                getattr(resp, "usage", None), max_new,
            )
    except Exception:
        pass

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


def _hf_chat_extract_json_with_retry(
    *,
    cfg: dict[str, Any],
    system: str,
    user: str,
) -> str:
    """Retry HF extraction on transient errors; raise non-retryable immediately."""
    last_exc: Optional[Exception] = None
    for attempt in range(_EXTRACTION_RETRY_MAX_ATTEMPTS):
        try:
            return _hf_chat_extract_json(cfg=cfg, system=system, user=user)
        except Exception as e:
            last_exc = e
            if not is_retryable_api_error(e) or attempt == _EXTRACTION_RETRY_MAX_ATTEMPTS - 1:
                raise
            delay = _EXTRACTION_RETRY_BASE_DELAY_SEC * (_EXTRACTION_RETRY_BACKOFF_FACTOR**attempt)
            LOG.warning(
                "HF grounded-facts transient error (attempt %s/%s), retrying in %.1fs: %s",
                attempt + 1, _EXTRACTION_RETRY_MAX_ATTEMPTS, delay, e,
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    return ""


# JSON schema for Ollama structured outputs (``format=<schema>``, Ollama >= 0.5).
# This is the local equivalent of the HF path's response_format: decoding is
# grammar-constrained, so the model cannot emit prose, fences, or broken
# quoting -- only truncation (num_predict) can still cut it short, which
# salvage_grounded_facts_json() handles.
FACTS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "fact": {"type": "string"},
                    "source_chunk_ids": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                },
                "required": ["type", "fact", "source_chunk_ids", "quote"],
            },
        }
    },
    "required": ["facts"],
}

_LOCAL_JSON_ONLY_SUFFIX = (
    "\n\nOUTPUT FORMAT (strict): reply with ONE JSON object and nothing else -- "
    'no prose, no markdown fences, no <think> block. Start with {"facts": [ and end with ]}. '
    'Copy each "source_chunk_ids" value exactly from the [CHUNK <id> | title] headers above.'
)
_LOCAL_COMPACT_RETRY_SUFFIX = (
    "\n\nYour previous answer could not be used ({reason}). Return AT MOST {max_facts} facts, "
    "keep every quote under 10 words, and make sure the JSON object is complete and closed."
)


def _local_json_format(cfg: dict[str, Any]) -> Any:
    """Ollama ``format`` for the local facts call: schema (default) | json | off."""
    raw = str(cfg.get("Local_grounded_facts_json_format") or "schema").strip().lower()
    if raw in ("off", "false", "none", "0", "no"):
        return None
    if raw == "json":
        return "json"
    return FACTS_JSON_SCHEMA


def _grounded_facts_provider(cfg: dict[str, Any]) -> str:
    """``hf`` (default: HF API, local fallback) or ``local`` (skip HF entirely)."""
    raw = str(cfg.get("Grounded_facts_provider") or "hf").strip().lower()
    return "local" if raw in ("local", "ollama", "offline") else "hf"


def _load_facts_llm(cfg: dict[str, Any], *, json_format: Any = None) -> Any:
    """Local Ollama / vLLM / Transformers fallback when HF is unavailable."""
    max_new = int(cfg.get("HF_grounded_facts_max_new_tokens") or 300)
    temperature = float(cfg.get("HF_grounded_facts_temperature") or 0.1)
    top_p = float(cfg.get("Generation_fast_top_p") or 0.8)

    provider = generation_provider(cfg)
    if provider == "vllm":
        LOG.info("Using vLLM for grounded-facts fallback")
        llm = load_vllm_llm(cfg, max_new_tokens=max_new, temperature=temperature, top_p=top_p)
        if json_format:
            # vLLM's OpenAI-compatible server supports JSON mode natively.
            inner = getattr(llm, "_llm", None)
            if inner is not None and hasattr(inner, "bind"):
                llm._llm = inner.bind(response_format={"type": "json_object"})
        return llm
    if provider == "ollama":
        LOG.info("Using Ollama for grounded-facts fallback (json_format=%s)",
                 "schema" if isinstance(json_format, dict) else json_format)
        return load_ollama_llm(
            cfg, max_new_tokens=max_new, temperature=temperature, top_p=top_p,
            thinking=False, json_format=json_format,
        )

    from storyforge.rag.generation import _load_or_get_cached_local_model
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


def _invoke_local_facts(cfg: dict[str, Any], prompt: str, json_format: Any) -> str:
    """One local call; if the backend rejects the JSON format, retry once without it."""
    try:
        facts_llm = _load_facts_llm(cfg, json_format=json_format)
        return str(facts_llm.invoke(prompt) or "").strip()
    except Exception as e:
        if not json_format:
            raise
        LOG.warning(
            "Local grounded-facts call failed with json_format enabled (%s); retrying without it. "
            "Ollama < 0.5 does not support schema output -- set Local_grounded_facts_json_format: \"json\".",
            e,
        )
        facts_llm = _load_facts_llm(cfg, json_format=None)
        return str(facts_llm.invoke(prompt) or "").strip()


def _extract_grounded_facts_local(
    query: str,
    chunks: list[dict[str, Any]],
    cfg: dict[str, Any],
    prompts: dict[str, str],
) -> tuple[str, ParsedFacts]:
    """Local (Ollama / vLLM / Transformers) grounded-facts extraction.

    Hardened relative to a bare ``invoke`` + strict ``json.loads``:
      1. grammar-constrained JSON via Ollama ``format`` (schema by default);
      2. an explicit JSON-only instruction appended to the prompt;
      3. lenient salvage parsing (fences, prose, bare lists, truncation) with
         cited chunk ids validated against the actually-retrieved chunks;
      4. one compact retry (fewer, shorter facts) when the first answer
         yields 0 usable facts -- truncation is the most common cause;
      5. a loud ERROR with the failure category when it still yields nothing.
    """
    json_format = _local_json_format(cfg)
    retries = max(0, int(cfg.get("Local_grounded_facts_retries", 1) or 0))
    compact_max = int(cfg.get("Local_grounded_facts_compact_max_facts") or 12)
    known_ids = [str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")]

    base_prompt = (
        f"{(prompts['facts_system'] or '').strip()}\n\n"
        + prompts["facts_user"].format(
            query=query,
            retrieval_chunks=_format_chunks_for_prompt(chunks),
        ).strip()
        + _LOCAL_JSON_ONLY_SUFFIX
    )

    raw = ""
    parsed = ParsedFacts(facts=(), raw={})
    diag: dict[str, Any] = {}
    prompt = base_prompt
    for attempt in range(retries + 1):
        raw = _invoke_local_facts(cfg, prompt, json_format)
        parsed, diag = salvage_grounded_facts_json(raw, known_chunk_ids=known_ids or None)
        LOG.info(
            "Local grounded-facts attempt %d/%d: status=%s raw_fact_objects=%s kept=%d dropped=%s chars=%d",
            attempt + 1, retries + 1, diag.get("status"), diag.get("raw_fact_objects"),
            len(parsed.facts), diag.get("dropped"), len(raw),
        )
        if parsed.facts:
            return raw, parsed
        prompt = base_prompt + _LOCAL_COMPACT_RETRY_SUFFIX.format(
            reason=diag.get("status"), max_facts=compact_max
        )

    LOG.error(
        "LOCAL grounded-facts extraction FAILED for query=%r after %d attempt(s): status=%s "
        "raw_fact_objects=%s dropped=%s. Step 3 will have NO grounded facts (the agentic loop "
        "will not ACCEPT this). Raw response (first 300 chars): %r",
        query, retries + 1, diag.get("status"), diag.get("raw_fact_objects"),
        diag.get("dropped"), raw[:300],
    )
    return raw, parsed


def extract_grounded_facts(
    query: str,
    chunks: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[str, ParsedFacts]:
    """Step 2: extract grounded facts JSON (HF first, local fallback).

    ``Grounded_facts_provider: "local"`` skips the HF call entirely (offline /
    no HF credits) instead of paying a failed request + retry on every query.
    """
    prompts = _get_generation_prompts()
    if _grounded_facts_provider(cfg) == "local":
        LOG.info("Grounded_facts_provider=local: skipping HF, using local extraction.")
        return _extract_grounded_facts_local(query, chunks, cfg, prompts)

    try:
        grounded_raw = _hf_chat_extract_json_with_retry(
            cfg=cfg,
            system=prompts["facts_system"],
            user=prompts["facts_user"].format(
                query=query,
                retrieval_chunks=_format_chunks_for_prompt(chunks),
            ),
        )
    except Exception as e:
        LOG.warning("HF routed extraction unavailable; falling back to local extraction. Error: %s", e)
        return _extract_grounded_facts_local(query, chunks, cfg, prompts)

    try:
        parsed = parse_grounded_facts_json(grounded_raw)
    except Exception as e:
        LOG.warning(
            "Grounded-facts JSON parse failed (%s). Raw response (first 300 chars): %r",
            e, grounded_raw[:300],
        )
        parsed = ParsedFacts(facts=(), raw={})

    if not parsed.facts:
        # Parsing succeeded but yielded zero usable facts -- either the model
        # returned an empty/near-empty "facts" list, or every fact was dropped
        # by parse_grounded_facts_json() for missing fact text or source_chunk_ids
        # (see attribution.parse_grounded_facts_json). Log enough to tell those
        # apart without re-running: how many raw fact entries existed vs. how
        # many survived filtering, and a slice of the raw response.
        raw_facts_list = (parsed.raw or {}).get("facts") or []
        raw_count = len(raw_facts_list) if isinstance(raw_facts_list, list) else "n/a"
        LOG.warning(
            "Grounded-facts extraction yielded 0 usable facts for query=%r "
            "(raw 'facts' entries: %s, parsed+kept: 0). Raw response (first 300 chars): %r",
            query, raw_count, grounded_raw[:300],
        )
    return grounded_raw, parsed
