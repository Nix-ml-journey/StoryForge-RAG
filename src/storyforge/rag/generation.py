"""Step 3 — Story generation and refinement from grounded facts.

Public API:
    generate_from_facts(query, parsed, grounded_raw, cfg, ...) -> str
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from storyforge.rag.attribution import (
    attribution_violations,
    extract_named_entities_heuristic,
    format_facts_for_prompt,
)
from storyforge.rag.generation_backend import load_ollama_llm, use_ollama_for_generation

LOG = logging.getLogger(__name__)

_LOCAL_MODEL_CACHE: dict = {}     # (model_id, precision, use_cuda) -> (tokenizer, model)
_SECTION_HEADER_RE = re.compile(r"^\[SECTION\s+(\d+).*?\]\s*$", re.IGNORECASE | re.MULTILINE)

# Lives in attribution.py for unit tests; alias keeps imports local.
_format_facts_for_prompt = format_facts_for_prompt


# ---------------------------------------------------------------------------
# Local model loader (shared cache with extraction.py's fallback path)
# ---------------------------------------------------------------------------

def _resolve_generation_dtype(*, cfg: dict[str, Any], use_cuda: bool):
    """Resolve torch dtype from Generation_precision with safe fallbacks."""
    import torch
    if not use_cuda:
        return None
    raw_precision = str(cfg.get("Generation_precision") or "auto").strip().lower()
    precision = raw_precision or "auto"
    if precision in {"bf16", "bfloat16"}:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        LOG.warning("Generation_precision=%s requested but BF16 unsupported; falling back to fp16.", raw_precision)
        return torch.float16
    if precision in {"fp16", "float16"}:
        return torch.float16
    if precision in {"auto", "default"}:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    LOG.warning("Unknown Generation_precision=%s; using auto precision.", raw_precision)
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load_or_get_cached_local_model(model_id: str, cfg: dict[str, Any]):
    """Load the local causal LM once per (model_id, precision, device) tuple.

    Called by extraction.py's fallback path as well, so weights are never loaded twice.
    """
    import torch
    use_cuda = torch.cuda.is_available()
    precision = str(cfg.get("Generation_precision") or "auto").strip().lower() or "auto"
    cache_key = (model_id, precision, use_cuda)
    if cache_key in _LOCAL_MODEL_CACHE:
        LOG.debug("Reusing cached local model: %s (precision=%s)", model_id, precision)
        return _LOCAL_MODEL_CACHE[cache_key]

    LOG.info("Loading local model (first use): %s", model_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = _resolve_generation_dtype(cfg=cfg, use_cuda=use_cuda)
    try:
        import flash_attn  # noqa: F401  # type: ignore
        attn_impl = "flash_attention_2"
        LOG.info("Flash Attention 2 enabled for %s", model_id)
    except ImportError:
        attn_impl = "eager"
        LOG.info("flash-attn not installed — using eager attention. Run: pip install flash-attn --no-build-isolation")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda" if use_cuda else None,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    _LOCAL_MODEL_CACHE[cache_key] = (tok, model)
    return tok, model


# ---------------------------------------------------------------------------
# Generation params
# ---------------------------------------------------------------------------

def _is_thinking_mode(mode: Any) -> bool:
    mode_name = str(getattr(mode, "value", mode) or "").strip().lower()
    return mode_name in {"thinking", "think", "slow", "medium"}


def _mode_generation_params(
    cfg: dict[str, Any],
    *,
    mode: Any = None,
    max_new_tokens: Optional[int] = None,
) -> tuple[int, float, float]:
    is_thinking = _is_thinking_mode(mode)
    if max_new_tokens is not None:
        token_budget = int(max_new_tokens)
    elif is_thinking:
        token_budget = int(
            cfg.get("Single_pass_thinking_max_tokens")
            or cfg.get("Single_pass_fast_max_tokens")
            or 768
        )
    else:
        token_budget = int(cfg.get("Single_pass_fast_max_tokens") or 768)

    if is_thinking:
        temperature = float(
            cfg.get("Generation_thinking_temperature")
            or cfg.get("Generation_fast_temperature")
            or 0.4
        )
        top_p = float(
            cfg.get("Generation_thinking_top_p")
            or cfg.get("Generation_fast_top_p")
            or 0.8
        )
    else:
        temperature = float(cfg.get("Generation_fast_temperature") or 0.4)
        top_p = float(cfg.get("Generation_fast_top_p") or 0.8)
    return token_budget, temperature, top_p


def _load_generation_llm(
    cfg: dict[str, Any],
    *,
    mode: Any = None,
    max_new_tokens: Optional[int] = None,
) -> Any:
    default_max_new, temperature, top_p = _mode_generation_params(
        cfg, mode=mode, max_new_tokens=max_new_tokens
    )
    if use_ollama_for_generation(cfg):
        LOG.info(
            "Using Ollama for story generation (model=%s)",
            cfg.get("Generative_model") or cfg.get("Ollama_model") or "qwen3.5:9b",  # Ollama tag
        )
        return load_ollama_llm(
            cfg,
            max_new_tokens=default_max_new,
            temperature=temperature,
            top_p=top_p,
            thinking=_is_thinking_mode(mode),
        )

    model_id = cfg.get("Generative_model") or cfg.get("GENERATIVE_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
    repetition_penalty = float(cfg.get("Generation_repetition_penalty") or 1.08)
    no_repeat_ngram_size = int(cfg.get("Generation_no_repeat_ngram_size") or 4)

    from transformers import pipeline  # type: ignore
    from langchain_huggingface import HuggingFacePipeline

    tok, model = _load_or_get_cached_local_model(model_id, cfg)
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=default_max_new,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tok.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


# ---------------------------------------------------------------------------
# Story structure helpers
# ---------------------------------------------------------------------------

def _flow_section_headers(cfg: dict[str, Any]) -> str:
    """Fixed 5-section outline used by every generation and refine pass."""
    return "\n".join(
        [
            "[SECTION 1: WHO, WHERE, WHEN (The Setup)]",
            "[SECTION 2: WHAT (The Problem Starts)]",
            "[SECTION 3: TWIST/COMPLICATION (The Challenge)]",
            "[SECTION 4: HOW (The Big Action/Climax)]",
            "[SECTION 5: WHY/OUTCOME (The Moral and Conclusion)]",
        ]
    )


def _split_section_bodies(story: str) -> dict[int, str]:
    text = str(story or "")
    matches = list(_SECTION_HEADER_RE.finditer(text))
    if not matches:
        return {}
    sections: dict[int, str] = {}
    for i, m in enumerate(matches):
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[idx] = text[start:end].strip()
    return sections


def _sentence_count(text: str) -> int:
    body = str(text or "").strip()
    if not body:
        return 0
    return len(re.findall(r"[^.!?]+[.!?]", body))


def _sections_below_min_sentences(story: str, *, min_sentences: int) -> dict[int, int]:
    if min_sentences <= 0:
        return {}
    sections = _split_section_bodies(story)
    short: dict[int, int] = {}
    for i in range(1, 6):
        count = _sentence_count(sections.get(i, ""))
        if count < min_sentences:
            short[i] = count
    return short


# ---------------------------------------------------------------------------
# Attribution gate
# ---------------------------------------------------------------------------

def _apply_attribution_gate(story: str, facts: tuple, cfg: dict[str, Any]) -> str:
    """Post-check for names not present in grounded facts.

    Default: log only (Attribution_gate_truncate: false).
    Set Attribution_gate_truncate: true to restore hard truncation.
    """
    story_body_for_check = re.sub(r"^\[SECTION \d+:.*?\]\s*", "", story, flags=re.MULTILINE)
    violations = attribution_violations(story=story_body_for_check, facts=facts)
    if not violations:
        return story
    all_found = extract_named_entities_heuristic(story_body_for_check)
    violation_ratio = len(violations) / max(1, len(all_found))

    truncate_enabled = str(cfg.get("Attribution_gate_truncate") or "").strip().lower() in ("true", "1", "yes")
    truncation_threshold = int(cfg.get("Attribution_violation_threshold") or 8)
    if truncate_enabled and (len(violations) > truncation_threshold or violation_ratio > 0.6):
        LOG.warning(
            "Attribution gate: %d novel entities (%.0f%% of body text) exceed threshold %d -- truncating story.",
            len(violations),
            violation_ratio * 100,
            truncation_threshold,
        )
        paras = [p.strip() for p in story.split("\n\n") if p.strip()]
        return "\n\n".join(paras[: min(6, len(paras))]).strip()
    LOG.info(
        "Attribution gate: %d possible novel entit(y/ies) (%.0f%% of body text) -- logging only: %s",
        len(violations),
        violation_ratio * 100,
        sorted(violations),
    )
    return story


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_from_facts(
    query: str,
    parsed: Any,
    grounded_raw: str,
    cfg: dict[str, Any],
    *,
    mode: Any = None,
    refine_feedback: Optional[str] = None,
    prior_draft: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> str:
    """Step 3: write or refine a 5-section story from grounded facts.

    With refine_feedback + prior_draft, runs the refine prompt instead of a fresh draft.
    """
    # Lazy import breaks the extraction ↔ generation circular dependency at module level.
    from storyforge.rag.extraction import _get_generation_prompts

    prompts = _get_generation_prompts()
    formatted_facts = _format_facts_for_prompt(parsed)
    facts_for_prompt = formatted_facts if formatted_facts else grounded_raw

    gen_llm = _load_generation_llm(cfg, mode=mode, max_new_tokens=max_new_tokens)

    if refine_feedback and prior_draft:
        story_prompt = (
            f"{(prompts['refine_system'] or '').strip()}\n\n"
            + prompts["refine_user"]
            .format(
                query=query,
                section_headers=_flow_section_headers(cfg),
                grounded_facts=facts_for_prompt,
                prior_draft=prior_draft.strip(),
                feedback=refine_feedback.strip(),
            )
            .strip()
        )
    else:
        story_prompt = (
            f"{(prompts['story_system'] or '').strip()}\n\n"
            + prompts["story_user"]
            .format(
                query=query,
                section_headers=_flow_section_headers(cfg),
                grounded_facts=facts_for_prompt,
            )
            .strip()
        )

    story = str(gen_llm.invoke(story_prompt) or "").strip()
    return _apply_attribution_gate(story, parsed.facts, cfg)
