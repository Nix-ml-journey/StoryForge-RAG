"""Step 3 — Story generation and refinement from grounded facts.

Public API:
    generate_from_facts(query, parsed, grounded_raw, cfg, ...) -> str
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

from storyforge.rag.attribution import (
    attribution_violations,
    extract_named_entities_heuristic,
    format_facts_for_prompt,
)
from storyforge.rag.generation_backend import (
    generation_provider,
    load_ollama_llm,
    load_vllm_llm,
)
from storyforge.rag.length_profile import (
    LengthProfile,
    is_thinking_mode,
    length_token_cap,
    resolve_length_profile,
)
from storyforge.rag.length_profile import sentence_count as _sentence_count
from storyforge.rag.length_profile import split_section_bodies as _split_section_bodies
from storyforge.rag.section_rules import OVERSHOOT as _SECTION_OVERSHOOT
from storyforge.rag.section_rules import TERMINAL as _TERMINAL
from storyforge.rag.section_rules import trim_overlong, without_sentence_starts
from storyforge._load_lock import serialized

LOG = logging.getLogger(__name__)

_LOCAL_MODEL_CACHE: dict = {}     # (model_id, precision, use_cuda) -> (tokenizer, model)



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


@serialized
def _load_or_get_cached_local_model(model_id: str, cfg: dict[str, Any]):
    """Load the local causal LM once per (model_id, precision, device)."""
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



def _mode_generation_params(
    cfg: dict[str, Any],
    *,
    mode: Any = None,
    max_new_tokens: Optional[int] = None,
    length: Any = None,
    profile: Optional[LengthProfile] = None,
) -> tuple[int, float, float]:
    """Resolve (max_new_tokens, temperature, top_p). Tokens from length; mode picks sampling."""
    is_thinking = is_thinking_mode(mode)
    token_cap = length_token_cap(cfg)
    if max_new_tokens is not None:
        token_budget = min(int(max_new_tokens), token_cap)
    else:
        profile = profile or resolve_length_profile(cfg, length=length, mode=mode)
        token_budget = profile.max_new_tokens

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
    profile: Optional[LengthProfile] = None,
) -> Any:
    default_max_new, temperature, top_p = _mode_generation_params(
        cfg, mode=mode, max_new_tokens=max_new_tokens, profile=profile
    )
    provider = generation_provider(cfg)

    if provider == "vllm":
        from storyforge.rag.generation_backend import vllm_model_id
        LOG.info("[GENERATION] Served by vLLM: %s", vllm_model_id(cfg))
        return load_vllm_llm(
            cfg,
            max_new_tokens=default_max_new,
            temperature=temperature,
            top_p=top_p,
        )

    if provider == "ollama":
        LOG.info("[GENERATION] Served by Ollama: %s", cfg.get("Generative_model") or cfg.get("Ollama_model") or "qwen3.5:9b")
        return load_ollama_llm(
            cfg,
            max_new_tokens=default_max_new,
            temperature=temperature,
            top_p=top_p,
            thinking=is_thinking_mode(mode),
        )

    model_id = cfg.get("Generative_model") or cfg.get("GENERATIVE_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
    repetition_penalty = float(cfg.get("Generation_repetition_penalty") or 1.08)
    no_repeat_ngram_size = int(cfg.get("Generation_no_repeat_ngram_size") or 4)

    from transformers import pipeline  # type: ignore
    from langchain_huggingface import HuggingFacePipeline

    LOG.info("[GENERATION] Served by in-process Transformers: %s", model_id)
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


_SECTION_HEADERS = (
    "[SECTION 1: WHO, WHERE, WHEN (The Setup)]",
    "[SECTION 2: WHAT (The Problem Starts)]",
    "[SECTION 3: TWIST/COMPLICATION (The Challenge)]",
    "[SECTION 4: HOW (The Big Action/Climax)]",
    "[SECTION 5: WHY/OUTCOME (The Moral and Conclusion)]",
)


def _flow_section_headers() -> str:
    """Fixed 5-section outline used by every generation and refine pass."""
    return "\n".join(_SECTION_HEADERS)


def build_story_prompt(
    *,
    query: str,
    facts_for_prompt: str,
    profile: LengthProfile,
) -> str:
    """Build the Step 3 first-draft prompt (shared with SSE streaming)."""
    from storyforge.rag.extraction import _get_generation_prompts

    prompts = _get_generation_prompts()
    return (
        f"{(prompts['story_system'] or '').strip()}\n\n"
        + prompts["story_user"]
        .format(
            query=query,
            section_headers=_flow_section_headers(),
            grounded_facts=facts_for_prompt,
            length_guidance=profile.guidance_text(),
        )
        .strip()
    )


def build_refine_prompt(
    *,
    query: str,
    facts_for_prompt: str,
    prior_draft: str,
    feedback: str,
    profile: LengthProfile,
) -> str:
    """Build the Step 3 refine prompt."""
    from storyforge.rag.extraction import _get_generation_prompts

    prompts = _get_generation_prompts()
    return (
        f"{(prompts['refine_system'] or '').strip()}\n\n"
        + prompts["refine_user"]
        .format(
            query=query,
            section_headers=_flow_section_headers(),
            grounded_facts=facts_for_prompt,
            prior_draft=prior_draft.strip(),
            feedback=feedback.strip(),
            length_guidance=profile.guidance_text(),
        )
        .strip()
    )


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


def _apply_attribution_gate(story: str, facts: tuple, cfg: dict[str, Any]) -> str:
    """Log (or optionally truncate) names not present in grounded facts."""
    story_body_for_check = without_sentence_starts(re.sub(r"^\[SECTION \d+:.*?\]\s*", "", story, flags=re.MULTILINE))
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




def _trim_to_sentence(text: str) -> str:
    """Drop a trailing cut-off fragment so a section ends on a finished sentence."""
    text = text.strip()
    if not text or text[-1] in _TERMINAL:
        return text
    cut = max(text.rfind(c) for c in ".!?")
    return text[: cut + 1] if cut > len(text) // 3 else text


def _invoke_nonempty(prompt: str, cfg: dict[str, Any], *, mode: Any, max_new_tokens: Optional[int], profile: LengthProfile) -> str:
    """One generation call; an empty thinking-mode reply is retried once with fast sampling."""
    text = str(_load_generation_llm(cfg, mode=mode, max_new_tokens=max_new_tokens, profile=profile).invoke(prompt) or "").strip()
    if not text and is_thinking_mode(mode):
        LOG.warning("Empty draft from thinking mode -- retrying once with fast sampling.")
        text = str(_load_generation_llm(cfg, mode="fast", max_new_tokens=max_new_tokens, profile=profile).invoke(prompt) or "").strip()
    if not text:
        raise RuntimeError(
            "Story generation returned an empty draft. "
            "Try mode=\'fast\' or a smaller \'length\' target."
        )
    return text


def _sections_to_write(prior_draft: Optional[str], profile: LengthProfile) -> dict[int, str]:
    """Sections of ``prior_draft`` worth keeping: present, finished, long enough (over-long ones are trimmed upstream)."""
    if not prior_draft:
        return {}
    kept: dict[int, str] = {}
    for i, body in _split_section_bodies(prior_draft).items():
        if (
            1 <= i <= len(_SECTION_HEADERS)
            and body.rstrip().endswith(_TERMINAL)
            and _sentence_count(body) >= profile.min_sentences_per_section
        ):
            kept[i] = body
    return kept


def _generate_sectioned(
    query: str,
    facts_for_prompt: str,
    cfg: dict[str, Any],
    *,
    mode: Any,
    profile: LengthProfile,
    prior_draft: Optional[str] = None,
    feedback: Optional[str] = None,
    headers: tuple[str, ...] = _SECTION_HEADERS,
    prompts: Optional[dict[str, str]] = None,
    roles: tuple[str, ...] = (),
    rewrite_sections: Optional[frozenset[int]] = None,
) -> str:
    """Opt-in writer: one short call per section, each with its own word and token budget.

    ``headers`` / ``prompts`` / ``roles`` let another method (see ``generation_arc``) reuse the loop
    with its own outline and prompt set; the defaults are the 5W1H outline and ``prompts.yaml``.

    A single pass overshoots the length target and can run out of tokens before SECTION 5;
    here the ending always gets its own budget. On a refine, only the missing, short, or
    unfinished sections are rewritten; the rest of ``prior_draft`` is kept. With ``rewrite_sections`` the
    caller names exactly which sections to rewrite (rule-driven loop); every other section of
    ``prior_draft`` is kept as written.
    """
    from storyforge.rag.extraction import _get_generation_prompts

    prompts = prompts or _get_generation_prompts()
    words = profile.words_per_section
    max_words = round(words * 1.3)
    tokens = min(length_token_cap(cfg), math.ceil(words * _SECTION_OVERSHOOT * 1.35 * 1.3))
    if prior_draft and rewrite_sections is not None:
        bodies = {i: b for i, b in _split_section_bodies(prior_draft).items() if 1 <= i <= len(headers) and i not in rewrite_sections}
    else:
        bodies = _sections_to_write(prior_draft, profile)
    if prior_draft and rewrite_sections is None and len(bodies) == len(headers):
        bodies = {}  # nothing structural to fix: rewrite everything using the reviewer feedback
    feedback_block = f"- Reviewer feedback to address: {feedback.strip()}" if feedback and feedback.strip() else ""
    last = len(headers)

    for i, header in enumerate(headers, start=1):
        if i in bodies:
            continue
        so_far = "\n\n".join(f"{headers[j - 1]}\n{bodies[j]}" for j in range(1, i) if j in bodies) or "(nothing yet)"
        ending_rule = (
            "This is the FINAL section: resolve the story and end on a finished, resolved final sentence."
            if i == last
            else "End on a complete sentence that leads into the next section."
        )
        prompt = (
            f"{(prompts['section_system'] or '').strip()}\n\n"
            + prompts["section_user"]
            .format(
                query=query,
                grounded_facts=facts_for_prompt,
                story_so_far=so_far,
                header=header,
                role=roles[i - 1] if i <= len(roles) else "",
                words=words,
                min_sentences=profile.min_sentences_per_section,
                max_words=max_words,
                ending_rule=ending_rule,
                feedback_block=feedback_block,
            )
            .strip()
        )
        raw = _invoke_nonempty(prompt, cfg, mode=mode, max_new_tokens=tokens, profile=profile)
        bodies[i] = _trim_to_sentence(re.sub(r"^\s*\[SECTION[^\]]*\]\s*", "", raw))
        LOG.info("[GENERATION] sectioned: wrote section %d (%d words, budget %d).", i, len(bodies[i].split()), words)
    return "\n\n".join(f"{h}\n{bodies[i]}" for i, h in enumerate(headers, start=1))


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
    length: Any = None,
    profile: Optional[LengthProfile] = None,
    rewrite_sections: Optional[frozenset[int]] = None,
) -> str:
    """Step 3: write or refine a 5-section story from grounded facts."""
    profile = profile or resolve_length_profile(cfg, length=length, mode=mode)
    formatted_facts = format_facts_for_prompt(parsed)
    facts_for_prompt = formatted_facts if formatted_facts else grounded_raw

    if str(cfg.get("Story_generation_method") or "arc").strip().lower() == "arc":
        from storyforge.rag.generation_arc import generate_arc

        story = generate_arc(
            query, facts_for_prompt, cfg, mode=mode, profile=profile,
            prior_draft=prior_draft, feedback=refine_feedback, rewrite_sections=rewrite_sections,
        )
        return _apply_attribution_gate(trim_overlong(story, profile), parsed.facts, cfg)

    if str(cfg.get("Story_generation_mode") or "single").strip().lower() == "sectioned":
        story = _generate_sectioned(
            query, facts_for_prompt, cfg, mode=mode, profile=profile,
            prior_draft=prior_draft, feedback=refine_feedback, rewrite_sections=rewrite_sections,
        )
        return _apply_attribution_gate(trim_overlong(story, profile), parsed.facts, cfg)

    if refine_feedback and prior_draft:
        story_prompt = build_refine_prompt(
            query=query,
            facts_for_prompt=facts_for_prompt,
            prior_draft=prior_draft,
            feedback=refine_feedback,
            profile=profile,
        )
    else:
        story_prompt = build_story_prompt(
            query=query,
            facts_for_prompt=facts_for_prompt,
            profile=profile,
        )

    story = _invoke_nonempty(story_prompt, cfg, mode=mode, max_new_tokens=max_new_tokens, profile=profile)
    return _apply_attribution_gate(story, parsed.facts, cfg)
