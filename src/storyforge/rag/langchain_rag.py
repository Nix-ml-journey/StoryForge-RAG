"""
Three-step RAG for story generation.

  Step 1 — retrieve: Chroma vector search (+ optional cross-encoder rerank)
  Step 2 — extract:  grounded facts as JSON (HF API, or local model fallback)
  Step 3 — generate: local Transformers model writes a 5-section story from facts

The agentic loop (`agentic_loop.py`) reuses the Step 1–3 primitives and adds
evaluate → refine / re-retrieve → accept.
"""

from __future__ import annotations

import logging

import os
import re

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Optional



from storyforge.config.config import load_config, load_prompts

from storyforge.rag.attribution import (

    ParsedFacts,

    attribution_violations,

    build_debug_attribution_stub,

    extract_named_entities_heuristic,

    format_facts_for_prompt,

    parse_grounded_facts_json,

)



from langchain_core.documents import Document

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFacePipeline



from huggingface_hub import InferenceClient



from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline



LOG = logging.getLogger(__name__)

# Process-wide caches (avoid reloading heavy models on every request).
_LOCAL_MODEL_CACHE: dict = {}      # model_id -> (tokenizer, model)
_RERANKER_CACHE: dict = {}         # model_id -> CrossEncoder
_VECTORSTORE_CACHE: dict = {}      # (chroma_dir, collection, embed_model) -> Chroma


def _get_reranker(model_id: str):
    """Load (or return cached) a cross-encoder reranker on GPU."""
    if model_id in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_id]
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOG.info("Loading reranker model %s on %s", model_id, device)
        reranker = CrossEncoder(model_id, device=device)
        _RERANKER_CACHE[model_id] = reranker
        return reranker
    except ImportError:
        LOG.warning(
            "sentence-transformers not installed — reranker disabled. "
            "Run: pip install sentence-transformers"
        )
        return None


def _rerank_docs(
    query: str,
    docs: list,
    *,
    reranker_model_id: str,
    top_n: int = 6,
) -> list:
    """Re-rank retrieved chunks by relevance; return top_n (or unchanged docs if reranker missing)."""
    if not docs:
        return docs
    reranker = _get_reranker(reranker_model_id)
    if reranker is None:
        return docs[:top_n]
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = [d for _, d in ranked[:top_n]]
    LOG.debug("Reranker kept %d/%d chunks (top_n=%d)", len(result), len(docs), top_n)
    return result





@dataclass(frozen=True)

class RAG3StepResult:

    content: str

    retrieval_context: str

    grounded_extraction: str

    grounded_facts: tuple[dict[str, Any], ...] = ()

    retrieval_chunks: tuple[dict[str, Any], ...] = ()

    debug_attribution: dict[str, Any] | None = None





def _get_paths_and_names(cfg: dict[str, Any]) -> tuple[Path, str, str]:

    base = Path(cfg.get("BASE_PATH") or Path(__file__).resolve().parents[1]).resolve()

    chroma_dir = (base / (cfg.get("Chroma_path") or "chroma_db")).resolve()

    collection = cfg.get("Chroma_collection_name") or "StoryForgeRag_v1"

    embed_model = cfg.get("Vector_store_model") or "all-MiniLM-L6-v2"

    return chroma_dir, str(collection), str(embed_model)





def _build_vectorstore(cfg: dict[str, Any]) -> Chroma:

    chroma_dir, collection, embed_model = _get_paths_and_names(cfg)

    # Reuse Chroma + embedding model across requests (~10s saved per call for BGE).
    cache_key = (chroma_dir, collection, embed_model)
    if cache_key in _VECTORSTORE_CACHE:
        LOG.debug("Reusing cached vectorstore (%s / %s)", collection, embed_model)
        return _VECTORSTORE_CACHE[cache_key]

    # Embeddings run on GPU when available (sentence-transformers under the hood).
    try:

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    except Exception:

        device = "cpu"

    # BGE expects a query prefix at search time only (not when ingesting chunks).
    # LangChain dropped query_instruction, so we subclass embed_query for BGE models.
    is_bge = "bge" in embed_model.lower()
    encode_kwargs: dict = {"normalize_embeddings": True}
    model_kwargs: dict = {"device": device}

    if is_bge:
        _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

        class _BGEEmbeddings(HuggingFaceEmbeddings):
            def embed_query(self, text: str) -> list[float]:
                return super().embed_query(_BGE_QUERY_PREFIX + text)

        embeddings: HuggingFaceEmbeddings = _BGEEmbeddings(
            model_name=embed_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    vectorstore = Chroma(
        collection_name=collection,
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
    )
    _VECTORSTORE_CACHE[cache_key] = vectorstore
    LOG.info("Vectorstore loaded and cached (%s / %s)", collection, embed_model)
    return vectorstore





def _docs_to_context(docs: list[Document]) -> str:

    parts: list[str] = []

    for d in docs:

        title = (d.metadata or {}).get("Title") or (d.metadata or {}).get("title") or "Unknown"

        parts.append(f"[{title}]\n{d.page_content}")

    return "\n\n".join(parts).strip()





def _docs_to_chunks(docs: list[Document]) -> list[dict[str, Any]]:

    out: list[dict[str, Any]] = []

    for d in docs:

        md = d.metadata or {}

        chunk_id = md.get("chunk_id") or md.get("id") or ""

        title = md.get("Title") or md.get("title") or "Unknown"

        out.append(

            {

                "chunk_id": str(chunk_id or ""),

                "title": str(title),

                "metadata": dict(md),

                "text": d.page_content,

            }

        )

    return out





def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:

    parts: list[str] = []

    for ch in chunks:

        cid = ch.get("chunk_id") or ""

        title = ch.get("title") or "Unknown"

        txt = ch.get("text") or ""

        parts.append(f"[CHUNK {cid} | {title}]\n{txt}")

    return "\n\n".join(parts).strip()





def _select_diverse_stories(

    docs: list[Document],

    *,

    n_stories: int = 3,

    chunks_per_story: int = 2,

) -> list[Document]:

    """Pick chunks from up to `n_stories` different titles so one book does not dominate retrieval."""

    by_title: dict[str, list[Document]] = {}

    for d in docs:

        md = d.metadata or {}

        title = str(md.get("Title") or md.get("title") or "Unknown").strip() or "Unknown"

        by_title.setdefault(title, []).append(d)



    picked: list[Document] = []

    picked_titles: set[str] = set()

    for title, group in by_title.items():

        if len(picked_titles) >= n_stories:

            break

        picked_titles.add(title)

        picked.extend(group[: max(1, chunks_per_story)])



    target = max(1, n_stories) * max(1, chunks_per_story)

    if len(picked) < min(target, len(docs)):

        seen = {id(d) for d in picked}

        for d in docs:

            if id(d) in seen:

                continue

            picked.append(d)

            if len(picked) >= target:

                break

    return picked





# Lives in attribution.py for unit tests; alias keeps imports local.
_format_facts_for_prompt = format_facts_for_prompt


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





def _load_or_get_cached_local_model(model_id: str, cfg: dict[str, Any]):
    """
    Load the local causal LM once per model_id.

    Step 3 (story) and Step 2 fallback (facts) share this cache so the GPU
    does not load the same weights twice.
    """
    if model_id in _LOCAL_MODEL_CACHE:
        LOG.debug("Reusing cached local model: %s", model_id)
        return _LOCAL_MODEL_CACHE[model_id]

    import torch

    LOG.info("Loading local model (first use): %s", model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Flash Attention 2 when installed; otherwise standard eager attention.
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else None
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        LOG.info("Flash Attention 2 enabled for %s", model_id)
    except ImportError:
        attn_impl = "eager"
        LOG.info("flash-attn not installed — using eager attention. Run: pip install flash-attn --no-build-isolation")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda" if use_cuda else None,
        dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    _LOCAL_MODEL_CACHE[model_id] = (tok, model)
    return tok, model


def _load_local_generation_llm(cfg: dict[str, Any], *, max_new_tokens: Optional[int] = None) -> HuggingFacePipeline:
    model_id = cfg.get("Generative_model") or cfg.get("GENERATIVE_MODEL") or "Qwen/Qwen2.5-1.5B-Instruct"
    if max_new_tokens is not None:
        default_max_new = int(max_new_tokens)
    else:
        default_max_new = int(cfg.get("Single_pass_fast_max_tokens") or 768)

    tok, model = _load_or_get_cached_local_model(model_id, cfg)
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=default_max_new,
        do_sample=True,
        temperature=float(cfg.get("Generation_fast_temperature") or 0.4),
        top_p=float(cfg.get("Generation_fast_top_p") or 0.8),
        pad_token_id=tok.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)





def _load_local_facts_llm(cfg: dict[str, Any]) -> HuggingFacePipeline:
    """Step 2 fallback when the HF API is unavailable (same cached weights as Step 3)."""
    model_id = cfg.get("Generative_model") or cfg.get("GENERATIVE_MODEL") or "Qwen/Qwen2.5-1.5B-Instruct"
    max_new = int(cfg.get("HF_grounded_facts_max_new_tokens") or cfg.get("Layer1_max_tokens") or 300)
    temperature = float(cfg.get("HF_grounded_facts_temperature") or cfg.get("Layer1_temperature") or 0.1)

    tok, model = _load_or_get_cached_local_model(model_id, cfg)
    fact_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=float(cfg.get("Generation_fast_top_p") or 0.8),
        pad_token_id=tok.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=fact_pipe)





def _hf_token(cfg: dict[str, Any]) -> str:

    return (

        (cfg.get("facehugging_api") or "").strip()

        or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip()

        or os.environ.get("HF_TOKEN", "").strip()

        or os.environ.get("HUGGINGFACE_API_KEY", "").strip()

    )





def _hf_chat_extract_json(

    *,

    cfg: dict[str, Any],

    system: str,

    user: str,

) -> str:

    """
    Step 2 via Hugging Face Inference API (chat_completion).

    Uses the router chat endpoint because many instruct models are only served there.
    """

    model_id = (

        cfg.get("HF_grounded_facts_model")

        or cfg.get("HF_structured_profile_model")

        or cfg.get("HF_evaluation_model")

        or "Qwen/Qwen2.5-1.5B-Instruct"

    )

    temperature = float(cfg.get("HF_grounded_facts_temperature") or cfg.get("Layer1_temperature") or 0.1)

    max_new = int(cfg.get("HF_grounded_facts_max_new_tokens") or cfg.get("Layer1_max_tokens") or 300)

    token = _hf_token(cfg)

    if not token:

        raise ValueError("Missing Hugging Face token for grounded facts extraction (facehugging_api / env).")



    client = InferenceClient(token=token)

    resp = client.chat_completion(

        model=str(model_id),

        messages=[

            {"role": "system", "content": str(system or "").strip()},

            {"role": "user", "content": str(user or "").strip()},

        ],

        max_tokens=max_new,

        temperature=temperature,

    )

    # Response may be an object or a plain dict depending on huggingface_hub version.

    try:

        return (resp.choices[0].message.content or "").strip()  # type: ignore[attr-defined]

    except Exception:

        # Dict-shaped fallback

        if isinstance(resp, dict):

            choices = resp.get("choices") or []

            if choices and isinstance(choices[0], dict):

                msg = choices[0].get("message") or {}

                if isinstance(msg, dict) and msg.get("content"):

                    return str(msg.get("content") or "").strip()

        return str(resp).strip()





def _get_generation_prompts() -> dict[str, str]:

    p = (load_prompts() or {}).get("generation") or {}

    return {

        "facts_system": p.get("grounded_facts_system") or "Return grounded facts as JSON.",

        "facts_user": p.get("grounded_facts_user")

        or "QUERY:\n{query}\n\nCHUNKS:\n{retrieval_chunks}\n\nReturn JSON facts.",

        "story_system": p.get("grounded_story_system") or "Write a grounded story.",

        "story_user": p.get("grounded_story_user")

        or "QUERY:\n{query}\n\nSECTION_HEADERS:\n{section_headers}\n\nGROUNDED_FACTS:\n{grounded_facts}",

        "refine_system": p.get("grounded_story_refine_system")
        or "You revise a grounded story draft using reviewer feedback while staying strictly grounded.",

        "refine_user": p.get("grounded_story_refine_user")
        or (
            "QUERY:\n{query}\n\nSECTION_HEADERS:\n{section_headers}\n\n"
            "GROUNDED_FACTS:\n{grounded_facts}\n\nPREVIOUS_DRAFT:\n{prior_draft}\n\n"
            "REVIEWER_FEEDBACK:\n{feedback}\n\nRewrite the complete story."
        ),

    }





def retrieve_docs(
    query: str,
    cfg: dict[str, Any],
    *,
    n_stories: int = 3,
    chunks_per_story: int = 2,
    k_boost: float = 1.0,
    use_reranker: Optional[bool] = None,
) -> list[Document]:
    """
    Step 1: vector search in Chroma, spread across multiple story titles.

    k_boost — widen retrieval (agentic re-retrieve uses this).
    use_reranker — override setup.yaml Reranker_enabled when set.
    """
    vectorstore = _build_vectorstore(cfg)
    base_k = int(cfg.get("Story_generation_n_results") or 3)
    k = max(base_k, n_stories * max(1, chunks_per_story) * 4)
    k = int(round(k * max(1.0, float(k_boost))))
    docs = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)
    docs = _select_diverse_stories(docs, n_stories=n_stories, chunks_per_story=chunks_per_story)

    # Optional cross-encoder rerank (see Reranker_enabled / Reranker_model in setup.yaml).
    if use_reranker is None:
        reranker_on = str(cfg.get("Reranker_enabled") or "").strip().lower() not in ("false", "0", "no", "")
    else:
        reranker_on = bool(use_reranker)
    if reranker_on:
        reranker_model_id = str(cfg.get("Reranker_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        top_n = int(cfg.get("Story_generation_rerank_top_n") or 6)
        docs = _rerank_docs(query, docs, reranker_model_id=reranker_model_id, top_n=top_n)
    return docs


def extract_grounded_facts(
    query: str,
    chunks: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[str, "ParsedFacts"]:
    """
    Step 2: turn retrieved chunks into grounded facts (JSON).

    HF API first; local model if the API fails. Returns (raw_text, parsed_facts).
    """
    prompts = _get_generation_prompts()
    try:
        grounded_raw = _hf_chat_extract_json(
            cfg=cfg,
            system=prompts["facts_system"],
            user=prompts["facts_user"].format(query=query, retrieval_chunks=_format_chunks_for_prompt(chunks)),
        )
    except Exception as e:
        LOG.warning("HF routed extraction unavailable; falling back to local extraction. Error: %s", e)
        facts_llm = _load_local_facts_llm(cfg)
        facts_prompt = (
            f"{(prompts['facts_system'] or '').strip()}\n\n"
            + prompts["facts_user"].format(query=query, retrieval_chunks=_format_chunks_for_prompt(chunks)).strip()
        )
        grounded_raw = str(facts_llm.invoke(facts_prompt) or "").strip()

    try:
        parsed = parse_grounded_facts_json(grounded_raw)
    except Exception:
        parsed = ParsedFacts(facts=(), raw={})
    return grounded_raw, parsed


def _apply_attribution_gate(story: str, facts: tuple, cfg: dict[str, Any]) -> str:
    """
    Optional post-check for names not present in grounded facts.

    Default: log only (Attribution_gate_truncate: false in setup.yaml).
    Hard truncation used to fire on almost every story because the heuristic
    treats many capitalized words as "entities". Grounding is already enforced
    in Step 2 and the generation prompt.

    Set Attribution_gate_truncate: true to restore cutting the story to 6 paragraphs.
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


def generate_from_facts(
    query: str,
    parsed: "ParsedFacts",
    grounded_raw: str,
    cfg: dict[str, Any],
    *,
    refine_feedback: Optional[str] = None,
    prior_draft: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> str:
    """
    Step 3: write or refine a 5-section story from grounded facts.

    With refine_feedback + prior_draft, runs the refine prompt instead of a fresh draft.
    """
    prompts = _get_generation_prompts()

    # Bullet list for the prompt when JSON parsed OK; else pass through raw extraction.
    formatted_facts = _format_facts_for_prompt(parsed)
    facts_for_prompt = formatted_facts if formatted_facts else grounded_raw

    gen_llm = _load_local_generation_llm(cfg, max_new_tokens=max_new_tokens)

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


def generate_story_3step_langchain(
    query: str,
    *,
    cfg: Optional[dict[str, Any]] = None,
    n_stories: int = 3,
    chunks_per_story: int = 2,
    show_progress: bool = True,
    debug: bool = False,
) -> RAG3StepResult:
    cfg = cfg or load_config()

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(total=3, desc="RAG 3-step", unit="step")
        except Exception:
            pbar = None

    docs = retrieve_docs(query, cfg, n_stories=n_stories, chunks_per_story=chunks_per_story)
    chunks = _docs_to_chunks(docs)
    retrieval_context = _docs_to_context(docs)
    if pbar:
        pbar.update(1)

    grounded_raw, parsed = extract_grounded_facts(query, chunks, cfg)
    if pbar:
        pbar.update(1)

    story = generate_from_facts(query, parsed, grounded_raw, cfg)
    if pbar:
        pbar.update(1)
        pbar.close()

    facts = parsed.facts
    debug_payload = None
    if debug:
        debug_payload = build_debug_attribution_stub(story=story, facts=facts)

    return RAG3StepResult(
        content=story,
        retrieval_context=retrieval_context,
        grounded_extraction=grounded_raw,
        grounded_facts=tuple([f.__dict__ for f in facts]),
        retrieval_chunks=tuple(chunks),
        debug_attribution=debug_payload,
    )
