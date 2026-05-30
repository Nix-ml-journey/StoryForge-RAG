# StoryForge-RAG — Upgrade Roadmap
## Targeting RTX 5060 Ti 16 GB (Blackwell)

This document covers every meaningful upgrade available to you right now, grouped by
impact and effort. Every item here has been checked against your hardware constraint:
**RTX 5060 Ti, 16 GB VRAM, Blackwell architecture**.

Nothing in this document requires a cloud GPU or a machine upgrade.

---

## Hardware Baseline (What You Have)

| Spec | Value |
|------|-------|
| GPU | RTX 5060 Ti |
| Architecture | Blackwell (GB206) |
| VRAM | 16 GB GDDR7 |
| CUDA Compute | 12.0+ |
| FP16 Tensor Cores | Yes |
| BF16 | Yes |
| Flash Attention 2 | Yes (Blackwell natively supported) |
| INT8 / INT4 (bitsandbytes) | Yes |

With 16 GB VRAM you can run **7B–8B class models** at full BF16 precision, or
**13B–14B models** with INT4/GPTQ quantization. This is a meaningful step up from
the current Qwen2.5-1.5B the project uses.

---

## PRIORITY 1 — High impact, low risk (do these first)

### 1.1 Upgrade the generation model from 1.5B → 7B

**Current:** `Qwen/Qwen2.5-1.5B-Instruct` (Step 3 generation, `setup.yaml: Generative_model`)

**Recommended:** `Qwen/Qwen2.5-7B-Instruct` — fits in 16 GB VRAM at BF16.

| Model | VRAM (BF16) | Story quality |
|-------|-------------|---------------|
| Qwen2.5-1.5B-Instruct | ~3.5 GB | Baseline |
| Qwen2.5-7B-Instruct | ~14.5 GB | Noticeably better coherence and vocabulary |
| Qwen3-4B (2025) | ~8.5 GB | Better instruction following than 2.5-7B |
| Qwen3-8B (2025) | ~16 GB | Near-ceiling for full BF16 on 16 GB |

**Change in `setup.yaml`:**
```yaml
Generative_model: "Qwen/Qwen2.5-7B-Instruct"
# Larger model needs more tokens to breathe:
Single_pass_fast_max_tokens: 1800
Model_max_prompt_tokens: 6144
```

**No code changes needed.** The pipeline already loads via `AutoModelForCausalLM`
and uses the config keys. Just change the model name and token budgets.

> **VRAM note:** Qwen2.5-7B at BF16 uses ~14.5 GB. You have ~1.5 GB headroom.
> If you hit OOM, drop to `Generation_fast_temperature: 0.3` and
> `Single_pass_fast_max_tokens: 1500`, or switch to the 4B Qwen3 variant.

---

### 1.2 Enable Flash Attention 2 for faster inference

Blackwell supports Flash Attention 2 natively. Enabling it reduces memory bandwidth
usage and speeds up generation, especially for longer prompts.

**In `langchain_rag.py`** — find where the model is loaded with `AutoModelForCausalLM.from_pretrained`
and add one argument:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",  # ← add this
)
```

**Install requirement:**
```
pip install flash-attn --no-build-isolation
```

Expected gain: 20–40% faster generation, slightly lower peak VRAM.

> Flash Attention 2 requires the model to be loaded in BF16 or FP16 (not FP32).
> The current pipeline already uses BF16, so this should work out of the box.

---

### 1.3 Upgrade the grounded-facts extraction model (Step 2)

**Current:** `Qwen/Qwen2.5-1.5B-Instruct` via HuggingFace API

**Recommended:** `Qwen/Qwen2.5-7B-Instruct` (same model as generation) — better
JSON extraction accuracy, fewer malformed outputs, fewer facts dropped by the
`parse_grounded_facts_json` repair step.

```yaml
HF_grounded_facts_model: "Qwen/Qwen2.5-7B-Instruct"
HF_grounded_facts_max_new_tokens: 700
```

This is an API call (HuggingFace Inference), not a local load, so no VRAM cost.

---

## PRIORITY 2 — Significant quality upgrades, moderate effort

### 2.1 Add a cross-encoder reranker after Chroma retrieval

**What it does:** After Chroma returns the top-K chunks by vector similarity, a
reranker scores each chunk against the query using a small cross-encoder model.
The top-N reranked results go to Step 2. This significantly improves which facts
the story is grounded in.

**Recommended model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- VRAM: ~200 MB (runs on GPU alongside the generation model)
- Latency: <100 ms for 10–20 chunks

**Install:**
```
pip install sentence-transformers
```

**Where to add it:** In `langchain_rag.py`, after `_select_diverse_stories()`:

```python
from sentence_transformers import CrossEncoder

_RERANKER = None

def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
    return _RERANKER

def _rerank_docs(query: str, docs: list, top_n: int = 6) -> list:
    if not docs:
        return docs
    reranker = _get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]
```

Add `Story_generation_rerank_top_n: 6` to `setup.yaml` and `setup.example.yaml`.

---

### 2.2 Upgrade the embedding model for better retrieval

**Current:** `all-MiniLM-L6-v2` (384-dim, 22 M params)

**Recommended:** `BAAI/bge-base-en-v1.5` (768-dim, 109 M params)

| Model | Dim | MTEB score | VRAM |
|-------|-----|------------|------|
| all-MiniLM-L6-v2 | 384 | 56.3 | ~90 MB |
| BAAI/bge-base-en-v1.5 | 768 | 63.6 | ~440 MB |
| BAAI/bge-large-en-v1.5 | 1024 | 64.2 | ~1.3 GB |

Better embeddings → better chunk retrieval → better grounded facts → better story.

**Change in `setup.yaml`:**
```yaml
Vector_store_model: "BAAI/bge-base-en-v1.5"
```

> **Important:** You must re-ingest your entire Chroma collection after changing the
> embedding model. Old embeddings (384-dim) are incompatible with new ones (768-dim).
> Run `POST /vector_store/reset` then `POST /vector_store/ingest_stories`.

---

### 2.3 Add hybrid search (BM25 + semantic)

**What it does:** BM25 (keyword-based) retrieval catches exact-name matches that
vector similarity misses (character names, place names, rare words). Combining it
with semantic retrieval via Reciprocal Rank Fusion (RRF) gives the best of both.

**Install:**
```
pip install rank-bm25
```

**Where to add it:** In `vector_store/ingest_stories.py`, build a BM25 index at
ingest time. In `langchain_rag.py`, run both retrieval paths and merge with RRF
before the reranker step.

Config key to add to `setup.yaml`:
```yaml
# Weight for BM25 vs semantic in hybrid fusion (0.0 = pure semantic, 1.0 = pure BM25)
Hybrid_bm25_weight: 0.3
```

---

### 2.4 Switch to structured output / JSON mode for Step 2

The current Step 2 prompt asks the model to return JSON but the model can deviate,
requiring `repair_json()`. Structured generation forces valid JSON every time by
constraining the token sampling to the grammar.

**Option A — Use `outlines` library (best, works with local Transformers):**
```
pip install outlines
```
Constrains the grounded-facts extraction to a valid JSON schema at the token level.
No hallucinated keys, no trailing commas, no markdown fences ever.

**Option B — Use `response_format={"type": "json_object"}` (if you switch to vLLM):**
See section 3.2 below.

---

## PRIORITY 3 — Structural / tooling upgrades

### 3.1 Serve the local model with vLLM instead of bare Transformers

**What it does:** vLLM uses PagedAttention for dramatically higher throughput and
can serve multiple requests concurrently without OOM. It also exposes an
OpenAI-compatible REST API that the LangChain pipeline can call via
`ChatOpenAI(base_url="http://localhost:8001/v1")`.

**Install:**
```
pip install vllm
```

**Start vLLM server (run alongside your FastAPI app):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8001
```

**Benefits on your hardware:**
- 3–5× higher throughput (tokens/sec) vs bare Transformers
- Native Flash Attention 2 already enabled by default in vLLM
- Continuous batching — multiple API calls don't OOM
- `--dtype bfloat16` is the right choice for Blackwell

**Change needed in `langchain_rag.py`:** Replace the `pipeline` / `AutoModelForCausalLM`
block with a `langchain_openai.ChatOpenAI` call pointing to `localhost:8001`.

---

### 3.2 Add a 4-bit quantized model path for lower VRAM

If you ever want to run a 13B model (e.g. `Qwen2.5-14B-Instruct-GPTQ-Int4`) or just
free up VRAM headroom for the reranker:

**Install:**
```
pip install bitsandbytes>=0.43.0 auto-gptq optimum
```

**In model loading:**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda",
)
```

Add a config toggle to `setup.yaml`:
```yaml
# Options: "bf16" (default, best quality), "int4" (half VRAM, slight quality drop)
Generation_precision: "bf16"
```

---

### 3.3 Project structure — split `langchain_rag.py`

`langchain_rag.py` is the largest file in the project. A clean split along its
natural boundaries will make it easier to test and extend:

| New file | What moves there |
|----------|-----------------|
| `rag/retrieval.py` | `_select_diverse_stories()`, `_build_retriever()`, Chroma query logic |
| `rag/extraction.py` | Step 2: HF API call, `_invoke_hf_*`, grounded-facts parsing |
| `rag/generation.py` | Step 3: model load/cache, `_generate_story_section()`, token budget logic |
| `rag/langchain_rag.py` | Keep only `generate_story_3step_langchain()` as the top-level orchestrator |

This maps directly to the 3-step pipeline and makes each step independently testable.

---

### 3.4 Add a streaming endpoint for story generation

The current `/create-eval/story_generate` waits for the full story before
returning. With vLLM (section 3.1) or raw Transformers `TextIteratorStreamer`, you
can stream tokens back as Server-Sent Events (SSE), which gives a much better UX.

**FastAPI SSE endpoint:**
```python
from fastapi.responses import StreamingResponse

@create_eval_router.post("/story_generate_stream")
async def story_generate_stream(request: StoryGenerateRequest):
    async def token_generator():
        for token in generate_tokens_streaming(request.query, ...):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(token_generator(), media_type="text/event-stream")
```

---

### 3.5 Replace `requests`-based HF evaluation with a local evaluator

The evaluation step currently calls the HuggingFace Inference API (remote, rate-limited,
can fail). With a 7B generation model loaded locally you have enough VRAM — after
generation — to run evaluation on the same GPU using a smaller local model.

**Approach:** After generation completes and the generation model is unloaded (or
using `torch.cuda.empty_cache()`), load `Qwen/Qwen2.5-3B-Instruct` locally for
evaluation. This eliminates the HF API dependency entirely for evaluation.

Add to `setup.yaml`:
```yaml
# "api" (current) or "local" (loads model on GPU after generation)
Evaluation_mode: "api"
Local_evaluation_model: "Qwen/Qwen2.5-3B-Instruct"
```

---

## Quick Summary Table

| Upgrade | Impact | Effort | VRAM cost | Code change |
|---------|--------|--------|-----------|-------------|
| 1.1 Qwen2.5-7B generation model | ⭐⭐⭐⭐⭐ | Minimal | +11 GB | Config only |
| 1.2 Flash Attention 2 | ⭐⭐⭐ | Low | −1 GB | 1 line |
| 1.3 Qwen2.5-7B for extraction | ⭐⭐⭐ | Minimal | None (API) | Config only |
| 2.1 Cross-encoder reranker | ⭐⭐⭐⭐ | Low | +200 MB | ~30 lines |
| 2.2 BGE-base embedding upgrade | ⭐⭐⭐⭐ | Low | +350 MB | Config + re-ingest |
| 2.3 Hybrid BM25+semantic search | ⭐⭐⭐ | Medium | None | ~60 lines |
| 2.4 Structured JSON output | ⭐⭐⭐ | Medium | None | ~40 lines |
| 3.1 vLLM serving | ⭐⭐⭐⭐ | Medium | Same | ~50 lines |
| 3.2 INT4 quantization path | ⭐⭐ | Low | −7 GB | ~20 lines |
| 3.3 Split langchain_rag.py | ⭐⭐ | Medium | None | Refactor only |
| 3.4 Streaming endpoint | ⭐⭐⭐ | Medium | None | ~40 lines |
| 3.5 Local evaluation model | ⭐⭐ | Medium | +6 GB (post-gen) | ~60 lines |

---

## Recommended Order of Execution

1. **First session:** Do 1.1 (model upgrade) + 1.3 (extraction model) — config only, immediate quality lift.
2. **Second session:** Do 1.2 (Flash Attention 2) + 2.1 (reranker) — low code, high RAG quality gain.
3. **Third session:** Do 2.2 (BGE embeddings) + re-ingest — needs a re-ingest run but is config-driven.
4. **Fourth session:** Do 3.1 (vLLM) — replaces the model loading block, unlocks streaming.
5. **Later:** 2.3 hybrid search, 2.4 structured output, 3.3 split file, 3.5 local eval.

> **Before any upgrade:** run `python -m pytest tests/test_attribution_gate.py -q` as a
> regression check. The attribution gate and grounded facts tests are the fastest way to
> confirm the RAG pipeline is still correct after a change.
