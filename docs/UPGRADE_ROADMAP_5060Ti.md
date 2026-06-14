# StoryForge-RAG — Upgrade Roadmap
## Targeting RTX 5060 Ti 16 GB (Blackwell)

> **Status legend:** ✅ Done · 🔄 In progress · ⬜ Not started

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

## PRIORITY 0 — Completed ✅

The following upgrades have been implemented and are active in the current codebase.

### 0.1 ✅ Switch Step 3 generation to Ollama in Docker

**Replaced:** bare `AutoModelForCausalLM` (loaded into Python process VRAM)  
**With:** `qwen3.5:9b` served by Ollama in a Docker container with NVIDIA GPU passthrough.

Benefits: faster cold start (model loaded once, stays warm), no Python VRAM overhead, streaming support via `ChatOllama.astream`.

`docker-compose.yml` configures the service; `Ollama_base_url` in `setup.yaml` points the pipeline at `http://localhost:11434`.

Thinking-mode support: `ChatOllama(think=True/False)` and `strip_thinking_tags()` in `generation_backend.py` handle `<think>...</think>` blocks cleanly.

### 0.2 ✅ Upgrade embedding model: all-MiniLM → BGE-base

`Vector_store_model: "BAAI/bge-base-en-v1.5"` with a BGE query-prefix subclass in `retrieval.py`. MTEB score 63.6 vs 56.3. Re-ingest was run after switching.

### 0.3 ✅ Add cross-encoder reranker

`Reranker_enabled: true`, `Reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"` in `setup.yaml`. Runs after Chroma retrieval. ~200 MB VRAM.

### 0.4 ✅ Add agentic loop

`Agentic_loop_enabled: true` — evaluate → REFINE / RE_RETRIEVE / ACCEPT. Controlled by score thresholds in `setup.yaml`. See `src/storyforge/rag/agentic_loop.py`.

### 0.5 ✅ Upgrade Step 2 extraction model to Qwen3-8B

`HF_grounded_facts_model: "Qwen/Qwen3-8B"` in `setup.yaml`. Better JSON extraction accuracy via HF Inference API (no local VRAM cost).

### 0.6 ✅ Split `langchain_rag.py` into focused modules

| New file | Responsibility |
|----------|---------------|
| `rag/retrieval.py` | Step 1: Chroma vector + hybrid BM25 retrieval |
| `rag/extraction.py` | Step 2: HF API grounded facts, local fallback |
| `rag/generation.py` | Step 3: model load/cache, story generation, attribution gate |
| `rag/langchain_rag.py` | Thin orchestrator + backward-compat re-exports |

`agentic_loop.py` imports continue to resolve from `langchain_rag.py` unchanged.

### 0.7 ✅ Hybrid BM25 + dense retrieval (RRF fusion)

`Hybrid_search_enabled: true`, `Hybrid_bm25_weight: 0.3` in `setup.yaml`.  
Implemented in `rag/retrieval.py` (`_bm25_rank_docs` + `_rrf_fuse`). Requires `pip install rank-bm25`.

`retrieve_docs` also accepts `filter_metadata` for Chroma `where`-filter targeting a specific series or story type.

### 0.8 ✅ Streaming endpoint

`POST /orchestration/generate_stream` — SSE endpoint. Steps 1–2 run synchronously; Step 3 streams tokens from Ollama via `ChatOllama.astream`. Tests in `tests/test_api_contracts.py`.

### 0.9 ✅ Expand context window + token budgets

`Model_max_prompt_tokens: 12288`, `Single_pass_fast_max_tokens: 3200`, `Single_pass_thinking_max_tokens: 4000`.

---

## PRIORITY 1 — High impact, low risk (do these first)

### 1.1 ✅ Upgrade the generation model from 1.5B → 7B (done via Ollama)

**Current:** `qwen3.5:9b` served by Ollama (Step 3)

**Recommended:** `Qwen/Qwen2.5-7B-Instruct` — fits in 16 GB VRAM at BF16.

| Model | VRAM (BF16) | Story quality |
|-------|-------------|---------------|
| Qwen2.5-1.5B-Instruct | ~3.5 GB | Baseline |
| Qwen2.5-7B-Instruct | ~14.5 GB | Noticeably better coherence and vocabulary |
| Qwen3-4B (2025) | ~8.5 GB | Better instruction following than 2.5-7B |
| Qwen3-8B (2025) | ~16 GB | Near-ceiling for full BF16 on 16 GB |

Since we switched to Ollama in Docker, the generation model is managed outside the Python process.

To upgrade the Ollama model:
```bash
docker exec -it ollama ollama pull qwen3.5:14b   # or any model that fits VRAM
```
Then update `setup.yaml: Generative_model: "qwen3.5:14b"`.

> With 16 GB VRAM, `qwen3.5:9b` runs comfortably. The 14B variant requires ~9 GB; try it if you want stronger writing at the cost of ~50 % slower generation.

---

### 1.2 ✅ Enable Flash Attention 2 (done — automatic for Transformers fallback path)

`generation.py` tries `flash_attention_2` at model load time and logs which path was taken. Ollama manages its own attention backend internally.

---

### 1.3 ✅ Upgrade the grounded-facts extraction model (done — now Qwen3-8B)

`HF_grounded_facts_model: "Qwen/Qwen3-8B"` in `setup.yaml`. HF API, no VRAM cost.

---

## PRIORITY 2 — Significant quality upgrades, moderate effort

### 2.1 ✅ Cross-encoder reranker — done (see 0.3)

---

### 2.2 ✅ BGE-base embeddings — done (see 0.2)

---

### 2.3 ✅ Hybrid BM25 + semantic search — done (see 0.7)

---

### 2.4 ⬜ Switch to structured output / JSON mode for Step 2

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

### 3.0 ✅ Ollama as local model server (done — replaces bare Transformers)

Ollama runs `qwen3.5:9b` in Docker with NVIDIA GPU passthrough. The FastAPI app talks to it at `http://localhost:11434` via `ChatOllama`. Benefits vs bare Transformers:

- Model stays warm between requests (no cold-start reload).
- Streaming via `ChatOllama.astream` — used by `POST /orchestration/generate_stream`.
- GPU memory managed by Ollama outside the Python process.
- `docker compose up -d` is the only required startup command.

To switch model: `docker exec -it ollama ollama pull <model>` then update `Generative_model` in `setup.yaml`.

---

### 3.1 ⬜ Add vLLM as an alternative high-throughput backend

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

### 3.3 ✅ Split `langchain_rag.py` — done (see 0.6)

---

### 3.4 ✅ Streaming endpoint — done (see 0.8)

`POST /orchestration/generate_stream` — SSE with per-token Ollama streaming. Tests in `tests/test_api_contracts.py`.

---

### 3.5 ⬜ Replace `requests`-based HF evaluation with a local evaluator

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

| Upgrade | Status | Impact | VRAM cost |
|---------|--------|--------|-----------|
| Ollama Docker for generation | ✅ Done | ⭐⭐⭐⭐⭐ | GPU managed by Ollama |
| BGE-base embeddings | ✅ Done | ⭐⭐⭐⭐ | +350 MB |
| Cross-encoder reranker | ✅ Done | ⭐⭐⭐⭐ | +200 MB |
| Agentic loop | ✅ Done | ⭐⭐⭐⭐ | None extra |
| Qwen3-8B Step 2 extraction | ✅ Done | ⭐⭐⭐ | None (API) |
| Split `langchain_rag.py` | ✅ Done | ⭐⭐ (DX) | None |
| Hybrid BM25+dense (RRF) | ✅ Done | ⭐⭐⭐ | None |
| SSE streaming endpoint | ✅ Done | ⭐⭐⭐ | None |
| Context window expansion | ✅ Done | ⭐⭐⭐ | +~1 GB |
| Structured JSON output (outlines) | ⬜ Next | ⭐⭐⭐ | None |
| vLLM as high-throughput backend | ⬜ Later | ⭐⭐⭐⭐ | Same |
| INT4 quantization path | ⬜ Later | ⭐⭐ | −7 GB |
| Local evaluation model | ⬜ Later | ⭐⭐ | +6 GB (post-gen) |

---

## Recommended next steps

1. **Try `qwen3.5:14b`** — `docker exec -it ollama ollama pull qwen3.5:14b` then swap `Generative_model` — immediate story quality lift if VRAM allows.
2. **Structured JSON output (2.4)** — `outlines` library eliminates `repair_json()` calls entirely.
3. **Re-evaluate vLLM (3.1)** — relevant if you add concurrent users or want batch evaluation.
4. **Local evaluation model (3.5)** — useful if HF API rate limits become a bottleneck.

> **Before any upgrade:** run `python -m pytest -q` as a regression check.
> The attribution gate, evaluation, and agentic loop tests confirm the RAG pipeline is still correct.
