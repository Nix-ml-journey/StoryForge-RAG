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

With 16 GB VRAM you can run **7B–9B class models** at full BF16 precision, or
**13B–14B models** with INT4/GPTQ quantization (or via Ollama's own quantizations).
The project currently generates with **Ollama `qwen3.5:9b`** by default — not the
older Qwen2.5-1.5B baseline mentioned in early notes.

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

`Model_max_prompt_tokens: 12288`. Token floors: `Single_pass_fast_max_tokens: 3200`, `Single_pass_thinking_max_tokens: 4000`. Effective generation budget is now `max(floor, length-derived)` capped by `Story_length_max_new_tokens_cap`.

### 0.10 ✅ Unified story-length target

Raising token budgets alone did not grow stories past ~450 words — the prompt still asked for 3–6 sentences per section, and the accept gate only required a short draft.

**Fix:** `src/storyforge/rag/length_profile.py` derives prompt `{length_guidance}`, `max_new_tokens`, `min_words`, and `min_sentences_per_section` from one target.

Request field `length` accepts:

- presets: `short` (450) / `medium` (900) / `long` (1500) / `epic` (2200)
- narration duration: `"13min"` (uses `Story_length_words_per_minute`)
- explicit word count: `"1800"`

Config keys: `Story_length_presets`, `Story_length_default_fast` / `_thinking`, `Story_length_words_per_minute`, `Story_length_max_new_tokens_cap`.

Also raised `Story_generation_n_results` to 10 and `HF_grounded_facts_max_new_tokens` to 1600 so long targets have enough grounded material. Wired through generate / run_step / stream APIs. Tests in `tests/test_length_profile.py`.

**Fixed (see empty-draft recovery below):** thinking mode can still return an empty body in rare cases, but `generation.py` now retries once with fast sampling and raises a clear `RuntimeError` if that also fails. The length guard in `langchain_rag.py` and the agentic loop in `agentic_loop.py` both catch that error (pre-refine draft / best-so-far). `mode: "fast"` + `length: "13min"` is still the most reliable path for long targets.

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

### 2.4 ✅ Switch to structured output / JSON mode for Step 2

`HF_grounded_facts_json_mode: true` in `setup.yaml` / `setup.example.yaml`.

`_hf_chat_extract_json` in `rag/extraction.py` passes `response_format={"type": "json_object"}` to the HF `InferenceClient.chat_completion` call when this flag is on. If the backend raises `TypeError` (older API versions), it retries without the flag and logs a warning — no user action required.

With JSON mode active, the Step 2 model (Qwen3-8B via HF API) is constrained to valid JSON at the token level, eliminating markdown fences, trailing commas, and hallucinated keys. The `repair_json()` call in `parse_grounded_facts_json` remains as a safety net for the local-fallback path.

Contract tests in `tests/test_extraction.py` cover: json_mode adds `response_format`, disabled-by-config path, TypeError fallback, and the full `extract_grounded_facts` public API.

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

### 3.1 ✅ Add vLLM as an alternative high-throughput backend

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

**Implementation (done):**
- `generation_backend.py`: added `vllm_base_url()`, `vllm_model_id()`, `load_vllm_llm()` (uses `langchain_openai.ChatOpenAI`)
- `generation.py` `_load_generation_llm()`: vLLM branch dispatches before ollama/transformers
- `orchestration_routes.py` `_stream_story_sse()`: vLLM streaming path via `ChatOpenAI.astream`
- `setup.example.yaml`: `Generation_provider: vllm` + `vLLM_base_url` / `vLLM_model` keys

**To use:** set `Generation_provider: vllm` in `setup.yaml` and start the server above.

---

### 3.1b ✅ Empty-draft recovery

Thinking mode (`qwen3.5:9b` with `think=True`) occasionally returns an empty story body — the model emits only a `<think>…</think>` block, and `strip_thinking_tags()` removes everything.

**Implementation (done):**
- `generation.py` `generate_from_facts()`: after `gen_llm.invoke()`, if the story is empty and the mode is thinking, retries once with `mode="fast"` (fast sampling, no think block). If the retry also returns empty, raises `RuntimeError("Story generation returned an empty draft.")`.
- `langchain_rag.py` length guard: wraps the refine call in `try/except RuntimeError` and falls back to the original draft instead of propagating the error — the user always gets something back.
- `agentic_loop.py` `run_agentic_story_loop()`: the first pass only covered the 3-step path — the agentic loop called `generate_from_facts()` with no `try/except`, so a `RuntimeError` on any iteration still aborted the whole run and discarded every earlier iteration's best draft. Now wrapped per-iteration: catches `RuntimeError`, records a `generation_failed` entry in `iterations`, and stops with the best draft found so far (`stop_reason: "generation_failed_using_best_so_far"`), or a clean empty result (`stop_reason: "generation_failed"`) if it happens on the first iteration.
- `tests/test_empty_draft_recovery.py`: 4 tests covering thinking retry, both-empty `RuntimeError`, fast-mode `RuntimeError`, and length-guard fallback.
- `tests/test_agentic_loop.py`: 2 tests covering the agentic-loop failure path (first-iteration failure vs. failure-after-partial-progress).

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

### 3.5 ✅ Replace `requests`-based HF evaluation with a local evaluator

The evaluation step previously always called the HuggingFace Inference API (remote,
rate-limited, can fail) — twice per agentic-loop iteration when combined with Step 2's
own HF call. `Evaluation_mode: "local"` loads a small model in-process instead,
removing that round-trip and its rate-limit risk from every iteration.

**Implementation (done):**
- `evaluation.py`: `evaluate_model()` checks `Evaluation_mode` first — `"local"` short-circuits
  the HF/Gemini provider-priority chain entirely and returns a local evaluator descriptor.
- `_load_local_evaluator_model()` loads and caches (by model id + device) a small causal LM
  via Transformers; `_invoke_local_once()` runs it through the tokenizer's chat template.
- `_invoke_local_with_fallback()` catches any local failure (OOM, missing weights, etc.) and
  automatically retries through the existing HF → Gemini chain, so a bad local config
  degrades gracefully instead of failing the whole agentic loop.
- `Local_evaluation_device` defaults to `"cpu"` — loading a second model on `"cuda"` alongside
  Ollama's own CUDA context has been observed to starve VRAM on a 16 GB card (see the
  `Reranker_device` note). Set to `"cuda"` only if you have headroom.
- Tests in `tests/test_evaluation.py`: local mode selection, API-priority bypass, routing,
  fallback-on-failure, and raising when every backend fails.

**Config (`setup.yaml` / `setup.example.yaml`):**
```yaml
# "api" (default) or "local" (loads a small model in-process, no HF/Gemini round-trip)
Evaluation_mode: "api"
Local_evaluation_model: "Qwen/Qwen2.5-3B-Instruct"
Local_evaluation_device: "cpu"   # "cuda" only if you have VRAM to spare alongside Ollama
Local_evaluation_max_new_tokens: 700
```

**To use:** set `Evaluation_mode: "local"` in `setup.yaml`. First evaluation call in a
process loads and caches the model; every call after that (across all agentic-loop
iterations, and across HTTP requests within the same running server) reuses it.

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
| Unified story-length target | ✅ Done | ⭐⭐⭐⭐ | None (time cost only) |
| Structured JSON output (HF json_mode) | ✅ Done | ⭐⭐⭐ | None |
| vLLM as high-throughput backend | ✅ Done | ⭐⭐⭐⭐ | Same |
| Empty-draft recovery (3-step + agentic) | ✅ Done | ⭐⭐⭐ | None |
| Local evaluation model | ✅ Done | ⭐⭐⭐ | 0 (cpu default) / +6 GB (cuda) |
| INT4 quantization path | ⬜ Later | ⭐⭐ | −7 GB |

---

## Recommended next steps

1. **Try `qwen3.5:14b`** — `docker exec -it ollama ollama pull qwen3.5:14b` then swap `Generative_model` — quality lift if VRAM allows.
2. **Tune accept rate** — run agentic loop with `length: "long"` / `"13min"` and track word count vs target; adjust `Agentic_loop_accept_score` if stories are over-refined.
3. **Re-evaluate vLLM (3.1)** — relevant if you add concurrent users or want batch evaluation.
4. **INT4 quantization (3.2)** — if you want to run a 13B model with the same VRAM budget.
5. **Turn on local evaluation** — set `Evaluation_mode: "local"` in `setup.yaml` if HF API rate limits or latency are a bottleneck; keep `Local_evaluation_device: "cpu"` unless you've confirmed VRAM headroom alongside Ollama.
6. **Fix BGE passage-prefix convention** — ingest currently prefixes passages the same way as queries; BGE's documented recipe only prefixes queries. Fixing it means re-embedding the whole corpus.

> **Before any upgrade:** run `python -m pytest -q` as a regression check.
> Length-profile, attribution gate, evaluation, and agentic loop tests confirm the RAG pipeline is still correct.
