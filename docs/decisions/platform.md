# Platform decisions (API surface and models)

Each ADR below keeps its original number. Newer decisions are appended; superseded ones keep their status.

---

## ADR-0004: Remove the vector-store insert and update routes

### Status
Accepted

### Date
2026-10-05

### Context
`/vector_store/insert` and `/update` wrote through `Collection.add/update` on a collection created
without an embedding function. Rows written that way carry no usable embedding for BGE retrieval
and the update path could not re-embed, so data written through them was invisible or stale.

### Decision
Delete both routes, their request/response models, the Orchestrator methods, the
`response_parameter` helpers, and `update_data`. `/delete` stays (ID-based, needs no embedding).
Content enters the index only through `scripts/reset_and_ingest.py` / the ingest routes.

### Consequences
- One write path, always embedded with the configured model.
- Clients that called insert/update must re-ingest instead.

---

## ADR-0005: Model choices for a 16 GB GPU

### Status
Accepted

### Date
2026-10-05

### Context
The target machine is an RTX 5060 Ti with 16 GB VRAM. Generation, Step 2 facts and (now) evaluation all
run on one Ollama server, and embeddings/reranker run on CPU. An OOM mid-run loses a long generation.

### Decision
- Generator and Step 2 facts: `qwen3.5:9b` (~6.6 GB).
- Judge: `qwen3.5:4b` (~3.3 GB) via `Ollama_evaluation_model`, a different, smaller model than the writer.
- vLLM (optional backend only): `Qwen/Qwen3.5-9B`, the Hugging Face counterpart of the Ollama tag.
- `HF_grounded_facts_model` is left at `Qwen/Qwen3-8B`; it is only read when `Grounded_facts_provider: hf`.

### Alternatives Considered
- `qwen3.5:27b` / `qwen3:30b` / `qwen3:32b`: 17-20 GB weights, do not fit.
- `qwen3:14b` (9.3 GB): fits alone, but an older generation than `qwen3.5:9b` and leaves little room for the judge.
- Judge = generator: simplest, but a model tends to score its own prose generously.

### Consequences
- Both models together plus KV cache use roughly 11-12 GB; lower `Model_max_prompt_tokens` first if OOM appears.
- Optional Ollama server env vars `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` shrink the cache further.
