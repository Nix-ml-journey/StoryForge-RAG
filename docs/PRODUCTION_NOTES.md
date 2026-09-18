# Production Notes

StoryForge-RAG is currently a portfolio and local-development project. It demonstrates the shape of a production AI/data pipeline, but it is not yet designed for production traffic.

## Current Boundaries

- **Local generation dependency:** Step 3 story writing defaults to **Ollama in Docker** (`qwen3.5:9b`). vLLM and Transformers are also supported via `Generation_provider`. Deployment still needs explicit GPU / VRAM planning.
- **External APIs:** Step 2 grounded facts and evaluation typically use Hugging Face Inference API (with Gemini fallback for eval, or `Evaluation_mode: "local"` to avoid the API entirely). Book search can use Google Books / Archive.org. These services can rate-limit, fail transiently, or change availability. Extraction falls back to local generation (Ollama / vLLM / Transformers) when HF returns 502 / gateway errors.
- **Length reliability:** One `length` target keeps prompt, token budget, and accept gates aligned. Thinking mode can still return an empty draft (budget spent in `<think>` blocks); `generate_from_facts()` retries once with fast-mode sampling and raises a clear error if that also comes back empty. The length-guard refine pass falls back to the pre-refine draft; the agentic loop returns the best draft so far (or a clean empty result on first-iteration failure). Prefer `mode: "fast"` + `length: "13min"` for long targets if you still see empty results.
- **Small corpus risk:** Retrieval quality is limited by the size and cleanliness of the indexed story corpus. Long targets need enough chunks / facts (`Story_generation_n_results`, `HF_grounded_facts_max_new_tokens`) or prose pads with repetition.
- **Manual metadata checks:** Some metadata curation remains intentionally manual. That improves control, but it is not yet a fully automated data quality workflow. See [`DATA_PREP.md`](./DATA_PREP.md).
- **File-based runtime outputs:** Generated stories, summaries, evaluations, downloaded books, and Chroma data are local folders under `data/`. A production system should move these to managed storage.
- **Streaming vs non-streaming:** `/orchestration/generate_stream` shares the length prompt but cannot refine mid-stream, and it does not run the attribution gate. Length enforcement and attribution after generation are non-streaming / agentic-loop concerns.

## Production Path

To harden this project for production, I would add:

- **Job queue:** Run ingestion, summarization, generation, and evaluation as background jobs with retries and status tracking.
- **Persistent database:** Store metadata, job status, evaluation reports, and generated outputs in a database instead of only local files.
- **Object storage:** Store raw books, extracted text, generated stories, and evaluation artifacts in object storage.
- **Observability:** Add structured logs, request IDs, stage timing, model/version metadata, length-target metrics (requested vs actual word count), and error dashboards.
- **Evaluation dashboard:** Track retrieval accuracy, generation quality scores, failure rates, and before/after tuning changes.
- **Deployment config:** Separate local, test, and production settings with environment-specific secrets management.
- **CI/CD:** Keep the lightweight tests fast, then add optional heavier integration checks behind manual workflows.

## What Is Already In Place

- FastAPI route structure for repeatable workflows (`/orchestration`, `/create-eval`, vector store, streaming).
- Shared config loading with `setup.example.yaml` fallback.
- Unified story-length target (`Story_length_*` + request `length` field).
- Hugging Face-first evaluation with Gemini fallback, or fully local (`Evaluation_mode: "local"`) to remove the API round-trip from the agentic loop entirely.
- Hybrid BM25 + dense retrieval, reranker, and agentic refine / re-retrieve loop.
- vLLM as an alternative generation backend for higher-throughput / concurrent use (`Generation_provider: "vllm"`).
- Empty-draft recovery: a failed thinking-mode draft retries once with fast sampling and raises a clear error instead of returning nothing; the length-guard refine pass falls back to the pre-refine draft; the agentic loop catches the same error per iteration and returns the best draft so far (or a clean empty result on first-iteration failure).
- Lightweight tests for config, length profile, prompts, agentic decisions, evaluation provider selection (API + local), empty-draft recovery (3-step + agentic), and retrieval metrics.
- Post-extract data-prep checklist: [`DATA_PREP.md`](./DATA_PREP.md).
- Retrieval evaluation script for top-k accuracy and expected fact coverage.

## What I Would Not Do Yet

- I would not run full local model generation inside default CI.
- I would not commit private datasets, generated outputs, Chroma databases, or local API keys.
- I would not add distributed infrastructure before retrieval quality and long-form accept rates are more mature.
