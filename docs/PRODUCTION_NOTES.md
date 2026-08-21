# Production Notes

StoryForge-RAG is currently a portfolio and local-development project. It demonstrates the shape of a production AI/data pipeline, but it is not yet designed for production traffic.

## Current Boundaries

- **Local generation dependency:** Step 3 story writing defaults to **Ollama in Docker** (`qwen3.5:9b`). vLLM and Transformers are also supported via `Generation_provider`. Deployment still needs explicit GPU / VRAM planning.
- **External APIs:** Step 2 grounded facts and evaluation typically use Hugging Face Inference API (with Gemini fallback for eval). Book search can use Google Books / Archive.org. These services can rate-limit, fail transiently, or change availability. Extraction falls back to local Ollama when HF returns 502 / gateway errors.
- **Length reliability:** One `length` target now keeps prompt, token budget, and accept gates aligned. Thinking mode can still return empty drafts (budget spent in `<think>` blocks); the length guard retries once. Prefer `mode: "fast"` + `length: "13min"` when long targets come back empty.
- **Small corpus risk:** Retrieval quality is limited by the size and cleanliness of the indexed story corpus. Long targets need enough chunks / facts (`Story_generation_n_results`, `HF_grounded_facts_max_new_tokens`) or prose pads with repetition.
- **Manual metadata checks:** Some metadata curation remains intentionally manual. That improves control, but it is not yet a fully automated data quality workflow.
- **File-based runtime outputs:** Generated stories, summaries, evaluations, downloaded books, and Chroma data are local folders under `data/`. A production system should move these to managed storage.
- **Streaming vs non-streaming:** `/orchestration/generate_stream` shares the length prompt but cannot refine mid-stream. Length enforcement after generation is a non-streaming / agentic-loop concern.

## Production Path

To harden this project for production, I would add:

- **Job queue:** Run ingestion, summarization, generation, and evaluation as background jobs with retries and status tracking.
- **Persistent database:** Store metadata, job status, evaluation reports, and generated outputs in a database instead of only local files.
- **Object storage:** Store raw books, extracted text, generated stories, and evaluation artifacts in object storage.
- **Observability:** Add structured logs, request IDs, stage timing, model/version metadata, length-target metrics (requested vs actual word count), and error dashboards.
- **Evaluation dashboard:** Track retrieval accuracy, generation quality scores, failure rates, and before/after tuning changes.
- **Deployment config:** Separate local, test, and production settings with environment-specific secrets management.
- **CI/CD:** Keep the lightweight tests fast, then add optional heavier integration checks behind manual workflows.
- **Empty-draft hardening:** Re-validate after length-guard refine; return clear `success: false` when the target is still unmet.

## What Is Already In Place

- FastAPI route structure for repeatable workflows (`/orchestration`, `/create-eval`, vector store, streaming).
- Shared config loading with `setup.example.yaml` fallback.
- Unified story-length target (`Story_length_*` + request `length` field).
- Hugging Face-first evaluation with Gemini fallback.
- Hybrid BM25 + dense retrieval, reranker, and agentic refine / re-retrieve loop.
- Lightweight CI tests for config, length profile, prompts, agentic decisions, evaluation provider selection, and retrieval metrics.
- Retrieval evaluation script for top-k accuracy and expected fact coverage.

## What I Would Not Do Yet

- I would not run full local model generation inside default CI.
- I would not commit private datasets, generated outputs, Chroma databases, or local API keys.
- I would not add distributed infrastructure before retrieval quality and long-form accept rates are more mature.
