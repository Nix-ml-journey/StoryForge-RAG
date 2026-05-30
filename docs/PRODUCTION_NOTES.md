# Production Notes

StoryForge-RAG is currently a portfolio and local-development project. It demonstrates the shape of a production AI/data pipeline, but it is not yet designed for production traffic.

## Current Boundaries

- **Local model dependency:** Full story generation depends on a local Hugging Face model and enough CPU/GPU memory. The project includes 4-bit loading options, but deployment still needs explicit hardware planning.
- **External APIs:** Book search, summary creation, and evaluation can depend on Google Books, Archive.org, Hugging Face, and Gemini. These services can rate-limit, fail transiently, or change availability.
- **Small corpus risk:** Retrieval quality is limited by the size and cleanliness of the indexed story corpus. The retrieval evaluation script now exists, but real results depend on populated Chroma data.
- **Manual metadata checks:** Some metadata curation remains intentionally manual. That improves control, but it is not yet a fully automated data quality workflow.
- **File-based runtime outputs:** Generated stories, summaries, evaluations, downloaded books, and Chroma data are local folders. A production system should move these to managed storage.

## Production Path

To harden this project for production, I would add:

- **Job queue:** Run ingestion, summarization, generation, and evaluation as background jobs with retries and status tracking.
- **Persistent database:** Store metadata, job status, evaluation reports, and generated outputs in a database instead of only local files.
- **Object storage:** Store raw books, extracted text, generated stories, and evaluation artifacts in object storage.
- **Observability:** Add structured logs, request IDs, stage timing, model/version metadata, and error dashboards.
- **Evaluation dashboard:** Track retrieval accuracy, generation quality scores, failure rates, and before/after tuning changes.
- **Deployment config:** Separate local, test, and production settings with environment-specific secrets management.
- **CI/CD:** Keep the lightweight tests fast, then add optional heavier integration checks behind manual workflows.

## What Is Already In Place

- FastAPI route structure for repeatable workflows.
- Shared config loading with `setup.example.yaml` fallback.
- Hugging Face-first evaluation with Gemini fallback.
- Lightweight CI tests for parsing, metadata, merge, evaluation provider selection, and retrieval metrics.
- Retrieval evaluation script for top-k accuracy and expected fact coverage.

## What I Would Not Do Yet

- I would not run full local model generation inside default CI.
- I would not commit private datasets, generated outputs, Chroma databases, or local API keys.
- I would not add distributed infrastructure before retrieval quality and evaluation metrics are more mature.
