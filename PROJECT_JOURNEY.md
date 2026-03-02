# Project Journey: StoryForge-RAG

I built `StoryForge-RAG` around one core question:

**Can I ship a complete AI pipeline that turns raw books into generated stories with measurable quality checks?**

This document walks through that journey from design to deployment decisions, including what failed, what changed, and why the final system looks the way it does.

I also used AI assistants during development for research, debugging support, and iteration speed.

I recently shipped a **two-pass generation feature** to improve long-form output quality under strict token limits:

- **Pass 1:** generate a compact outline from retrieved story context.
- **Pass 2:** generate the full story from that outline, freeing more generation budget for complete narrative arcs.

I also moved generation behavior into clearer prompt contracts in `prompts.yaml` (single-pass + outline + story-from-outline templates), added explicit generation profiles (`FAST`, `THINKING`) with separate sampling/token settings, and added post-generation cleanup guards in `Generative_AI/generative_ai.py` to reduce malformed endings and degeneration artifacts. The direction is promising, but results are still not as consistent as I expected.

This update made outputs longer and more complete, and improved stability during repeated runs, but there is still room to tune consistency further.

## Why I Shared This Publicly

I intentionally shared this project with both strengths and known constraints instead of presenting a "perfect" demo.

This project is not 100% AI-generated code; I used AI as an assistant while making implementation decisions myself. I want recruiters to see both my current capabilities and the technical areas I am actively improving.

I still want to improve many parts of this system, and I am looking for a job environment where I can keep learning and practicing at a higher level.

This work reflects the engineering foundation I developed through hands-on AI/data projects, including experience from my time at Pixalate, Inc.

## What This Project Demonstrates

This repository demonstrates practical AI/data engineering capabilities:

- end-to-end system design across ingestion, retrieval, generation, and evaluation,
- API-first architecture for repeatable workflows,
- real-world failure handling (rate limits, OOM, unstable outputs),
- and iterative, evidence-based model/pipeline decisions.

I am actively using this project in my AI/data engineering job search to show how I design and ship full systems, not only isolated experiments.

## 1) Product Vision

The target workflow was:

- ingest long-form book/story content,
- transform it into structured, retrievable knowledge,
- generate new stories from semantic context,
- and evaluate output quality in a repeatable format.

I approached this as an engineering problem first, and a prompting problem second.

## 2) Architecture I Chose

To keep the system maintainable, I separated responsibilities by module:

- `Book_search/` handles source discovery, download, and extraction entry points.
- `Data/` handles metadata creation and merge logic.
- `Vector_Store/` handles Chroma ingestion and retrieval.
- `Generative_AI/` handles context building and generation.
- `Evaluation/` handles rubric-based scoring.
- `Orchestrator/` + `API/` handle pipeline control and service endpoints.

This separation let me iterate on one stage at a time without destabilizing the full pipeline.

## 3) What Failed Early

### Data cleanliness

Raw PDF/EPUB text often contained broken formatting and line noise.  
That directly reduced retrieval precision and generation quality.

### Data alignment

Stories, metadata JSON files, and merged records had to stay aligned by file stem.  
Even small mismatches caused missing context during retrieval/ingestion.

### Generation reliability

Early outputs sometimes collapsed into repetition loops or incoherent blocks.  
That highlighted the gap between "works once" and "works consistently."

### Codebase clarity

Fast iteration left duplicate example paths in the repo.  
Removing them improved maintainability and reduced ambiguity.

## 4) Key Decisions and Trade-offs

### Source strategy

I initially planned to rely more on Google Books, but for practical access to older/public-domain material, I prioritized `archive.org` as the primary source.

As the pipeline evolved, I updated download handling to be format-aware (`pdf` / `epub`) and improved source selection to avoid redundant downloads of already-fetched Archive identifiers.

### Embedding strategy

I first considered a T5/Gemma + LangChain wrapping approach for vector encoding, and also explored a BART-based wrapping option. After evaluation, I decided this added unnecessary complexity for my use case, so I simplified the stack.
It was unstable in this pipeline, so I moved to `sentence-transformers` for more consistent embedding behavior and cleaner Chroma integration.

### Generation model strategy

I evaluated multiple model sizes:

- `SmolLM3` had output-length limits for long-form story targets.
- `Qwen/Qwen2.5-14B-Instruct` improved some outputs but raised latency significantly in my setup.
- `Qwen/Qwen2.5-7B-Instruct` gave the best quality/speed trade-off for local iteration.

### Memory strategy

On RTX 5060 Ti, I encountered OOM during generation.  
I stabilized local inference by enabling 4-bit loading with bitsandbytes.

### Generation token budget strategy

With the model capped at 8192 prompt tokens, only 768–1536 tokens remained for generation in single-pass mode after the system prompt, reference context, and query consumed most of the budget. Generated stories often felt truncated or rushed because of this.

I solved this with a **two-pass generation** approach:

- **Pass 1 (Outline):** The model receives the full reference context (default: 3 story chunks from the vector store) and produces a short structural outline (default: 300 tokens) containing setting, characters, and plot beats.
- **Pass 2 (Full Story):** The bulky reference material is replaced by the compact outline, freeing up the prompt window. The model now generates the full story with a much larger pass-2 budget (default: up to 2048 in `FAST`, up to 3072 in `THINKING`) instead of 768–1536.

This effectively compresses the reference knowledge into the model's own creative plan, then gives it room to write. The feature is toggled via `Two_pass_generation` in `setup.yaml`, with automatic fallback to single-pass if the outline step produces empty output.

### Summary throughput strategy

Single-file summary requests hit rate limits quickly.  
I switched to batched summarization (3 files/request) and remapped summaries back to source records.

## 5) Reliability Upgrades

- Introduced explicit orchestrator steps for clearer pipeline control.
- Standardized merged output format for predictable ingestion.
- Added generation profiles (`FAST`, `THINKING`) for controlled behavior.
- Added two-pass generation (outline → expand) to maximize token budget for story output.
- Added prompt-aware context truncation against model prompt limits (to reduce hard truncation side-effects).
- Added output cleanup/truncation safeguards to reduce malformed text.
- Added retry logic for transient evaluation API failures.
- Added duplicate-download guards to skip already-downloaded Archive results.
- Improved Archive file matching logic to select requested formats more reliably.
- Added public-safe repo setup:
  - `setup.example.yaml`
  - `.gitignore` for secrets and generated artifacts
  - removed duplicate example code paths

## 6) Scope Boundaries

- Manual metadata curation is intentionally retained for story-level control.
- The workflow combines automated stages with controlled manual quality steps.
- Evaluation focuses on practical rubric scoring via API-driven checks.

## 7) Practical Constraints

- I iterated through multiple metadata structures before selecting the current one, which proved most stable with retrieval + summarization.
- For summary generation, Gemini 2.5 produced the most consistent behavior in this pipeline.
- This project was developed under tight budget limits, so model and infrastructure choices prioritized free-tier or low-cost options while keeping practical performance.

## 8) API and Orchestrator Lessons

I struggled with API design in early versions. I initially grouped too many endpoints under a single route style, which created duplicate logic and made maintenance difficult.

I then refactored the API in stages:

- first split into a smaller set of route groups,
- then separated responsibilities more clearly,
- and finally organized around distinct endpoint sections so each part of the pipeline is easier to reason about.

The current structure uses dedicated route groups and cleaner separation of concerns, which reduced coupling and improved endpoint reliability.

I also introduced async handling in the API layer (with thread offloading where needed) to avoid unnecessary blocking during heavier operations.

### Why the orchestrator matters, even in a small project

At the beginning, I assumed an orchestrator was only necessary for large production systems. During development, I learned that even a smaller AI pipeline benefits from a clear orchestration layer:

- it centralizes step order and response contracts,
- it reduces breakage between API handlers and backend functions,
- and it makes pipeline evolution safer when steps change.

I redesigned the pipeline flow multiple times before settling on the current shape. The final version balances automation and manual checks so users can validate data quality between key steps.
