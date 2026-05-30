# Project Journey: StoryForge-RAG

I built `StoryForge-RAG` around one core question:

**Can I ship a complete AI pipeline that turns raw books into generated stories with measurable quality checks?**

This document walks through that journey from design to deployment decisions, including what failed, what changed, and why the final system looks the way it does.

I also used AI assistants during development for research, debugging support, and iteration speed.

I recently changed generation to a **grounded single-pass pipeline** to reduce fact loss and hallucination:

- **Step 1 — Retrieval:** retrieve the most relevant story context from Chroma.
- **Step 2 — Grounded extraction:** extract concrete `Who/What/When/Where/Why/How` facts from that context.
- **Step 3 — Single-pass generation:** write the final story once from the grounded facts, avoiding a lossy summary-and-expansion chain.

I also moved generation behavior into clearer prompt contracts in `prompts.yaml` (single-pass + grounded extraction templates), added explicit generation profiles (`FAST`, `THINKING`, `SHORT`) with separate sampling/token settings, optional extraction temperature overrides, repetition-control processors (`repetition_penalty`, `no_repeat_ngram`, frequency/presence penalties), and kept post-generation cleanup guards in `Generative_AI/generative_ai.py` to reduce malformed endings and degeneration artifacts. The direction is promising, but results are still not as consistent as I expected.

I introduced `Generative_AI/penalty_processors.py` as an advanced decoding control layer to reduce repetitive word loops during generation. It applies frequency/presence penalties while excluding common stopwords from those penalties, so the model is discouraged from repeating low-value content words without over-penalizing normal function words needed for grammatical sentences.

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
- `Generative_AI/` handles context building, structure choice, and generation.
- `Evaluation/` handles rubric-based scoring (Gemini-based evaluation; kept enabled, but service availability was a real constraint in practice).
- `Orchestrator/` + `API/` handle pipeline control and service endpoints (`/orchestration`, `/create-eval`, etc.).

This separation let me iterate on one stage at a time without destabilizing the full pipeline.

## 3) What Failed Early

### Data cleanliness

Raw PDF/EPUB text often contained broken formatting and line noise.  
That directly reduced retrieval precision and generation quality.

I later added a retrieval-aware generation mode to separate **single** and **series** story behavior in practice:

- if generation mode is `single`, retrieval only uses records tagged as single in the vector store,
- if generation mode is `series`, retrieval only uses records tagged as series in the vector store.

Because older indexed records did not consistently store this field, I removed the previous vector-store data and re-ingested after validation. I also ran a double-check pass to ensure the saved metadata is now consistent, so retrieval filtering behaves correctly for both modes.

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

With the model prompt budget capped, single-pass generation often left limited room for output after the system prompt, reference context, and query consumed most of the window. Generated stories often felt truncated or rushed because of this.

I now handle this with a **grounded single-pass** approach that keeps the compression step but removes the lossy rewrite chain:

- **Retrieval:** pull the most relevant context from Chroma.
- **Grounded extraction:** compress retrieved context into concrete 5W1H facts.
- **Single-pass generation:** write the final story once from those facts.

Approximate budgets evolved with the pipeline; authoritative numbers live in `setup.example.yaml` (`Layer1_max_tokens`, `Single_pass_*`, etc.). Treat older fixed tables as illustrative only.

Why 5W1H helps in this pipeline:

- **Who:** locks in named characters, traits, and motivations early.
- **What:** defines the central conflict so later layers stay on-task.
- **When:** anchors timeline and pacing, reducing temporal inconsistencies.
- **Where:** fixes setting continuity, which reduces scene drift.
- **Why:** anchors character decisions, making actions feel causally consistent.
- **How:** captures conflict progression and resolution, so the final pass writes from grounded facts instead of improvising from raw retrieval.

This gives the model a cleaner path from context to final prose without forcing facts through summary compression and later expansion. The legacy `Three_layer_generation` key in `setup.yaml` now toggles this grounded pipeline, with automatic fallback to retrieval single-pass if extraction returns empty output.

### Query–retrieval alignment strategy (fail closed)

One of the biggest regressions I hit was **query–retrieval misalignment**: the vector search sometimes returned a concatenation of unrelated tales. Without guardrails, grounded extraction could produce a plausible 5W1H from the wrong story, and final generation would faithfully elaborate the wrong plot.

To make this failure mode explicit and controllable, I added two protective mechanisms:

- **Alignment scoring + fail-closed**: if the retrieved context has **zero keyword overlap** with the query, Layer 1 returns a structured **"insufficient context"** 5W1H (Unknown fields) instead of guessing.
- **Best-match context filtering**: when retrieval returns multiple story blocks (each labeled like `[Title by Author]`), the pipeline scores each block by query keyword overlap and filters/sorts to keep the best match (dropping zero-overlap blocks when any relevant block exists).

This doesn’t “fix retrieval” in the embedding sense, but it prevents the pipeline from confidently generating the wrong story when retrieval is clearly off.

### Summary throughput strategy

Single-file summary requests hit rate limits quickly.  
I switched to batched summarization (3 files/request) and remapped summaries back to source records.

### Iteration: structure, facts, and repetition

During tuning I hit predictable failure modes:

- **Invented scenes:** Earlier summary and expansion stages could ignore facts and improvise. The current fix path is to pass grounded extraction directly into final generation.
- **Missing HOW in extraction:** If the model skips plot-bearing elements, the final pass has little to work with. Mitigations: prompts and sampling that stress complete 5W1H, and treating empty extraction as a signal to fall back.
- **Repeated sections:** The older sectioned pipeline could repeat the same beat across generated sections. Single-pass generation avoids section stitching and reduces that failure mode.
- **Cleanup side effects:** Aggressive string fixes can create new artifacts; cleanup logic has to be tested against real outputs and adjusted when it does more harm than good.

**Cut-offs** at the end of generation remain a practical constraint: tight max-token settings or stop conditions can still truncate prose. Tuning `Single_pass_*` output budgets is an ongoing balance between length, coherence, and runtime.

For reproducible debugging without running the full API, I added **`debug_layers/`** (see `debug_layers/README.md`): run Layer 1, then 2, then 3 from the CLI with pinned queries and optional pinned flow structure names.

## 5) Reliability Upgrades

- Introduced explicit orchestrator steps for clearer pipeline control.
- Standardized merged output format for predictable ingestion.
- Added generation profiles (`FAST`, `THINKING`, `SHORT`) for controlled behavior.
- Replaced the older three-layer generation path with retrieval, grounded extraction, and single-pass generation.
- Added repetition-control logits processors (frequency/presence penalties with stopword-aware filtering).
- Added `penalty_processors.py` to reduce repeated wording and exclude selected common words from penalty application.
- Added prompt-aware context truncation against model prompt limits (to reduce hard truncation side-effects).
- Added output cleanup/truncation safeguards to reduce malformed text.
- Added retrieval guardrails: query–context alignment scoring, fail-closed "insufficient context" signal, and best-match story filtering when retrieval returns multiple tales.
- Removed the normal summary-and-expansion generation path to reduce fact loss and hallucination risk.
- Added retry logic and fallback model for transient evaluation API failures.
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
- For **summary generation**, the pipeline uses **Hugging Face Inference (summarization)** as the default (`Data/hf_summarize_and_merge.py`) and is tuned for consistency.
- **Gemini availability issue (real-world constraint):** Evaluation still uses Gemini (primary + fallback + retry), but over several days it frequently returned **“service unavailable” (503 / high demand)**. For now, I’m prioritizing iteration on the **generation path and output quality**, while keeping Gemini evaluation in place for when the service is stable (or until I replace evaluation later).
- This project was developed under tight budget limits, so model and infrastructure choices prioritized free-tier or low-cost options while keeping practical performance.

### Summary formatting cleanup (post-processing)
Even when summaries are successfully generated, real outputs can still contain small PDF/OCR-style artifacts (spacing around apostrophes/quotes, stray symbols, spaces before punctuation, etc.).

To make merged data more consistent for vector search and downstream generation, I added a post-processing pass that normalizes `metadatas.Summary` across `Data_Merged/*.json` using the same text cleanup principles documented in `STORY_FORMAT_GUIDE.md`:

- `Data/fix_merged_summaries_format.py` (batch fixer)
- Integrated into Orchestrator step 3 (`3_merge_check_summary`) so it runs **after** HF summaries are created.

## 8) Current status (incomplete outputs and RAG scale)

**Honest checkpoint:** with the pipeline as it runs today, outputs are still often **not complete**—stories can feel truncated, under-detailed, or uneven across sections. I am working through fixes rather than treating this as “done.”

**Hypothesis I am testing:** generation is only as good as the **context RAG provides**. Right now the vector store only has **about 47 indexed records** from the material I have ingested. That is a narrow retrieval base for varied queries and long multi-layer expansion: neighbors may be weakly related, repetitive, or missing plot-relevant detail, which shows up as thin or invented downstream behavior no matter how much I tune decoding.

**What I am doing next:** grow and diversify ingestion (more sources, cleaner merges), revisit **chunking and metadata** so retrieved passages carry enough story signal, and keep separating “model/token limits” from “empty or wrong context” using `debug_layers/` and retrieval inspection—so improvements are evidence-based, not guesswork.

## 9) API and Orchestrator Lessons

I struggled with API design in early versions. I initially grouped too many endpoints under a single route style, which created duplicate logic and made maintenance difficult.

I then refactored the API in stages:

- first split into a smaller set of route groups,
- then separated responsibilities more clearly,
- and finally organized around distinct endpoint sections so each part of the pipeline is easier to reason about.

The current structure uses dedicated route groups (`/orchestration` for pipeline steps with optional `extract_style` on generation, `/create-eval` for story/summary generation and evaluation) and cleaner separation of concerns, which reduced coupling and improved endpoint reliability.

I also introduced async handling in the API layer (with thread offloading where needed) to avoid unnecessary blocking during heavier operations.

### Why the orchestrator matters, even in a small project

At the beginning, I assumed an orchestrator was only necessary for large production systems. During development, I learned that even a smaller AI pipeline benefits from a clear orchestration layer:

- it centralizes step order and response contracts,
- it reduces breakage between API handlers and backend functions,
- and it makes pipeline evolution safer when steps change.

I redesigned the pipeline flow multiple times before settling on the current shape. The final version balances automation and manual checks so users can validate data quality between key steps.
