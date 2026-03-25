# Project Journey: StoryForge-RAG

I built `StoryForge-RAG` around one core question:

**Can I ship a complete AI pipeline that turns raw books into generated stories with measurable quality checks?**

This document walks through that journey from design to deployment decisions, including what failed, what changed, and why the final system looks the way it does.

I also used AI assistants during development for research, debugging support, and iteration speed.

I recently shipped a **three-layer generation feature** to improve long-form output quality under strict token limits:

- **Layer 1 (5W1H):** extract concrete story elements from retrieved context.
- **Layer 2 (Sectioned summary):** map those elements into a **chosen narrative structure** (from `flow_structure.yaml`)—usually **five sections generated one-by-one** so each block stays tied to facts.
- **Layer 3 (Story expansion):** expand the summary into full prose, typically **one generation pass per section** for length and coherence.

I also moved generation behavior into clearer prompt contracts in `prompts.yaml` (single-pass + layer1/layer2/layer3 templates), added explicit generation profiles (`FAST`, `THINKING`, `SHORT`) with separate sampling/token settings, optional per-layer temperature overrides, repetition-control processors (`repetition_penalty`, `no_repeat_ngram`, frequency/presence penalties), and kept post-generation cleanup guards in `Generative_AI/generative_ai.py` to reduce malformed endings and degeneration artifacts. The direction is promising, but results are still not as consistent as I expected.

`Generative_AI/choser.py` picks the narrative **structure** after Layer 1 and before Layer 2 so Layer 2 only maps 5W1H into a fixed section order instead of inventing structure and content at once. FAST mode restricts choices to linear-friendly flows; THINKING can use richer non-linear patterns.

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
- `Evaluation/` handles rubric-based scoring (Gemini primary model with fallback after retries on transient errors).
- `Orchestrator/` + `API/` handle pipeline control and service endpoints (`/orchestration`, `/create-eval`, etc.).

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

With the model prompt budget capped, single-pass generation often left limited room for output after the system prompt, reference context, and query consumed most of the window. Generated stories often felt truncated or rushed because of this.

I solved this with a **three-layer generation** approach that **stages** work:

- **Layer 1 (5W1H extraction):** compress reference context into concrete elements.
- **Layer 2 (Sectioned summary):** map 5W1H into the chosen flow from `flow_structure.yaml`, with **per-section** generation when enabled so sections do not overwrite each other in one shot.
- **Layer 3 (Full story):** expand section-by-section with configurable min/max token budgets per section; **`SHORT`** mode halves Layer 3 budgets versus **`FAST`** for shorter stories.

Approximate per-stage budgets evolved with the pipeline; authoritative numbers live in `setup.example.yaml` (`Layer1_max_tokens`, `Layer2_*`, `Layer3_section_*`, etc.). Treat older fixed tables as illustrative only.

Why 5W1H helps in this pipeline:

- **Who:** locks in named characters, traits, and motivations early.
- **What:** defines the central conflict so later layers stay on-task.
- **When:** anchors timeline and pacing, reducing temporal inconsistencies.
- **Where:** fixes setting continuity, which reduces scene drift.
- **Why:** anchors character decisions, making actions feel causally consistent.
- **How:** pre-plans conflict progression and resolution (including a twist), so Layer 3 expands a structured plan instead of improvising from raw retrieval.

This effectively compresses reference knowledge in stages and gives the model a cleaner path from context to final prose. The feature is toggled via `Three_layer_generation` in `setup.yaml`, with automatic fallback to single-pass if intermediate layers return empty output.

### Summary throughput strategy

Single-file summary requests hit rate limits quickly.  
I switched to batched summarization (3 files/request) and remapped summaries back to source records.

### Iteration: structure, facts, and repetition

During tuning I hit predictable failure modes:

- **Invented scenes:** If Layer 2 saw the full 5W1H at once without clear mapping to each narrative section, a smaller model could ignore facts and improvise. The fix path was **section-scoped facts** (only the 5W1H fragments relevant to the current section) and explicit instructions to rewrite facts, not invent new scenes.
- **Missing HOW in Layer 1:** If the model skipped plot-bearing elements, downstream layers had little to expand. Mitigations: prompts and sampling that stress complete 5W1H, and treating empty or thin layers as signals to retry or fall back.
- **Repeated sections:** Generating all sections in one call without cross-section awareness could repeat the same beat. **Per-section Layer 2** and clearer structure choice (`choser.py` + `flow_structure.yaml`) address that by construction.
- **Cleanup side effects:** Aggressive string fixes can create new artifacts; cleanup logic has to be tested against real outputs and adjusted when it does more harm than good.

**Cut-offs** at the end of generation remain a practical constraint: tight max-token settings or stop conditions can still truncate prose. Tuning Layer 3 min/max tokens and `Min_generation_ratio` is an ongoing balance between length, coherence, and runtime.

For reproducible debugging without running the full API, I added **`debug_layers/`** (see `debug_layers/README.md`): run Layer 1, then 2, then 3 from the CLI with pinned queries and optional pinned flow structure names.

## 5) Reliability Upgrades

- Introduced explicit orchestrator steps for clearer pipeline control.
- Standardized merged output format for predictable ingestion.
- Added generation profiles (`FAST`, `THINKING`, `SHORT`) for controlled behavior.
- Added three-layer generation with structure selection, sectioned Layer 2, and multi-pass Layer 3.
- Added repetition-control logits processors (frequency/presence penalties with stopword-aware filtering).
- Added `penalty_processors.py` to reduce repeated wording and exclude selected common words from penalty application.
- Added prompt-aware context truncation against model prompt limits (to reduce hard truncation side-effects).
- Added output cleanup/truncation safeguards to reduce malformed text.
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
- For summary generation, Gemini 2.5 produced the most consistent behavior in this pipeline.
- This project was developed under tight budget limits, so model and infrastructure choices prioritized free-tier or low-cost options while keeping practical performance.

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
