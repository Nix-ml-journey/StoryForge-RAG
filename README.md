# StoryForge-RAG

StoryForge-RAG is an end-to-end AI pipeline for book ingestion, metadata enrichment, vector search (Chroma), story generation, and automated quality evaluation via FastAPI.

Built as a practical AI/data engineering project to demonstrate system design, model trade-offs, and real-world reliability handling (rate limits, OOM, unstable generations).

## Why It Matters

Most AI demos stop at generation. This project covers the full workflow:

- source intake and text extraction,
- data/metadata preparation,
- retrieval + generation,
- and automated evaluation.

## Project Journey

For the full development story, architecture decisions, trade-offs, and lessons learned, read:

- [`PROJECT_JOURNEY.md`](./PROJECT_JOURNEY.md)

## Highlights

- End-to-end pipeline from raw books to evaluated generated stories
- Retrieval-first architecture with Chroma + embeddings
- Model selection based on quality/speed constraints (`Qwen2.5-7B` as practical local trade-off)
- Three-layer generation (5W1H → sectioned summary flow → multi-pass story expansion) with optional narrative structures from `flow_structure.yaml`
- Prompt-split generation design in `prompts.yaml` (single-pass + layer1/layer2/layer3 templates)
- Generation modes: `FAST`, `THINKING`, and `SHORT` (~50% shorter Layer 3 budgets than FAST)
- Optional per-layer temperature/top-p overrides in `setup.yaml`
- Prompt-budget-aware context truncation before generation
- Repetition controls (repetition/frequency/presence penalties + no-repeat ngram)
- Advanced decoding via `Generative_AI/penalty_processors.py` to reduce repetition loops while excluding common stopwords from penalty application
- Post-generation cleanup safeguards to trim malformed tails and degeneration artifacts
- Reliability work for real failures (API rate limits, GPU OOM, malformed outputs)
- Format-aware Archive download handling (`pdf` / `epub`) with cleaner file selection
- Duplicate-download protection by identifier to avoid redundant fetches
- Evaluation with primary + fallback Gemini models and retry on transient API errors
- API groups: `/orchestration` (pipeline steps), `/create-eval` (generate/evaluate stories and summaries)
- `debug_layers/` scripts to run Layer 1–3 in isolation for debugging (see `debug_layers/README.md`)

## Three-Layer Story Generation

The generation path distills retrieved context, plans a sectioned narrative, then expands it into full prose:

1. **Layer 1 (5W1H extraction):** distills retrieved context into concrete `Who/What/When/Where/Why/How` (optional style/tone extraction via orchestration `extract_style`).
2. **Structure choice:** `Generative_AI/choser.py` picks a narrative flow from `flow_structure.yaml` (mode-filtered: FAST favors linear-friendly structures; THINKING can use non-linear flows).
3. **Layer 2 (sectioned summary):** maps 5W1H into the chosen structure—typically **five sections one-by-one** (`Layer2_per_section`) so each block stays grounded in facts.
4. **Layer 3 (story expansion):** **multi-pass prose** (one expansion per summary section by default) for length and coherence under token limits.

If any intermediate layer returns empty output, generation falls back to single-pass mode.

Enable/configure in `setup.yaml` (see `setup.example.yaml` for commented defaults):

- `Three_layer_generation`
- `Story_generation_n_results`
- `Generation_mode_fast` / `Generation_mode_thinking` / `Generation_mode_short`
- `Generation_*_temperature` / `Generation_*_top_p` and optional `Layer1_*`, `Layer2_*`, `Layer3_*` overrides
- Single-pass and layer token budgets, `Layer2_per_section`, `Layer2_section_max_tokens`, `Layer3_section_count`, `Layer3_section_min_tokens` / `Layer3_section_max_tokens` (and SHORT variants)
- `Min_generation_ratio`, penalties, `Model_max_prompt_tokens`

Related prompt templates live in `prompts.yaml` under `generation`:

- `full_story_system` / `full_story_user`
- `layer1_5w1h_system` / `layer1_5w1h_user`
- `layer2_summary_system` / `layer2_summary_user`
- `layer3_story_system` / `layer3_story_user`

## What It Does

- Fetches public-domain books (Google Books + Archive.org)
- Downloads selected source formats (`pdf` / `epub`) from Archive when available
- Extracts text from PDF/EPUB sources
- Builds metadata + merged records for retrieval
- Ingests records into Chroma vector store
- Generates stories from retrieved context (single-pass or three-layer generation)
- Evaluates generated outputs with rubric-based scoring

## Tech Stack

- Python, FastAPI, Pydantic
- ChromaDB, sentence-transformers
- Transformers (local generation model)
- Gemini API (summary/evaluation)

## Project Structure

- `API/` - FastAPI routes (`orchestration`, `create-eval`, data, etc.)
- `Orchestrator/` - Pipeline orchestration and step control
- `Book_search/` - Search/download + extraction entry points
- `Data/` - Metadata and merge logic
- `Vector_Store/` - Chroma ingestion/query utilities
- `Generative_AI/` - Context building, story generation, penalty processors, flow structure choice
- `Evaluation/` - Automated scoring and feedback
- `debug_layers/` - Optional layer-by-layer CLI debugging

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Create local config:
   - copy `setup.example.yaml` to `setup.yaml`
   - fill local API keys and paths in `setup.yaml`
3. Start API:
   - `python main.py`
4. Open docs:
   - `http://localhost:8000/docs`

## Main Flow

1. Search/download book files
2. Extract text from book files
3. Create metadata templates
4. Merge metadata + documents
5. Ingest into Chroma
6. Generate story (single-pass or three-layer: 5W1H → structure → sectioned summary → expanded story)
7. Evaluate generated story/summary

## Who This Is For

- Recruiters and hiring managers reviewing applied AI/data engineering projects
- Engineers who want a concrete reference for retrieval + generation + evaluation pipelines

## Current status

With the current setup, generated stories are still often **incomplete** (cut short or thin in places), and I am actively debugging and tuning that. My main working hypothesis is **retrieval coverage**: the Chroma index built from what I have ingested so far only exposes **about 47 records** to RAG, which is small for diverse long-form conditioning—so weak or repetitive context upstream may be part of the problem, alongside token limits and generation settings. I am expanding ingestion and revisiting chunking/metadata to improve what retrieval returns.

## Security Notes

- Do not commit `setup.yaml` (contains local secrets)
- Use `setup.example.yaml` as the public-safe template

## License

MIT
