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
- Three-layer generation (5W1H -> summary flow -> story expansion) for stronger structure control
- Prompt-split generation design in `prompts.yaml` (single-pass + layer1/layer2/layer3 templates)
- Mode-based generation controls (`FAST` / `THINKING`) with separate token/sampling settings
- Prompt-budget-aware context truncation before generation
- Repetition controls (repetition/frequency/presence penalties + no-repeat ngram)
- Advanced decoding via `Generative_AI/penalty_processors.py` to reduce repetition loops while excluding common stopwords from penalty application
- Post-generation cleanup safeguards to trim malformed tails and degeneration artifacts
- Reliability work for real failures (API rate limits, GPU OOM, malformed outputs)
- Format-aware Archive download handling (`pdf` / `epub`) with cleaner file selection
- Duplicate-download protection by identifier to avoid redundant fetches
- API + orchestrator refactor for cleaner endpoint responsibilities

## New Feature: Three-Layer Story Generation

The latest generation upgrade adds a three-layer flow designed for longer, cleaner stories under token limits:

1. **Layer 1 (5W1H extraction):** distills retrieved context into concrete `Who/What/When/Where/Why/How`.
2. **Layer 2 (story summary flow):** converts the 5W1H into a connected narrative plan.
3. **Layer 3 (story expansion):** expands that plan into the final full story.

If any intermediate layer returns empty output, generation falls back to single-pass mode.

Enable/configure in `setup.yaml`:

- `Three_layer_generation: true`
- `Story_generation_n_results` (retrieved context chunks; default 3)
- `Generation_mode_fast` / `Generation_mode_thinking`
- `Generation_fast_temperature` / `Generation_thinking_temperature`
- `Single_pass_fast_max_tokens` / `Single_pass_thinking_max_tokens`
- `Layer1_max_tokens`
- `Layer2_max_tokens`
- `Layer3_fast_max_tokens` / `Layer3_thinking_max_tokens`
- `Min_generation_ratio` (applies to layers 2 and 3)
- `Repetition_penalty`, `No_repeat_ngram_size`
- `Frequency_penalty`, `Presence_penalty`
- `Model_max_prompt_tokens` (prompt budget ceiling)

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

- `API/` - FastAPI routes and endpoint contracts
- `Orchestrator/` - Pipeline orchestration and step control
- `Book_search/` - Search/download + extraction entry points
- `Data/` - Metadata and merge logic
- `Vector_Store/` - Chroma ingestion/query utilities
- `Generative_AI/` - Context building and story generation
- `Evaluation/` - Automated scoring and feedback

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
6. Generate story (single-pass or three-layer: 5W1H -> summary -> full story)
7. Evaluate generated story/summary

## Who This Is For

- Recruiters and hiring managers reviewing applied AI/data engineering projects
- Engineers who want a concrete reference for retrieval + generation + evaluation pipelines

## Security Notes

- Do not commit `setup.yaml` (contains local secrets)
- Use `setup.example.yaml` as the public-safe template

## License

MIT
