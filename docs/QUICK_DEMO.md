# Quick Demo

This guide gives reviewers a fast way to understand and verify StoryForge-RAG without needing a GPU, Gemini key, Hugging Face key, Google Books key, or local Chroma database.

The full pipeline still needs local configuration and model/API access. The quick path below focuses on the parts that can be checked safely on a fresh clone: project structure, data helpers, parsing logic, and documentation.

## 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## 2. Run The Lightweight Tests

```bash
python -m pytest
```

These tests use temporary folders and do not touch local runtime data such as `data/stories/`, `data/chroma_db/`, `data/outputs/`, or a private `setup.yaml`.

Current test coverage focuses on:

- Config loading and `setup.example.yaml` fallback (`storyforge.config`)
- Story-length profile resolution (`tests/test_length_profile.py`)
- Prompt contracts (including `{length_guidance}` placeholders)
- Hugging Face-first evaluation provider selection and Gemini fallback
- Retrieval metric calculations (`storyforge.evaluation.retrieval_eval`)
- Story-json → ingest manifest workflow
- Agentic loop decisions, attribution gate, and API route contracts

Run the same lightweight suite locally with `python -m pytest` (see `tests/`). Prefer the project venv if you use one (`.\venv\Scripts\python.exe -m pytest`).

## 3. Review The Architecture

Start with:

- [`../README.md`](../README.md) for a short overview
- [`README.md`](./README.md) for the current architecture, length targets, and API examples
- [`PROJECT_JOURNEY.md`](./PROJECT_JOURNEY.md) for trade-offs, failures, and lessons learned
- [`UPGRADE_ROADMAP_5060Ti.md`](./UPGRADE_ROADMAP_5060Ti.md) for hardware / backend upgrade notes
- [`PROJECT_UPDATE_ROADMAP.md`](./PROJECT_UPDATE_ROADMAP.md) for a historical hiring-readiness snapshot (not current architecture)
- [`PRODUCTION_NOTES.md`](./PRODUCTION_NOTES.md) for production boundaries and next steps

The main system shape is:

```text
Stories (.txt) → story_json → ingest manifest
        |
        v
Chroma Vector Store (BGE + hybrid BM25 + rerank)
        |
        v
Step 1 Retrieve → Step 2 Grounded facts (HF) → Step 3 Story (Ollama)
        |
        v
Length guard / agentic evaluate → refine / re-retrieve / accept
```

Story length is one request field (`length`: preset / `"12min"` / word count). It drives prompt guidance, token budget, and accept gates together. See [`README.md`](./README.md#story-length).

## Demo Mode Status

The current reviewer demo is a lightweight validation path, not a full mocked API mode. It proves the deterministic pieces of the project with tests and documents how to run the real API when local config is available.

What is mocked or lightweight today:

- Evaluation provider selection is tested with mocked Hugging Face/Gemini behavior.
- Retrieval metrics are tested with mocked retrieval results.
- Length-profile resolution is pure arithmetic / config parsing (no GPU).
- Data merge and metadata checks run against temporary folders.

What still requires real local setup:

- End-to-end `/create-eval/story_generate`
- Chroma-backed retrieval
- Ollama (or vLLM / Transformers) generation
- Live HF/Gemini calls

## 4. Optional Full API Run

The API path requires local setup:

1. Copy `setup.example.yaml` to `setup.yaml` for real local runs.
2. Set `BASE_PATH` to this project folder.
3. Start Ollama: `docker compose up -d` then `docker exec -it ollama ollama pull qwen3.5:9b`.
4. Add API keys only for the features you want to run (HF for Step 2 / eval).
5. Start the API:

```bash
python main.py
```

Open:

```text
http://localhost:8000/docs
```

Useful endpoints:

| Path | Purpose |
|------|---------|
| `POST /create-eval/story_generate` | Generate with `mode` + optional `length` |
| `POST /orchestration/run_step` | Single pipeline step (incl. generate) |
| `POST /orchestration/generate_stream` | SSE token stream |
| `/vector_store/*` | Inspect / query Chroma |
| `/book-docs` | Book search / download |
| `/data-docs` | Data merge / summarization |

Example generate body:

```json
{
  "query": "A scholar discovers something in an old house that he shouldn't have",
  "generation_type": "full_story",
  "save": true,
  "mode": "fast",
  "length": "long",
  "story_type": "mix"
}
```

After editing `setup.yaml` or `prompts.yaml`, restart the API — config and prompts are cached.

## What Requires External Resources

- Full story generation requires Ollama (default) or another configured provider, plus enough GPU memory.
- Book search/download can use Google Books and Archive.org access.
- Summary creation uses Hugging Face Inference API.
- Story/summary evaluation tries Hugging Face first, then Gemini fallback if configured.
- RAG generation expects a populated Chroma database.
- Real retrieval evaluation expects a populated Chroma database, but the metric logic is covered by lightweight tests.

To run retrieval evaluation after ingesting data:

```bash
py scripts/retrieval_eval.py --cases tests/fixtures/retrieval_eval_cases.example.json --k 3
```

HF-first evaluation is configured with:

- `Evaluation_provider_priority`
- `HF_evaluation_model`
- `HF_evaluation_max_new_tokens`
- `HF_evaluation_temperature`
- `facehugging_api` or an HF token environment variable

## What This Demo Proves

The lightweight demo proves the repository has runnable tests and that core data/parsing helpers behave predictably without external services.

It also verifies that fresh-clone imports can fall back to `setup.example.yaml` through the shared config loader instead of requiring a private local `setup.yaml` immediately.

The full project demonstrates the larger applied AI system: ingestion, retrieval, grounded generation with a selectable length target, orchestration, and evaluation.
