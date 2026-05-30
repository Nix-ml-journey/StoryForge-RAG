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

These tests use temporary folders and do not touch local runtime data such as `Stories/`, `Metadata/`, `Data_Merged/`, `chroma_db/`, or generated outputs.

Current test coverage focuses on:

- Shared config loading and `setup.example.yaml` fallback in `storyforge_config.py`
- Hugging Face-first evaluation provider selection and Gemini fallback in `Evaluation/evaluation.py`
- Retrieval metric calculations in `Evaluation/retrieval_eval.py`
- Layer 2 section parsing in `Generative_AI/section_parse.py`
- Metadata summary validation in `Data/metadata.py`
- Story/metadata merge behavior in `Data/data_merge.py`

The same lightweight suite is also wired into `.github/workflows/tests.yml` for push, pull request, and manual GitHub Actions runs.

## 3. Review The Architecture

Start with:

- `README.md` for the project overview and main flow.
- `PROJECT_JOURNEY.md` for trade-offs, failures, and lessons learned.
- `PROJECT_UPDATE_ROADMAP.md` for the current job-readiness improvement plan.
- `PRODUCTION_NOTES.md` for production boundaries and next steps.

The main system shape is:

```text
Book Search / Extract
        |
        v
Metadata + Story Merge
        |
        v
Chroma Vector Store
        |
        v
Retrieval + Grounded Single-Pass Generation
        |
        v
LLM-Based Evaluation
```

## Demo Mode Status

The current reviewer demo is a lightweight validation path, not a full mocked API mode. It proves the deterministic pieces of the project with tests and documents how to run the real API when local config is available.

What is mocked or lightweight today:

- Evaluation provider selection is tested with mocked Hugging Face/Gemini behavior.
- Retrieval metrics are tested with mocked retrieval results.
- Data merge and metadata checks run against temporary folders.

What still requires real local setup:

- End-to-end `/create-eval/story_generate`
- Chroma-backed retrieval
- Local generation model inference
- Live HF/Gemini calls

## 4. Optional Full API Run

The API path requires local setup:

1. Copy `setup.example.yaml` to `setup.yaml` for real local runs.
2. Set `BASE_PATH` to this project folder.
3. Add API keys only for the features you want to run.
4. Start the API:

```bash
python main.py
```

Open:

```text
http://localhost:8000/docs
```

Useful docs pages:

- `http://localhost:8000/docs` - orchestration endpoints
- `http://localhost:8000/create-eval` - story generation/evaluation endpoints
- `http://localhost:8000/vector` - vector store inspection/query endpoints
- `http://localhost:8000/book-docs` - book search/download endpoints
- `http://localhost:8000/data-docs` - data merge/summarization endpoints

## What Requires External Resources

- Full local story generation requires the configured Hugging Face model and enough CPU/GPU memory.
- Book search/download can use Google Books and Archive.org access.
- Summary creation uses Hugging Face Inference API.
- Story/summary evaluation tries Hugging Face first, then Gemini fallback if configured.
- RAG generation expects a populated Chroma database.
- Real retrieval evaluation expects a populated Chroma database, but the metric logic is covered by lightweight tests.

To run retrieval evaluation after ingesting data:

```bash
python -m Evaluation.retrieval_eval --cases Evaluation/retrieval_eval_cases.example.json --output Evaluation/retrieval_eval_report.json --k 3
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

The full project demonstrates the larger applied AI system: ingestion, retrieval, generation, orchestration, and evaluation.
