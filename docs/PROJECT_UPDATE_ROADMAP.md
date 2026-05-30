# StoryForge-RAG Project Update Roadmap

This document is an honest review of the current project as a job-search portfolio piece. The project has real technical value: it is not just a prompt demo. It covers data ingestion, metadata preparation, vector search, local generation, orchestration, FastAPI routes, and LLM-based evaluation.

The reason it may not be helping you get interviews or offers yet is not that the idea is bad. The issue is that the repository does not yet make the project easy to trust, easy to run, easy to review, or easy to connect to a hiring manager's expectations.

## What This Project Already Shows

Strong points:

- You built an end-to-end AI/data pipeline, not only a notebook.
- The architecture has separated areas: `Book_search`, `Data`, `Vector_Store`, `Generative_AI`, `Evaluation`, `Orchestrator`, and `API`.
- The generation path has meaningful engineering ideas: grounded extraction, retrieval filtering, query-context alignment checks, repetition controls, token budgeting, and fallback behavior.
- The project documents real constraints instead of pretending everything is perfect.
- The README and project journey already explain several trade-offs: model size, OOM handling, API rate limits, retrieval quality, and evaluation availability.

This is a good base for an AI engineering / data engineering portfolio project.

## Why This Project May Not Be Getting You a Job Yet

### 1. It is difficult for a reviewer to run

The app depends on `setup.yaml`, local folders, API keys, local Chroma data, Hugging Face access, Gemini access, and a local generation model. `setup.example.yaml` helps, but a reviewer still cannot quickly verify the full project in 5-10 minutes.

Hiring managers often skim. If they cannot run a simple demo quickly, they may move on even if the system is interesting.

Fix:

- Add a `QUICK_DEMO.md` with one fast path that works without downloading big models.
- Add a small sample dataset committed to the repo, such as 2-3 tiny public-domain story excerpts.
- Add a lightweight demo mode that uses mocked generation/evaluation responses.
- Add curl examples for the main API routes.

### 2. There are no automated tests

Search found no `test*.py` files and no clear pytest/unittest coverage. This is one of the biggest job-readiness gaps.

Without tests, the project looks like a personal experiment instead of a maintainable engineering project.

Fix:

- Add unit tests for data merge behavior.
- Add tests for metadata summary checks.
- Add tests for section parsing.
- Add tests for query-context alignment.
- Add FastAPI route tests with `TestClient` and mocked heavy dependencies.
- Add one integration-style test that runs a tiny local pipeline with sample files.

Minimum first tests:

```text
tests/test_section_parse.py
tests/test_metadata.py
tests/test_data_merge.py
tests/test_generation_guards.py
tests/test_api_contracts.py
```

### 3. The repo still looks like active debugging work

There are many modified, deleted, and untracked files. Some file names also feel experimental, such as `check_cuda.py`, `model.py`, `Data/check_merged_data.py`, and several fix/check scripts.

This is normal while building, but for hiring it can look unfinished.

Fix:

- Clean the git status before sharing the repo.
- Move scripts into a clear `scripts/` folder.
- Rename vague files:
  - `model.py` -> `scripts/list_gemini_models.py`
  - `check_cuda.py` -> `scripts/check_cuda_compatibility.py`
  - `Data/check_merged_data.py` -> `scripts/check_merged_data.py` or `Data/validation.py`
- Remove old deleted files from the repo history/current branch if they are no longer needed.
- Keep generated outputs and local data out of git.

### 4. The project lacks a polished reviewer story

The README explains a lot, but it is long and technical. A recruiter needs a shorter proof:

- What problem did you solve?
- What did you build?
- What engineering decisions did you make?
- What measurable result improved?
- How can I run or watch it?

Fix:

- Add a short top-level section: "Demo in 3 minutes".
- Add screenshots or short terminal/API examples.
- Add before/after examples of generation quality.
- Add an architecture diagram.
- Add a "What I would improve in production" section.

### 5. Some implementation choices look fragile

Several modules read `setup.yaml` at import time. That makes tests and deployment harder because importing a module can fail if local config is missing.

Examples:

- `main.py`
- `Book_search/fetch_book.py`
- `Book_search/extract_text.py`
- `Data/metadata.py`
- `Vector_Store/chromadb.py`
- `Generative_AI/config.py`
- `Evaluation/evaluation.py`

Fix:

- Create one typed settings loader.
- Pass config into services instead of reading global config in many modules.
- Avoid heavy model/database initialization at import time.
- Make modules importable even when API keys or local folders are missing.

### 6. The vector store design needs clearer retrieval evaluation

The README says the vector store currently has about 47 records. That is too small to prove robust RAG. The project also needs more evidence that retrieval quality is good.

Fix:

- Add a retrieval evaluation dataset: query, expected title, expected key facts.
- Track top-k recall and whether retrieved context matches the intended story.
- Add chunking strategy documentation.
- Compare retrieval from summaries vs full text chunks.
- Log retrieval examples in a reproducible report.

### 7. The project needs clearer production boundaries

The system uses external services, local GPU inference, Chroma, file outputs, and manual metadata steps. That is fine, but the repo should clearly explain what is production-ready and what is experimental.

Fix:

- Add a `PRODUCTION_NOTES.md`.
- Explain current limitations honestly:
  - local GPU dependency,
  - external API instability,
  - small indexed corpus,
  - incomplete generated stories,
  - manual metadata quality checks.
- Then explain how you would productionize:
  - job queue,
  - persistent database,
  - object storage,
  - observability,
  - retries,
  - eval dashboards,
  - CI/CD,
  - secrets management.

## Priority Roadmap

### Priority 1: Make it reviewable in one sitting

Goal: A hiring manager should understand and run something in under 10 minutes.

Tasks:

- Add `QUICK_DEMO.md`.
- Add tiny sample data.
- Add demo mode with mocked generation/evaluation.
- Add exact commands:
  - install,
  - run tests,
  - start API,
  - call one endpoint.
- Add an architecture diagram to `README.md`.

### Priority 2: Add automated tests

Goal: Prove that the project is engineered, not only manually tested.

Tasks:

- Add `pytest`.
- Add tests for pure functions first.
- Mock Hugging Face, Gemini, Chroma, and model generation.
- Add CI with GitHub Actions.
- Add a badge to `README.md` once tests pass.

Suggested first test targets:

- `Generative_AI/section_parse.py`
- `Generative_AI/penalty_processors.py`
- `Data/data_merge.py`
- `Data/metadata.py`
- `API/create_eval_routes.py`

### Priority 3: Clean configuration and import behavior

Goal: Make the code easier to test, deploy, and review.

Tasks:

- Move config loading into one module.
- Use a typed settings object.
- Stop reading `setup.yaml` in many files at import time.
- Make missing API keys fail only when the specific feature is used.
- Keep `.env`, `setup.yaml`, local databases, and outputs out of git.

### Priority 4: Improve RAG evidence

Goal: Show measurable retrieval and generation quality.

Tasks:

- Create `Evaluation/retrieval_eval.py`.
- Add 20-50 evaluation queries.
- Measure top-1 and top-3 retrieval accuracy.
- Save a small report.
- Show examples where query-context alignment prevents wrong generation.

### Priority 5: Improve project packaging

Goal: Make the repo look like a serious Python project.

Tasks:

- Add `pyproject.toml`.
- Add `ruff` or `black`.
- Add `pytest`.
- Add `mypy` only if you are ready to maintain types.
- Move scripts into `scripts/`.
- Consider a package layout later:

```text
storyforge/
  api/
  data/
  generation/
  retrieval/
  evaluation/
tests/
scripts/
docs/
```

## What To Add To The README

Add these sections near the top:

```markdown
## Demo In 3 Minutes

This demo runs with sample data and mocked model calls, so reviewers can verify the API and pipeline without a GPU or paid API keys.

1. pip install -r requirements.txt
2. python -m pytest
3. python main.py
4. curl -X POST http://localhost:8000/create-eval/story_generate ...
```

```markdown
## Engineering Highlights

- Built a staged RAG generation pipeline: retrieval -> 5W1H extraction -> section plan -> story expansion.
- Added guardrails for query-context mismatch and repetitive generation.
- Designed FastAPI endpoints around orchestration, vector store inspection, generation, and evaluation.
- Added external evaluation with retry/fallback behavior.
```

```markdown
## Current Limitations

- Full generation requires local model setup and enough GPU memory.
- Evaluation depends on Gemini availability.
- Current corpus is small, so retrieval quality is still being improved.
- The project is optimized for learning and portfolio demonstration, not production traffic yet.
```

## How To Talk About This Project In Interviews

Use this framing:

> I built StoryForge-RAG as an end-to-end AI engineering project. It ingests public-domain books, prepares metadata, stores records in Chroma, retrieves story context, extracts grounded 5W1H facts, generates the final story in a single pass, and evaluates outputs with a rubric. The most important lesson was that long-form generation quality depends as much on retrieval quality, chunking, and evaluation as on the model itself.

Then explain trade-offs:

- Why you chose sentence-transformers for embeddings.
- Why you moved from raw retrieval single-pass to grounded extraction plus single-pass generation.
- Why local model size and token budgets mattered.
- Why API instability forced retry/fallback logic.
- Why retrieval evaluation is your next major improvement.

Avoid saying:

- "It is not complete."
- "AI wrote most of it."
- "I just followed tutorials."

Say instead:

- "This is an active portfolio project with documented limitations."
- "I used AI tools as an assistant, but I made the architecture and trade-off decisions."
- "The next engineering step is adding tests, demo mode, and retrieval metrics."

## Best Resume Bullet Version

Use bullets like these:

- Built an end-to-end RAG story-generation pipeline with FastAPI, ChromaDB, sentence-transformers, local Hugging Face generation, and LLM-based evaluation.
- Designed a grounded generation workflow using retrieval, 5W1H extraction, and single-pass story generation to reduce fact loss and hallucination.
- Added reliability guardrails for query-context mismatch, repetition loops, GPU OOM handling, API retry/fallback behavior, and generated-output cleanup.
- Implemented data ingestion, metadata merging, vector-store ingestion, story generation, and evaluation endpoints for repeatable pipeline execution.

## 30-Day Action Plan

Week 1:

- Clean git status.
- Add `QUICK_DEMO.md`.
- Add tiny sample data.
- Add basic pytest setup.

Week 2:

- Add tests for parsing, metadata, merge, and generation guardrails.
- Add FastAPI route tests with mocked services.
- Add GitHub Actions CI.

Week 3:

- Refactor config loading.
- Remove import-time dependency on local secrets where possible.
- Move helper scripts into `scripts/`.

Week 4:

- Add retrieval evaluation.
- Add before/after generation examples.
- Update README with demo, architecture diagram, test instructions, and limitations.

## Final Honest Assessment

This project can help you get a job, but not yet in its current presentation.

Right now, it shows effort, ambition, and real learning. To become a strong hiring project, it must show trust signals: tests, reproducibility, clean structure, demo data, measurable results, and a short explanation that non-specialists can understand.

Your best next move is not adding more generation features. Your best next move is making the project easier to run, easier to verify, and easier to discuss in interviews.
