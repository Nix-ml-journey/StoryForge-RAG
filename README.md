# StoryForge-RAG

End-to-end **RAG pipeline** for grounded narrative generation: ingest stories → Chroma retrieval → fact extraction → local 7B generation → automated evaluation. Optional **agentic loop** (evaluate → refine / re-retrieve → accept).

Built as a practical AI/data engineering portfolio piece on consumer GPU hardware.

## Quick start

```bash
pip install -r requirements.txt
copy setup.example.yaml setup.yaml   # then add your keys/paths
python main.py
# API docs: http://localhost:8000/docs
```

Run tests (no GPU or API keys required):

```bash
python -m pytest
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [`docs/README.md`](docs/README.md) | Full architecture, config, API flow, tests |
| [`docs/PROJECT_JOURNEY.md`](docs/PROJECT_JOURNEY.md) | Design decisions, failures, lessons learned |
| [`docs/QUICK_DEMO.md`](docs/QUICK_DEMO.md) | Fast reviewer path without heavy local models |
| [`docs/portfolio/`](docs/portfolio/) | Portfolio markdown + PDF for job applications |
| [`legacy/README.md`](legacy/README.md) | Deprecated pre-`src/storyforge` packages (reference only) |

## Project layout

```text
main.py              # FastAPI entry point
setup.yaml           # Local config (gitignored; copy from setup.example.yaml)
prompts.yaml         # Generation / extraction prompt templates
src/storyforge/      # Active application code
tests/               # Lightweight pytest suite
scripts/             # Optional CLI helpers (ingest, CUDA check, portfolio PDF)
data/                # Local stories, Chroma DB, outputs (runtime data)
docs/                # Documentation and portfolio assets
legacy/              # Old root-level packages (deprecated)
```

## Repo

https://github.com/Nix-ml-journey/StoryForge-RAG

## License

MIT
