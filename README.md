# StoryForge-RAG

End-to-end RAG pipeline: ingest stories → Chroma retrieval → grounded extraction → local LLM generation → automated evaluation.

**Full documentation:** [`docs/README.md`](docs/README.md)

## Quick start

```powershell
pip install -r requirements.txt
copy setup.example.yaml setup.yaml   # then edit paths/keys locally
python -m pytest
python main.py
```

Open `http://localhost:8000/docs` when the API is running.

## Repo layout (public)

| Path | Purpose |
|------|---------|
| `src/storyforge/` | Application package (API, RAG, vector store, orchestration) |
| `main.py` | FastAPI entry point |
| `scripts/` | Optional CLI helpers |
| `tests/` | Pytest suite |
| `data/*/sample/` | Public demo corpus only |
| `docs/` | Architecture, demo path, roadmaps |

## Personal / local only (not in git)

Your full story corpus, Chroma DB, secrets, generated outputs, and portfolio materials stay on your machine — see [`data/README.md`](data/README.md) and `.gitignore`.

## Links

- [Quick demo (no GPU)](docs/QUICK_DEMO.md)
- [Project journey](docs/PROJECT_JOURNEY.md)
- [GitHub](https://github.com/Nix-ml-journey/StoryForge-RAG)

## License

MIT — see [LICENSE](LICENSE).
