# StoryForge-RAG

End-to-end RAG pipeline: ingest stories → Chroma retrieval → grounded extraction → local LLM generation → automated evaluation.

**Full documentation:** [`docs/README.md`](docs/README.md)

---

## Current model stack

| Stage | Model / Service | Where |
|-------|----------------|-------|
| Embeddings | `BAAI/bge-base-en-v1.5` (768-dim) | Local GPU via `sentence-transformers` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local GPU |
| Step 2 — grounded facts | `Qwen/Qwen3-8B` via HF Inference API | Cloud, no VRAM cost |
| Step 3 — story generation | `qwen3.5:9b` via Ollama (Docker) | Local GPU, `localhost:11434` |
| Evaluation | `Qwen/Qwen2.5-7B-Instruct` via HF API → Gemini fallback | Cloud |

---

## Quick start

**Prerequisites:** Docker Desktop + NVIDIA Container Toolkit for Ollama GPU passthrough.

```powershell
# 1. Start Ollama in Docker (first run pulls ~6 GB)
docker compose up -d

# 2. Pull the generation model into Ollama
docker exec -it ollama ollama pull qwen3.5:9b

# 3. Install Python deps
pip install -r requirements.txt

# 4. Create your local config (never committed)
copy setup.example.yaml setup.yaml   # then edit BASE_PATH and API keys

# 5. Run tests
python -m pytest

# 6. Start the API
python main.py
```

Open `http://localhost:8000/docs` to explore all endpoints.

---

## Demo: generate a story (3-step RAG)

After ingesting at least one story (`POST /vector_store/ingest_stories`):

```bash
# Single-pass generation
curl -s -X POST http://localhost:8000/orchestration/run_step \
  -H "Content-Type: application/json" \
  -d '{"step": "4_generate_story_3step", "title": "Amun Chronicles"}' | python -m json.tool

# Streaming generation (SSE — tokens stream in real-time)
curl -N http://localhost:8000/orchestration/generate_stream \
  -X POST -H "Content-Type: application/json" \
  -d '{"query": "A warrior monk faces his greatest trial", "mode": "fast"}'
```

Example SSE output:

```
data: {"step": "retrieve", "status": "start"}
data: {"step": "retrieve", "status": "done", "n_chunks": 6}
data: {"step": "extract", "status": "done"}
data: {"step": "generate", "token": "Once "}
data: {"step": "generate", "token": "upon a time..."}
...
data: [DONE]
```

---

## Repo layout (public)

| Path | Purpose |
|------|---------|
| `src/storyforge/` | Application package (API, RAG, vector store, orchestration) |
| `src/storyforge/rag/retrieval.py` | Step 1: Chroma + hybrid BM25 retrieval |
| `src/storyforge/rag/extraction.py` | Step 2: HF API grounded facts extraction |
| `src/storyforge/rag/generation.py` | Step 3: Ollama / Transformers story generation |
| `src/storyforge/rag/langchain_rag.py` | 3-step orchestrator + backward-compat re-exports |
| `main.py` | FastAPI entry point |
| `docker-compose.yml` | Ollama service (GPU passthrough) |
| `scripts/` | Optional CLI helpers |
| `tests/` | Pytest suite |
| `data/*/sample/` | Public demo corpus only |
| `docs/` | Architecture, demo path, roadmaps |

---

## Personal / local only (not in git)

Your full story corpus, Chroma DB, secrets, generated outputs, and portfolio materials stay on your machine — see [`data/README.md`](data/README.md) and `.gitignore`.

---

## Environment variables (preferred over setup.yaml secrets)

```
STORYFORGE_HF_API_KEY  or  HUGGINGFACEHUB_API_TOKEN  or  HF_TOKEN
GEMINI_API_KEY
GOOGLE_BOOKS_API_KEY
```

---

## Links

- [Quick demo (no GPU)](docs/QUICK_DEMO.md)
- [Project journey](docs/PROJECT_JOURNEY.md)
- [Upgrade roadmap](docs/UPGRADE_ROADMAP_5060Ti.md)
- [GitHub](https://github.com/Nix-ml-journey/StoryForge-RAG)

## License

MIT — see [LICENSE](LICENSE).
