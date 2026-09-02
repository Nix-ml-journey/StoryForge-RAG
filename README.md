# StoryForge-RAG

End-to-end RAG for **grounded stories**, not Q&A chat: ingest public-domain text → Chroma retrieval → extract attributable facts → write a 5-section narrative on a local GPU → score and optionally refine.

Most RAG demos retrieve chunks and dump them into a prompt. This one treats generation as a **controlled pipeline**: facts must cite source chunks, length is one request field (not three knobs that disagree), and an agentic loop decides refine vs re-retrieve instead of always restarting.

**Full documentation:** [`docs/README.md`](docs/README.md) · design story: [`docs/PROJECT_JOURNEY.md`](docs/PROJECT_JOURNEY.md)

---

## What is different from typical RAG

| Typical RAG / story LLM | StoryForge-RAG |
|-------------------------|----------------|
| Retrieve chunks → paste into the generator | **3 steps:** retrieve → extract JSON facts with `source_chunk_ids` → generate **from facts only** |
| Open-ended “write a story” with no length contract | One `length` target (`short` / `"13min"` / `1800`) drives **prompt, token budget, and accept gate** together |
| Hallucinated names and places are common | Generation forbids new named entities; an **attribution check** logs (or truncates) names not in the facts |
| One retrieve-then-generate pass | Optional **agentic loop:** ACCEPT / REFINE (finish a grounded draft) / RE_RETRIEVE (only when faithfulness is thin) |
| Dense search only | **Hybrid BM25 + dense (RRF)** + cross-encoder rerank + diverse titles so one book does not dominate |
| Cloud LLM for everything, or one huge local model | **Split stack:** embeddings/rerank local, facts on HF API (no VRAM fight), story on **Ollama in Docker** (vLLM / Transformers optional) |
| Chroma’s default embedder silently mismatches ingest | Ingest **embeds with BGE explicitly** so query and corpus stay on the same 768-dim model |
| Demo quality sold as production-ready | Honest **quality tiers** (short is reliable; long-form is still being tuned) and production boundaries in [`docs/PRODUCTION_NOTES.md`](docs/PRODUCTION_NOTES.md) |

**Built for:** video-style narration scripts (3–16 minutes) that must stay faithful to a story corpus on a **consumer 16 GB GPU**, not for general chatbot RAG.

---

## Current model stack

| Stage | Model / Service | Where |
|-------|----------------|-------|
| Embeddings | `BAAI/bge-base-en-v1.5` (768-dim) | Local GPU via `sentence-transformers` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local GPU |
| Hybrid search | BM25 + dense (RRF fusion) | Local |
| Step 2 — grounded facts | `Qwen/Qwen3-8B` via HF Inference API | Cloud, no VRAM cost |
| Step 3 — story generation | `qwen3.5:9b` via Ollama (default) | Local GPU, `localhost:11434` |
| Evaluation | `Qwen/Qwen2.5-7B-Instruct` via HF API → Gemini fallback | Cloud |

Step 3 can also use **vLLM** or **Transformers** — set `Generation_provider` in `setup.yaml`.

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

Set `Host: "0.0.0.0"` in `setup.yaml` if you need access from other devices or containers.

---

## Story format

Generated stories use **5 simple sections** with headers like `[SECTION 1: WHO, WHERE, WHEN (The Setup)]`.

Rules enforced by prompts and code:
- All 5 sections must be present
- Story must stay grounded in retrieved facts (no invented named characters/places)
- Each section must reach the sentence and word minimums for the requested length

### Story length

`length` sets one target that drives the prompt's words-per-section
instruction, the token budget, and the accept gate together — so raising the
target actually produces a longer story instead of a bigger unused budget.

```bash
# A ~10-minute narration script for a video
curl -s -X POST http://localhost:8000/create-eval/story_generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Amun Chronicles", "mode": "thinking", "length": "10min"}'
```

Accepted values: a preset name (`short` 450 words, `medium` 900, `long` 1500,
`epic` 2200), a narration duration (`"12min"`), or a word count (`"1800"`).
Omit it to use the default for the mode. `mode` still controls sampling and
thinking; `length` controls how much prose gets written.

---

## Demo: generate a story

After ingesting at least one story (`POST /vector_store/ingest_stories`):

```bash
# Single-pass generation (default short length for fast mode)
curl -s -X POST http://localhost:8000/orchestration/run_step \
  -H "Content-Type: application/json" \
  -d '{"step": "4_generate_story_3step", "title": "Amun Chronicles", "mode": "fast"}' | python -m json.tool

# Via create-eval API (~13-minute narration target)
curl -s -X POST http://localhost:8000/create-eval/story_generate \
  -H "Content-Type: application/json" \
  -d '{"query": "A warrior monk faces his greatest trial", "mode": "fast", "length": "13min", "save": true}' | python -m json.tool

# Streaming generation (SSE — tokens stream in real-time)
curl -N http://localhost:8000/orchestration/generate_stream \
  -X POST -H "Content-Type: application/json" \
  -d '{"query": "A warrior monk faces his greatest trial", "mode": "fast", "length": "medium"}'
```

When `Agentic_loop_enabled: true`, step 4 uses evaluate → refine / re-retrieve → accept automatically. Completeness uses the resolved length target.

---

## Ingest and Chroma maintenance

Manifest ingest now computes **BGE embeddings explicitly** so Chroma does not fall back to a mismatched default embedder.

```powershell
# Full pipeline: stories → story_json → manifest → Chroma
py scripts/step1_prepare_and_enrich.py
py scripts/records_to_ingest_manifest.py
py scripts/ingest_manifest.py

# Or wipe and re-ingest from data/stories/
py scripts/reset_and_ingest.py

# After editing chunk text in story_json (re-embed only, keep metadata)
py scripts/refresh_chunk_embeddings.py --glob "Lovecraft__*"

# After enrich adds section tags (update metadata only, no re-embed)
py scripts/push_section_metadata.py --glob "Lovecraft__*"
```

---

## Repo layout (public)

| Path | Purpose |
|------|---------|
| `src/storyforge/` | Application package (API, RAG, vector store, orchestration) |
| `src/storyforge/rag/retrieval.py` | Step 1: Chroma + hybrid BM25 retrieval + reranker |
| `src/storyforge/rag/extraction.py` | Step 2: HF API grounded facts (retry + JSON mode) |
| `src/storyforge/rag/generation.py` | Step 3: Ollama / vLLM / Transformers story generation |
| `src/storyforge/rag/length_profile.py` | Resolve `length` → prompt + tokens + accept gate |
| `src/storyforge/rag/langchain_rag.py` | 3-step orchestrator + length guard |
| `main.py` | FastAPI entry point |
| `docker-compose.yml` | Ollama service (GPU passthrough) |
| `scripts/` | CLI helpers (ingest, eval, diagnostics) |
| `tests/` | Pytest suite |
| `data/*/sample/` | Public demo corpus only |
| `docs/` | Architecture, demo path, roadmaps |

---

## Key config knobs (`setup.example.yaml`)

| Setting | What it does |
|---------|--------------|
| `Generation_provider` | `ollama` (default), `vllm`, or `transformers` |
| `Vector_store_model` | Embedding model (default BGE-base) |
| `Hybrid_search_enabled` | BM25 + dense fusion |
| `Story_length_presets` | Named length targets (name → target word count) |
| `Story_length_default_fast` / `_thinking` | Default length target per mode |
| `Generation_fast_*` / `Generation_thinking_*` | Sampling and token floors per mode |
| `Agentic_loop_*` | Evaluate/refine/re-retrieve loop thresholds |
| `HF_grounded_facts_json_mode` | Strict JSON for Step 2 (falls back if unsupported) |

Copy `setup.example.yaml` → `setup.yaml` and edit locally. Secrets stay out of git.

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
