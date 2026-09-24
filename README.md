# StoryForge-RAG

End-to-end RAG for **grounded stories**, not Q&A chat: ingest public-domain text → Chroma retrieval → extract attributable facts → write a 5-section narrative on a local GPU → score and optionally refine.

Most RAG demos retrieve chunks and dump them into a prompt. This one treats generation as a **controlled pipeline**: facts must cite source chunks, length is one request field (not three knobs that disagree), and an agentic loop decides refine vs re-retrieve instead of always restarting.

**Full documentation:** [`docs/README.md`](docs/README.md) · after extract / data prep: [`docs/DATA_PREP.md`](docs/DATA_PREP.md) · design story: [`docs/PROJECT_JOURNEY.md`](docs/PROJECT_JOURNEY.md)

---

## Core idea — why this isn't another "chat with your PDF" repo

Most RAG repos on GitHub follow one shape: chunk documents, embed them, run top-k cosine
similarity, and paste the results into a prompt so an LLM can answer a question about them.
That pattern works for Q&A. It falls apart for narrative generation, where the output isn't
one factual answer — it's a few hundred to a few thousand words of connected prose that has
to stay consistent across five sections while pulling from multiple retrieved passages at
once.

StoryForge-RAG treats generation as a **controlled pipeline**, not "retrieve, then hope
the model behaves":

- **Facts are extracted before generation, not during it.** Step 2 turns retrieved chunks
  into structured JSON facts with `source_chunk_ids`. The model in Step 3 writes from that
  fact list — it never has to synthesize raw retrieved text on the fly, which is where most
  RAG hallucination actually comes from.
- **Length is one contract, not a token limit and a prayer.** A single `length` target
  drives the prompt's per-section guidance, the token budget, *and* the accept gate. Most
  RAG repos set `max_tokens` and stop there — nothing checks the output actually matches
  what was asked for.
- **The pipeline evaluates and corrects its own output.** The agentic loop scores each
  draft and picks ACCEPT / REFINE / RE_RETRIEVE based on rubric score, faithfulness, and
  completeness — not a single generate-and-return pass.
- **Retrieval is a real search stack, not `top_k=5`.** Dense + BM25 fused with RRF, a
  cross-encoder reranker, and diverse-title selection so one source book can't dominate the
  context — closer to a production search pipeline than the naive cosine-similarity loop in
  most quickstart repos.
- **The hardware target is stated up front, not assumed away.** This is built for one
  consumer 16 GB GPU with explicit local/API split (embeddings and rerank local, facts on
  HF API to avoid VRAM contention, story generation swappable between Ollama / vLLM /
  Transformers) — not "assume unlimited OpenAI budget" or "assume one fixed model that
  happens to fit on your machine."
- **Limitations are documented, not hidden.** [`docs/PRODUCTION_NOTES.md`](docs/PRODUCTION_NOTES.md)
  says plainly what's reliable (short stories), what's still improving (long-form), and what
  a production version would need. Most portfolio RAG repos oversell "production-ready."

The comparison table below has the specifics; this section is the "so what."

---

## What is different from typical RAG

| Typical RAG / story LLM | StoryForge-RAG |
|-------------------------|----------------|
| Retrieve chunks → paste into the generator | **3 steps:** retrieve → extract JSON facts with `source_chunk_ids` → generate **from facts only** |
| Open-ended “write a story” with no length contract | One `length` target (`short` / `"13min"` / `1800`) drives **prompt, token budget, and accept gate** together |
| Hallucinated names and places are common | Generation forbids new named entities; an **attribution check** logs (or truncates) names not in the facts |
| One retrieve-then-generate pass | Optional **agentic loop:** ACCEPT / REFINE (finish a grounded draft) / RE_RETRIEVE (only when faithfulness is thin) |
| Dense search only | **Hybrid BM25 + dense (RRF)** → cross-encoder **rerank** → diverse titles (rerank runs on the full pool before diversity) |
| Cloud LLM for everything, or one huge local model | **Split stack:** embeddings/rerank local, facts on HF API (no VRAM fight), story on **Ollama in Docker** (vLLM / Transformers optional) |
| Chroma’s default embedder silently mismatches ingest | Ingest **embeds with BGE explicitly** so query and corpus stay on the same 768-dim model |
| Demo quality sold as production-ready | Honest **quality tiers** (short is reliable; long-form is still being tuned) and production boundaries in [`docs/PRODUCTION_NOTES.md`](docs/PRODUCTION_NOTES.md) |
| Empty thinking drafts fail silently | **Empty-draft recovery:** thinking → one fast retry → clear error; length-guard and agentic loop fall back to the best available draft |

**Built for:** video-style narration scripts (3–16 minutes) that must stay faithful to a story corpus on a **consumer 16 GB GPU**, not for general chatbot RAG.

---

## Current model stack

| Stage | Model / Service | Where |
|-------|----------------|-------|
| Embeddings | `BAAI/bge-base-en-v1.5` (768-dim) | Local GPU via `sentence-transformers` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local GPU |
| Hybrid search | BM25 + dense (RRF fusion; needs `rank-bm25` from `requirements.txt`) | Local |
| Step 2 — grounded facts | `Qwen/Qwen3-8B` via HF Inference API | Cloud, no VRAM cost |
| Step 3 — story generation | `qwen3.5:9b` via Ollama (default) | Local GPU, `localhost:11434` |
| Evaluation | `Qwen/Qwen2.5-7B-Instruct` via HF API → Gemini fallback (or local) | Cloud, or local GPU/CPU |

Step 3 can also use **vLLM** or **Transformers** — set `Generation_provider` in `setup.yaml`.
Evaluation can run **in-process** instead of calling an API — set `Evaluation_mode: "local"` to remove the HF/Gemini round-trip from every agentic-loop iteration.

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

Manifest ingest computes **BGE embeddings explicitly** so Chroma does not fall back to a mismatched default embedder. BGE uses a **query-only** instruction prefix (`QUERY_PREFIX` in `vector_store/embeddings.py`); passages are embedded with no prefix. After any embedding-convention change, wipe and re-ingest — see [`docs/DATA_PREP.md`](docs/DATA_PREP.md).

Step 1 retrieval is **hybrid BM25 + dense (RRF) → cross-encoder rerank → title diversity**. Measure it with the real pipeline (not bare Chroma):

```powershell
py scripts/retrieval_eval.py --cases tests/fixtures/retrieval_eval_cases.example.json --k 3
```

Phase 1 retrieval tuning (2026-09) stopped at about **top1 0.80 / top3 0.90 / fact_coverage 0.77** on that harness. Details: [`docs/PROJECT_JOURNEY.md`](docs/PROJECT_JOURNEY.md).

Extracted text in `data/raw_extracted/` is scratch. Clean and split it into `data/stories/` first — [`docs/DATA_PREP.md`](docs/DATA_PREP.md).

```powershell
# Full pipeline: cleaned stories → story_json → review JSON → manifest → Chroma
py scripts/step1_prepare_and_enrich.py
py scripts/records_to_ingest_manifest.py
py scripts/ingest_manifest.py

# Or wipe and re-ingest from data/stories/ (uses data/story_json/<Title>.json when present:
# reviewed chunks + sections + Author/Summary; Title stays the filename stem)
py scripts/reset_and_ingest.py

# Check what landed in Chroma (read-only: coverage %, missing Author/Summary, bad Title/chunk_id)
py scripts/validate_chroma_metadata.py

# Phase 2 measurement: agentic length / accept rate (needs Ollama + ingested Chroma; no HTTP server)
py scripts/measure_generation_length.py --mode fast --length long

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
| `Evaluation_mode` | `api` (default, HF → Gemini) or `local` (in-process, no API round-trip) |
| `Local_evaluation_model` / `_device` | Local judge (default `Qwen/Qwen2.5-3B-Instruct` on CPU) |

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

- [Data prep after extract](docs/DATA_PREP.md)
- [Quick demo / how to use](docs/QUICK_DEMO.md)
- [Project journey](docs/PROJECT_JOURNEY.md)
- [Upgrade roadmap](docs/UPGRADE_ROADMAP_5060Ti.md)
- [Production notes](docs/PRODUCTION_NOTES.md)
- [Pattern audit (historical)](StoryForge_pattern_audit.md)
- [GitHub](https://github.com/Nix-ml-journey/StoryForge-RAG)

## License

MIT — see [LICENSE](LICENSE).
