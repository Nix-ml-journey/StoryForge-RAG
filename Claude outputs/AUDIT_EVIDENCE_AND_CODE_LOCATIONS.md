# Evidence & Code Locations for StoryForge-RAG Audit

**Original audit:** 2026-09-24  
**Verification update:** 2026-09-24 — P0.1 / P0.2 / P0.3 implemented and tested (`156 passed`).

---

## Verification summary

| Finding | Was | Now |
|---------|-----|-----|
| P0.1 rebuild ingest ignores story_json | True for `ingest_stories_dir` / `reset_and_ingest` | **Fixed** — loads story_json; stale detection; aligned manifest metadata |
| P0.1 “manifest never used” | Overstated — manifest path existed | Corrected: gap was rebuild path, not all ingest |
| P0.2 local JSON fallback fragile | True | **Fixed** — schema/format, salvage, citation filter, retry, provider=local |
| P0.3 no-eval ACCEPT with 0 facts | True | **Fixed** — facts checked first; `has_eval` on iterations |
| P0.4 embed mismatch | Process risk, not proven live breakage | **Still open** as P1 health-check |

---

## P0.1: story_json Metadata — FIXED

### Where the fix lives
| File | What to look at |
|------|-----------------|
| `src/storyforge/vector_store/ingest_stories.py` | `_load_story_json`, `_story_json_plan`, `_norm_text`, `ingest_stories_dir(..., records_dir=..., use_story_json=True)`, `IngestResult.from_story_json` / `stale_story_json` |
| `src/storyforge/data/records_to_manifest.py` | Same Author / Summary / Display_title / Is_series contract; Title = filename stem |
| `tests/test_ingest_story_json_metadata.py` | 8 regression tests (CRLF uses `write_bytes`) |

### Behavior now
- Matching `raw_text` ↔ `.txt` → embed reviewed chunks + section tags + story metadata
- Stale JSON → chunk from `.txt`, keep Author/Summary, warn
- No JSON → old behaviour (empty Author/Summary)

### Still true
- `Title` metadata remains the **filename stem** (chunk ids + retrieval_eval fixtures depend on it). Human title goes to `Display_title`.

---

## P0.2: Local Ollama JSON Fallback — FIXED

### Where the fix lives
| File | What to look at |
|------|-----------------|
| `src/storyforge/rag/extraction.py` | `Grounded_facts_provider`, local extract + retries, format/schema |
| `src/storyforge/rag/attribution.py` | `salvage_grounded_facts_json` |
| `src/storyforge/rag/generation_backend.py` | Ollama `format=` / json_format passthrough |
| `setup.example.yaml` | `Grounded_facts_provider`, `Local_grounded_facts_*` |
| `tests/test_local_facts_fallback.py` | 24 tests |

### Offline switch
```yaml
Grounded_facts_provider: "local"
# optional while HF is down:
Evaluation_mode: "local"
Local_evaluation_device: "cpu"
```

---

## P0.3: No-Eval Grounding Contract — FIXED

### Where the fix lives
| File | What to look at |
|------|-----------------|
| `src/storyforge/rag/agentic_loop.py` | `decide_action`: facts_count check **before** completeness ACCEPT when `not has_eval` |
| same | iteration dict includes `has_eval` |
| same | `stop_reason = max_iterations_no_grounded_facts` when exhausted with zero facts |
| `tests/test_agentic_loop.py` | no-eval + facts_count=0 → RE_RETRIEVE |

### Historical evidence (pre-fix)
`docs/PROJECT_JOURNEY.md` Phase 2 notes: accept_rate 0.88 under no-eval completeness-only path; later collapse when eval returned and facts were zero.

---

## P0.4: Embedding Model Mismatch — OPEN (P1)

| File | Finding |
|------|---------|
| `docs/DATA_PREP.md` | BGE passage-prefix fix — re-ingest required |
| `src/storyforge/vector_store/ingest_stories.py` | Query-only BGE convention documented |
| (missing) | Collection fingerprint / auto mismatch detect |

Not a live P0 if you already re-ingested after the 2026-09 BGE fix.

---

## Strengths (unchanged — still accurate)

- LengthProfile unified target (`length_profile.py`)
- Hybrid BM25 + dense RRF → rerank-before-diversity (`retrieval.py`)
- Empty-draft recovery (thinking → fast)
- Config-driven providers; `rank-bm25` required in `requirements.txt`

---

## Config keys added / documented for offline work

| Key | Purpose |
|-----|---------|
| `Grounded_facts_provider` | `"hf"` \| `"local"` |
| `Local_grounded_facts_json_format` | `"schema"` (default) or `"json"` for older Ollama |
| `Local_grounded_facts_retries` | compact retry count |
| `Local_grounded_facts_compact_max_facts` | retry asks for ≤ N facts |
| `Evaluation_mode: "local"` | offline scoring; keep device on CPU beside Ollama |

---

## Import side-effect (still open, separate)

`src/storyforge/vector_store/chromadb.py` opens a PersistentClient at import time (module-level collection). Can cause path / network-drive test noise. Worth a later lazy-init fix — not part of P0.1–P0.3.

---

## After HF credits renew

1. Set `Grounded_facts_provider: "hf"` again  
2. `py scripts/measure_generation_length.py --mode fast --length long`  
3. Split accept rate by iteration `has_eval`  
4. Compare HF vs local `facts_count` on the same queries  
5. Optionally run salvage parser on HF truncated answers too
