# Evidence & Code Locations for StoryForge-RAG Audit

**Original audit:** 2026-09-24  
**Verification update:** 2026-09-24 — P0.1 / P0.2 / P0.3 implemented and tested; owner's `.venv`: **`163 passed`**.
**Machine verification:** P0.1 rebuild ingest run on the real corpus (72 story_json / 0 stale / 0 missing, 2138 chunks); `validate_chroma_metadata.py` and `retrieval_eval.py` run via `.\.venv\Scripts\python.exe`.

---

## Verification summary

| Finding | Was | Now |
|---------|-----|-----|
| P0.1 rebuild ingest ignores story_json | True for `ingest_stories_dir` / `reset_and_ingest` | **Fixed + verified on machine** — 72/0/0 story_json ingest, Author 93.6% in Chroma |
| P0.1 “manifest never used” | Overstated — manifest path existed | Corrected: gap was rebuild path, not all ingest |
| P0.2 local JSON fallback fragile | True | **Fixed** — schema/format, salvage, citation filter, retry, provider=local |
| P0.3 no-eval ACCEPT with 0 facts | True | **Fixed** — facts checked first; `has_eval` on iterations |
| P0.4 embed mismatch | Process risk, not proven live breakage | **Still open** as P1 health-check |
| Offline follow-up (docs + `validate_chroma_metadata.py`) | — | **Verified** — 163 passed; report run on the 2138-chunk collection, `PROBLEMS: none` |

---

## P0.1: story_json Metadata — FIXED + VERIFIED ON MACHINE

### Where the fix lives
| File | What to look at |
|------|-----------------|
| `src/storyforge/vector_store/ingest_stories.py` | `_load_story_json`, `_story_json_plan`, `_norm_text`, `_read_text_keep_newlines` (Py3.10-safe CRLF-preserving read), `ingest_stories_dir(..., records_dir=..., use_story_json=True)`, `IngestResult.from_story_json` / `stale_story_json` |
| `src/storyforge/data/records_to_manifest.py` | Same Author / Summary / Display_title / Is_series contract; Title = filename stem |
| `tests/test_ingest_story_json_metadata.py` | 8 regression tests (CRLF uses `write_bytes`) |

### Behavior now
- Matching `raw_text` ↔ `.txt` → embed reviewed chunks + section tags + story metadata
- Stale JSON → chunk from `.txt`, keep Author/Summary, warn
- No JSON → old behaviour (empty Author/Summary)

### Still true
- `Title` metadata remains the **filename stem** (chunk ids + retrieval_eval fixtures depend on it). Human title goes to `Display_title`.

### Machine evidence (2026-09-24)
| Check | Result |
|---|---|
| `reset_and_ingest.py` | `From story_json: 72 · stale: 0 · no story_json: 0`, Files 72, Chunks 2138 |
| `validate_chroma_metadata.py` | `PROBLEMS: none` · Author 93.6% · Display_title 93.6% · Summary 6.4% · section 45.8% |
| `retrieval_eval.py --k 3` (venv) | top1 0.80 · top3 0.833 · fact_coverage 0.733 (Phase 1 baseline 0.80 · 0.90 · 0.77) |
| Ops | `WinError 32` deleting `data/chroma_db` (locked file) → Chroma API reset fallback, rebuild OK. Not a P0 |

**Where the percentages come from** (read from `data/story_json/*.json`):
- 35 classics (Kafka, Lovecraft, Frankenstein, Jekyll, Wells, Doyle) → `meta.author` + `meta.title` set, **`summary` empty** → 2002 chunks = the 93.6% Author / Display_title and the 93.6% missing Summary.
- 37 Firestone Idle RPG wiki pages (Amun … Zoruk) → **`summary` set, `meta.author` / `meta.title` empty** → 136 chunks = 6.4%. They also carry template placeholders: all 37 share `id: "id_01"` and have `Is_series: true` with an empty `series_name`. Nothing in `rag/` or `evaluation/` reads the `id` metadata, so this is hygiene, not a retrieval bug.
- 16 stories have no `section` tags at all → section 45.8% (P2).

**Interpretation:** the remaining gaps are content, not code. Firestone pages use a fixed convention (`meta.author: "Firestone Idle"`, `meta.title` = page name, `id` = stem, `Is_series: false`) instead of looking up a novelist; the recipe was dry-run on a copy (37 updated, re-run 0, all 72 still `story_json`, not stale). Classic summaries wait for HF enrich or stay empty. Recipe: `P0_CRITICAL_ISSUES_CHECKLIST.md` → "Firestone metadata recipe".

**Tooling limits found:** `scripts/push_section_metadata.py` refreshes `section` / `meta_json` only, **not** top-level `Author` / `Display_title` / `Summary` → after metadata edits, rebuild with `reset_and_ingest.py`. Bare `py scripts\...` runs system Python (`ModuleNotFoundError: langchain_chroma`) → use `.\.venv\Scripts\python.exe`.

**retrieval_eval shift:** top1 held; top3 −2 cases of 30 and coverage −0.037 after switching the collection to reviewed story_json chunks. Expected; follow up as corpus/chunk work (compare `Evaluation/retrieval_eval_report.json` per case), not retrieval-knob tuning.

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
2. `.\.venv\Scripts\python.exe scripts\measure_generation_length.py --mode fast --length long`  
3. Split accept rate by iteration `has_eval`  
4. Compare HF vs local `facts_count` on the same queries  
5. Optionally run salvage parser on HF truncated answers too
6. Classic `summary` enrich for the 35 books (or keep empty offline)

## Still open (not HF-dependent)

- P1: collection embedding fingerprint (old P0.4)
- Optional now: Firestone metadata fill → rebuild → validate (expect Author 100%)
- Optional now: P0.2 Ollama smoke probe with `Grounded_facts_provider: "local"`
- P2: section tags for 16 untagged stories
