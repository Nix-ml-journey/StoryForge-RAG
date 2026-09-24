# StoryForge-RAG Critical Issues Checklist

**Original audit:** 2026-09-24  
**Verification update:** 2026-09-24 (local / no HF credits)

**Suite status:** `163 passed` in project `.venv` after P0.1–P0.3 + offline follow-up (`validate_chroma_metadata` + Py3.10-safe newline read).

---

## Status board

| ID | Issue | Status | Notes |
|----|--------|--------|-------|
| **P0.1** | story_json metadata ignored by rebuild ingest | **FIXED** | `ingest_stories_dir()` now reads `story_json`; Title stays filename stem; Author/Summary/Display_title/Is_series + reviewed chunks used when raw_text matches |
| **P0.2** | Local Ollama facts JSON brittle | **FIXED (code)** | Schema/format, salvage parser, citation check, retry, `Grounded_facts_provider: "local"`. Still needs one Ollama smoke probe on your machine |
| **P0.3** | No-eval ACCEPT with `facts_count=0` | **FIXED** | `decide_action` checks facts first; iterations record `has_eval`; stop reason `max_iterations_no_grounded_facts` |
| **P0.4** | Embed prefix / model mismatch detection | **OPEN (P1)** | Re-ingest already done after BGE fix; still no automatic collection fingerprint |

---

## P0.1: story_json Workflow ↔ Ingest — FIXED

### Correction to original audit
`scripts/ingest_manifest.py` already could ingest from the manifest built from `story_json`. The real gap was **`reset_and_ingest.py` → `ingest_stories_dir()`**, which hardcoded empty Author/Summary and ignored reviewed chunks.

### Done in code
- [x] `ingest_stories_dir()` loads `data/story_json/<Title>.json`
- [x] Writes Author / Summary / Display_title / Is_series (Title = filename stem)
- [x] Uses reviewed chunks + section tags when `raw_text` matches `.txt`
- [x] Stale records: `.txt` chunks + story-level metadata only + warning
- [x] `records_to_manifest.py` aligned on same metadata fields
- [x] Tests: `tests/test_ingest_story_json_metadata.py` (8)
- [x] Windows CRLF: `_norm_text` + `read_text(newline="")` + test writes via `write_bytes`

### Verify on your machine (no HF)
```powershell
py scripts/reset_and_ingest.py
# Look for: "From story_json / stale / without" counts in the log
py scripts/validate_chroma_metadata.py
py scripts/peek_vector_store.py
py scripts/retrieval_eval.py --cases tests/fixtures/retrieval_eval_cases.example.json --k 3
```

```powershell
py -c "import sys;sys.path.insert(0,'src');from storyforge.vector_store.chromadb import get_or_create_collection;c=get_or_create_collection();r=c.get(limit=5);print([(m.get('Title'),m.get('Author'),(m.get('Summary') or '')[:40]) for m in (r.get('metadatas') or [])])"
```
Expect non-empty Author/Summary where you filled `story_json`.

---

## P0.2: Local Ollama Fallback JSON — FIXED (code)

### Done in code
- [x] Ollama JSON schema / format (`Local_grounded_facts_json_format`)
- [x] vLLM JSON mode path
- [x] Stronger local JSON-only prompt + chunk-id copy instruction
- [x] `salvage_grounded_facts_json()` (fences, prose, trailing commas, `<think>`, truncated arrays)
- [x] Cite-only-known-chunk-ids filter
- [x] One compact retry; ERROR log if still zero facts
- [x] `Grounded_facts_provider: "local"` skips HF entirely
- [x] Tests: `tests/test_local_facts_fallback.py` (24)

### Verify on your machine (Ollama only)
1. `ollama --version` (schema needs ~0.5+; else set `Local_grounded_facts_json_format: "json"`)
2. In `setup.yaml`: `Grounded_facts_provider: "local"` (optional: `Evaluation_mode: "local"`)
3. Probe:
```powershell
py -c "import sys;sys.path.insert(0,'src');from storyforge.config.config import load_config;from storyforge.rag.retrieval import retrieve_docs,_docs_to_chunks;from storyforge.rag.extraction import extract_grounded_facts;cfg=load_config();q='man explains why his room must stay refrigerated';raw,p=extract_grounded_facts(q,_docs_to_chunks(retrieve_docs(q,cfg)),cfg);print(len(p.facts));[print(f.source_chunk_ids,f.fact) for f in p.facts[:5]]"
```
Expect log like `Local grounded-facts attempt 1/2: status=ok kept=N` and `len(p.facts) > 0`.

---

## P0.3: No-Eval Accepts facts_count=0 — FIXED

### Done in code
- [x] No-eval path: `facts_count <= 0` → `RE_RETRIEVE` **before** completeness ACCEPT
- [x] Incomplete + facts → still REFINE; complete + facts → ACCEPT
- [x] Iteration records include `has_eval`
- [x] Exhausted loop with zero facts → `stop_reason=max_iterations_no_grounded_facts`
- [x] Tests extended in `tests/test_agentic_loop.py`

### Verify
```powershell
python -m pytest tests/test_agentic_loop.py -q
```
With local facts + agentic run: no iteration should ACCEPT with `facts_count: 0`.

---

## P0.4: Embedding Model Mismatch — STILL OPEN (demoted P1)

### Current State
- [x] BGE query-only prefix fixed in code (2026-09); re-ingest documented
- [ ] No collection metadata fingerprint / auto detect mismatch
- [ ] No loud fail if config model ≠ vectors in DB

### Still needed later
- [ ] Store embed model id (+ prefix convention) on collection at ingest
- [ ] Warn/fail on retrieval if mismatch

---

## Measurement (after HF credits renew)

Do **not** treat offline batches as the final Phase 2 truth run.

```powershell
# After HF renews — set Grounded_facts_provider: "hf" again
py scripts/measure_generation_length.py --mode fast --length long 2>&1 | tee Evaluation/post_p0_hf.log
```

Compare accept rate for iterations with `has_eval: true` vs false; compare HF vs local `facts_count` on the same queries.

---

## Sign-Off

- [x] P0.1–P0.3 understood, implemented, and regression-tested
- [x] Local pytest green: **163 passed** (includes `test_metadata_report` + api contracts)
- [x] `scripts/validate_chroma_metadata.py` added (offline follow-up)
- [ ] Ollama smoke probe for P0.2 (your machine)
- [ ] `reset_and_ingest` + validate / peek / retrieval_eval after metadata fill
- [ ] Clean HF measurement after credits renew
- [ ] P0.4 fingerprint still future work

**Status:** ✅ P0.1–P0.3 + offline docs/verify helper complete · ⏳ local Ollama / re-ingest verification · ⏳ HF re-measure next month
