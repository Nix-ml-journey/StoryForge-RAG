# StoryForge-RAG Critical Issues Checklist

**Original audit:** 2026-09-24  
**Verification update:** 2026-09-24 (local / no HF credits)

**Suite status:** `163 passed` in project `.venv` after P0.1–P0.3 + offline follow-up (`validate_chroma_metadata` + Py3.10-safe newline read).
**Machine verification (2026-09-24):** rebuild ingest `From story_json: 72 | stale: 0 | no story_json: 0` → 72 files / 2138 chunks, `PROBLEMS: none`; Author 93.6% · Display_title 93.6% · Summary 6.4% · section 45.8%; retrieval_eval top1 0.80 / top3 0.833 / fact_coverage 0.733.

> **Run every script with the venv:** `.\.venv\Scripts\python.exe scripts\<name>.py` (or activate `.venv` first). Bare `py scripts\retrieval_eval.py` uses system Python and fails with `ModuleNotFoundError: langchain_chroma`. Commands below already use the venv path.

---

## Status board

| ID | Issue | Status | Notes |
|----|--------|--------|-------|
| **P0.1** | story_json metadata ignored by rebuild ingest | **FIXED + VERIFIED ON MACHINE** | 72 / 0 / 0 story_json ingest, 2138 chunks, Author 93.6% in Chroma. Remaining gaps are content (Firestone author convention, classic summaries), not code |
| **P0.2** | Local Ollama facts JSON brittle | **FIXED (code)** | Schema/format, salvage parser, citation check, retry, `Grounded_facts_provider: "local"`. Still needs one Ollama smoke probe on your machine |
| **P0.3** | No-eval ACCEPT with `facts_count=0` | **FIXED** | `decide_action` checks facts first; iterations record `has_eval`; stop reason `max_iterations_no_grounded_facts` |
| **P0.4** | Embed prefix / model mismatch detection | **OPEN (P1)** | Re-ingest already done after BGE fix; still no automatic collection fingerprint |
| Offline follow-up | Docs sync + `validate_chroma_metadata.py` | **VERIFIED** | 163 passed; validate script run on the real 2138-chunk collection |

---

## P0.1: story_json Workflow ↔ Ingest — FIXED + VERIFIED ON MACHINE

### Correction to original audit
`scripts/ingest_manifest.py` already could ingest from the manifest built from `story_json`. The real gap was **`reset_and_ingest.py` → `ingest_stories_dir()`**, which hardcoded empty Author/Summary and ignored reviewed chunks.

### Done in code
- [x] `ingest_stories_dir()` loads `data/story_json/<Title>.json`
- [x] Writes Author / Summary / Display_title / Is_series (Title = filename stem)
- [x] Uses reviewed chunks + section tags when `raw_text` matches `.txt`
- [x] Stale records: `.txt` chunks + story-level metadata only + warning
- [x] `records_to_manifest.py` aligned on same metadata fields
- [x] Tests: `tests/test_ingest_story_json_metadata.py` (8)
- [x] Windows CRLF: `_norm_text` + `_read_text_keep_newlines()` (`open(..., newline="")`, Python 3.10+ safe — `Path.read_text(newline=)` is 3.13-only) + test writes via `write_bytes`

### Verified on machine (2026-09-24)
- [x] `reset_and_ingest.py` → `From story_json: 72 | stale: 0 | no story_json: 0`, Files 72, Chunks 2138
- [x] `validate_chroma_metadata.py` → `PROBLEMS: none`; Author 93.6%, Display_title 93.6%, Summary 6.4%, section 45.8%
- [x] `retrieval_eval.py` (venv) → top1 0.80 / top3 0.833 / fact_coverage 0.733 (see below)
- Ops tip, not a bug: `WinError 32` deleting `data/chroma_db` (file locked by a running API / Python process); the script falls back to a Chroma API collection reset and the rebuild still succeeds. Stop `python main.py` first if you want the folder removed.

Re-run any time:
```powershell
.\.venv\Scripts\python.exe scripts\reset_and_ingest.py
.\.venv\Scripts\python.exe scripts\validate_chroma_metadata.py
.\.venv\Scripts\python.exe scripts\retrieval_eval.py --cases tests/fixtures/retrieval_eval_cases.example.json --k 3
```

### retrieval_eval after the story_json rebuild

| | top1 | top3 | fact_coverage |
|---|---|---|---|
| Phase 1 baseline (rerank-before-diversity) | 0.80 | 0.90 | 0.77 |
| After rebuild from reviewed story_json chunks | **0.80** | **0.833** | **0.733** |

top1 held. top3 / coverage dipped because the collection now embeds the reviewed story_json chunks instead of a fresh `.txt` re-chunk — **expected shift, not a regression emergency**. On 30 cases, top3 0.90 → 0.833 is 2 cases. If you want them back, compare the case-by-case report in `Evaluation/retrieval_eval_report.json` and treat it as corpus/chunk work, not a retrieval-knob change (Phase 1 tuning stays stopped).

### Remaining metadata work (content, not code — none of it is P0)

| Gap | Where | Action | Priority |
|---|---|---|---|
| Author / Display_title empty (6.4% of chunks) | 37 Firestone Idle wiki stories (Amun … Zoruk) | Firestone convention below; then rebuild | Optional now, offline |
| Template placeholders | same 37: `id: "id_01"` (shared), `Is_series: true` with empty `series_name` | Fixed by the same recipe | Optional now (hygiene; retrieval/eval don't read `id`) |
| Summary empty (93.6% of chunks) | 35 classics (Kafka, Lovecraft, Frankenstein, Jekyll, Wells, Doyle) | Wait for HF enrich, hand-write, or leave empty — Summary isn't used by retrieval or generation | Waits for HF / optional |
| section tags 45.8% | 16 stories with no section tags at all | Hand-tag or re-run enrich later; `push_section_metadata.py` can push tags without re-embedding | P2 |

### Firestone metadata recipe (optional, offline, ~2 min)

The 37 stories without Author / Display_title (Amun, Anzo, Arvie, Asmondai, Astrid, Belien, …, Zoruk) are **Firestone Idle RPG wiki character/story pages**, not books with an unknown novelist. Don't look up authors; use one consistent convention:

| story_json field | Set to | Lands in Chroma as |
|---|---|---|
| `meta.author` | `"Firestone Idle"` (same string for all 37) | `Author` |
| `meta.title` | character / page name (= filename stem, `_` → space) | `Display_title` |
| `id` | filename stem (replaces template placeholder `id_01`, currently shared by all 37) | `id` |
| `Is_series` | `false` when `series.series_name` is empty (all 37 today: `true` + empty name) | `Is_series` |
| `summary` | leave as is (these 37 already have one) | `Summary` |

Only `meta` / `id` / `Is_series` change — `raw_text` and `chunks` are untouched, so ingest still treats all 72 records as fresh story_json (dry-run verified on a copy: 37 updated, re-run updates 0, plan stays 72 × `story_json`).

```powershell
# from the repo root
$fill = @'
import json, pathlib, sys
APPLY = "--apply" in sys.argv
AUTHOR = "Firestone Idle"
changed = 0
for fp in sorted(pathlib.Path("data/story_json").glob("*.json")):
    rec = json.loads(fp.read_text(encoding="utf-8"))
    meta = rec.setdefault("meta", {})
    stem = fp.stem
    if str(meta.get("author") or "").strip() or "__" in stem:
        continue  # already has an author, or an Author__Title book stem
    meta["author"] = AUTHOR
    meta["title"] = str(meta.get("title") or "").strip() or stem.replace("_", " ")
    if str(rec.get("id") or "") in ("", "id_01"):
        rec["id"] = stem
    if not str((rec.get("series") or {}).get("series_name") or "").strip():
        rec["Is_series"] = False
    changed += 1
    print(("SET " if APPLY else "WOULD SET ") + f"{stem}: title={meta['title']!r} id={rec['id']!r}")
    if APPLY:
        fp.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(("Updated " if APPLY else "Would update ") + f"{changed} record(s).")
'@
$fill | .\.venv\Scripts\python.exe -              # 1. dry run: expect "Would update 37"
$fill | .\.venv\Scripts\python.exe - --apply      # 2. write the 37 JSON files
Get-Content data\story_json\Amun.json -TotalCount 25   # spot-check: author / title / id / Is_series

# 3. rebuild (push_section_metadata.py does NOT refresh top-level Author/Display_title)
.\.venv\Scripts\python.exe scripts\reset_and_ingest.py       # expect 72 / 0 / 0 again
.\.venv\Scripts\python.exe scripts\validate_chroma_metadata.py
```

**Expected after:** Author **100%**, Display_title **100%**, Summary still **6.4%** (classics have none), section unchanged **45.8%**, `Stories without Author: none`. Retrieval text/vectors are identical (only metadata changed), so retrieval_eval numbers should not move.

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
.\.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');from storyforge.config.config import load_config;from storyforge.rag.retrieval import retrieve_docs,_docs_to_chunks;from storyforge.rag.extraction import extract_grounded_facts;cfg=load_config();q='man explains why his room must stay refrigerated';raw,p=extract_grounded_facts(q,_docs_to_chunks(retrieve_docs(q,cfg)),cfg);print(len(p.facts));[print(f.source_chunk_ids,f.fact) for f in p.facts[:5]]"
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
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_loop.py -q
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
.\.venv\Scripts\python.exe scripts\measure_generation_length.py --mode fast --length long 2>&1 | tee Evaluation/post_p0_hf.log
```

Compare accept rate for iterations with `has_eval: true` vs false; compare HF vs local `facts_count` on the same queries.

---

## Sign-Off

**Done**
- [x] P0.1–P0.3 implemented and regression-tested
- [x] Local pytest green: **163 passed** (includes `test_metadata_report` + api contracts)
- [x] Rebuild ingest on the real corpus: 72 / 0 / 0, 2138 chunks, `PROBLEMS: none`
- [x] `validate_chroma_metadata.py` run on the real collection (Author 93.6%)
- [x] `retrieval_eval.py` via venv after rebuild (0.80 / 0.833 / 0.733)

**Optional now (offline)**
- [ ] Firestone recipe: fill `meta.author` / `meta.title` (+ `id`, `Is_series`) for the 37 wiki stories → `reset_and_ingest.py` → validate (expect Author 100%)
- [ ] P0.2 Ollama smoke probe with `Grounded_facts_provider: "local"` — record `status=ok kept=N`
- [ ] (P2) section tags for the 16 untagged stories

**Waits for HF credits**
- [ ] Clean `measure_generation_length.py` truth run (split accept rate by `has_eval`)
- [ ] Summary enrich for the 35 classics (or leave empty offline)
- [ ] HF vs local facts comparison on the same queries

**Still open P1**
- [ ] Embedding fingerprint on the collection (old P0.4)

**Status:** ✅ P0.1–P0.3 done, P0.1 verified on machine · 🟡 optional Firestone metadata fill + P0.2 smoke probe · ⏳ HF re-measure next month · P1: embed fingerprint
