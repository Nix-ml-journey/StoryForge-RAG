# Data preparation after extract

Extraction only turns a PDF or EPUB into a messy `.txt`. **That file is not ready for RAG yet.** Retrieval and generation quality are decided here, by hand, before anything goes into Chroma.

Read this when you have just run extract and want the later story output to stay grounded.

---

## The path (one picture)

```text
PDF / EPUB
    ↓  extract
data/raw_extracted/*.txt          ← raw dump. Do not ingest this.
    ↓  YOU clean, split, name
data/stories/*.txt                ← one story per file
    ↓  prepare + enrich
data/story_json/*.json            ← YOU review metadata + chunks
    ↓  manifest + ingest
Chroma                            ← only then generate
```

Scripts do not magically fix a bad book dump. If the `.txt` still has the table of contents, two stories glued together, or broken sentences, Step 2 will extract junk facts and Step 3 will write junk stories.

---

## Folders

| Folder | What it is | You edit it? |
|--------|------------|--------------|
| `Downloaded_Books/` (or `Downloaded_rawbook_dir`) | PDFs / EPUBs from Archive.org | No |
| `data/raw_extracted/` | Extracted text. Headers, page numbers, whole books | **Yes — this is your workbench** |
| `data/stories/` | Clean stories, one `.txt` per tale | **Yes — this is what ingest reads** |
| `data/story_json/` | JSON records: text + chunks + author/title/summary | **Yes — fill blanks, fix tags** |
| `data/ingest/ingest_manifest.jsonl` | One line per chunk for Chroma | Rebuild with a script; don't hand-edit |
| `data/chroma_db/` | Vector index | Never edit by hand |

`data/*/sample/` is the tiny public demo. Your real corpus is gitignored.

---

## Step A — Extract (already done)

```powershell
# API: POST /orchestration/run_step  { "step": "0_fetch_and_extract", "title": "..." }
# or from code: Orchestrator().extract_text()
```

Output: `data/raw_extracted/<book>.txt`.

Treat this as **scratch**. Extraction only:

- pulls PDF/EPUB text
- joins some lines into paragraphs
- does **not** split stories, drop Gutenberg headers, or fix OCR

---

## Step B — Clean the extracted text (the important part)

Open the file in `data/raw_extracted/`. Do this **before** you copy anything into `data/stories/`.

### 1. Split one book into one-story files

A collection (Grimm, Lovecraft, Kafka) must become **one `.txt` per story**.

- Filename stem becomes the Chroma **Title**. Use a stable name:

```text
Lovecraft__The_Call_of_Cthulhu.txt
Kafka__The_Metamorphosis.txt
The_Hound_of_the_Baskervilles.txt
```

- Do not leave `Complete_Works_of_Poe.txt` as one file. Retrieval will mix tales.

### 2. Delete everything that is not the story

Remove:

- Project Gutenberg / Archive.org license blocks
- title pages, dedications, prefaces you do not want retrieved
- table of contents
- page numbers sitting on their own line (`12`, `xiv`)
- running headers (`THE CALL OF CTHULHU  47`)
- illustrations captions, footnotes that are only “see p. 12”
- “End of the Project Gutenberg…” footers
- chapter lists that are not the chapter body

Keep:

- the actual narrative
- character names, places, and plot (this is what facts are built from)

### 3. Fix the text so chunking works

The chunker splits on **blank lines** (paragraphs), then packs them to ~1800 characters with ~200 overlap.

| Do | Don't |
|----|--------|
| One blank line between paragraphs | One giant wall of text (chunks cut mid-sentence) |
| Real sentences ending in `. ? !` | OCR line-wrap leftovers (`the for-\nest`) |
| UTF-8, readable quotes and dashes | Binary garbage, `????`, boxed drawing characters |
| Dialogue that still makes sense | Headers glued onto the previous sentence |

Optional helper if your file uses `---` between blocks:

```powershell
py scripts/merge_paragraphs.py "data/raw_extracted\My_Story.txt"
```

That merges each `---`-separated block onto one line. It does **not** remove Gutenberg junk.

### 4. Sanity-check the file

Open the top, the middle, and the end.

- First paragraph is the real opening, not “CONTENTS”.
- A search for a character name finds story sentences, not the index.
- The file ends with the story ending, not a license.
- Length feels right (a short tale is thousands of words, not 80).

When it looks like a story you would give a human to read, copy it:

```text
data/raw_extracted/cleaned pieces  →  data/stories/<Title>.txt
```

---

## Step C — Prepare JSON, then check it

```powershell
py scripts/step1_prepare_and_enrich.py
```

This:

1. Reads every `data/stories/*.txt`
2. Writes `data/story_json/<Title>.json` (skips files that already exist unless you overwrite)
3. Tries to fill `summary` (Hugging Face) and per-chunk `section` tags (Ollama / local model)

**Enrichment is a draft.** You still open the JSON.

### What a good record looks like

See `data/story_json/sample/demo_village.json`. Minimum you should fill:

| Field | Why it matters |
|-------|----------------|
| `meta.author` / `meta.title` | Shows up in retrieval metadata; empty titles make eval look broken |
| `id` | Stable story id (defaults to the filename) |
| `Is_series` | `true` only if it is really a chapter in a named series |
| `series` / `volume` / `chapter` | Fill for series; leave empty for standalone |
| `summary` | Helps you, and some retrieval text; rewrite if the model invented a plot |
| `chunks[].text` | This is what gets embedded. Fix OCR here if you missed it in the `.txt` |
| `chunks[].section` | Tags like `setup`, `climax`, `resolution`. Wrong tags are better than empty, but a whole file tagged `setup` is a bad labeler run |

Allowed section tags: `setup`, `inciting_incident`, `rising_action`, `confrontation`, `twist`, `climax`, `fallout`, `resolution`, `epilogue`, `setting`, `negotiation`.

### JSON checklist (do this before ingest)

- [ ] Filename matches the story (`Kafka__The_Metamorphosis.json`)
- [ ] `meta.title` is the real title, not blank
- [ ] `Is_series` is false for a one-off tale
- [ ] `raw_text` is the same story you cleaned (not a leftover dump)
- [ ] Every chunk has non-empty `text`
- [ ] No chunk is only a page number or “CHAPTER III”
- [ ] Section tags vary across the file (not the same word 20 times)
- [ ] Summary names the actual characters/places, or you wrote it yourself

Re-run only what you need:

```powershell
# Rebuild JSON from .txt (overwrites existing records)
py scripts/prepare_story_records.py --overwrite --only "Lovecraft__The_Call_of_Cthulhu"

# Re-label / re-summarize without recreating files
py scripts/enrich_story_records.py --overwrite-summary --overwrite-sections
```

---

## Step D — Ingest only after the JSON looks right

```powershell
py scripts/records_to_ingest_manifest.py
py scripts/ingest_manifest.py
```

Full wipe + re-ingest from `data/stories/` (use when the corpus changed a lot):

```powershell
py scripts/reset_and_ingest.py
```

After small text edits in existing JSON (no new files):

```powershell
py scripts/refresh_chunk_embeddings.py --glob "Lovecraft__*"
```

After you only changed section tags:

```powershell
py scripts/push_section_metadata.py --glob "Lovecraft__*"
```

If you change the embedding model, you must re-ingest. Old vectors will not match new queries.

**2026-09: BGE passage-prefix fix -- re-ingest required.** Ingest previously
prefixed passages with `"Represent this passage for retrieval: "` before
embedding them, in addition to the query-side prefix applied at search time.
That is not BGE's documented recipe: `BAAI/bge-base-en-v1.5` only prefixes
the *query* side for asymmetric retrieval; passages get no instruction
prefix. `storyforge/vector_store/ingest_stories.py::_embed_chunks` no longer
adds that prefix. The embedding model and its dimensionality are unchanged,
so Chroma will not reject a mismatched batch the way it does for a real model
swap -- it will happily keep serving the *old*, wrongly-prefixed vectors
side by side with correctly-embedded new ones, silently degrading relevance
for anything not re-ingested. Any collection ingested before this fix must be
rebuilt from scratch:

```powershell
py scripts/reset_and_ingest.py
```

`refresh_chunk_embeddings.py` and `ingest_manifest.py` pick up the fix too
(they share `_embed_chunks`), but only for the records they touch -- they do
not repair chunks already sitting in Chroma with the old prefix, so a full
`reset_and_ingest.py` is the only way to guarantee a clean collection.

---

## How you know the prep was good

1. **Peek a few chunks**

```powershell
py scripts/peek_vector_store.py
```

You should see real story sentences, with the Title you chose — not “CONTENTS” or license text.

2. **Ask a question only that story can answer**

Query Chroma (API `POST /vector_store/query` or generate with `debug: true`) using a distinctive name or event. Top hits should be **that** title.

3. **Optional retrieval report**

```powershell
py scripts/retrieval_eval.py --cases tests/fixtures/retrieval_eval_cases.example.json --k 3
```

That scores title rank and whether expected words appear in the chunks. It cannot save you if the `.txt` was never cleaned.

---

## Typical failures (prep, not the model)

| Symptom later | Likely prep mistake |
|---------------|---------------------|
| Generated story mixes two books | Two tales in one `.txt`, or Title is a collected-works filename |
| Facts mention Gutenberg / “chapter list” | Header/footer not stripped |
| Empty or tiny generated story | Chunks are whitespace, or extract failed and you ingested it anyway |
| Wrong story retrieved for a famous query | Filename/Title does not match the tale; corpus too mixed |
| Every chunk tagged `setup` | Enrichment repeated one label; fix tags by hand |
| Eval `fact_coverage` always 0 | Missing `Title` in metadata (usually a bad ingest, not a missing metric) |

---

## Commands in order (copy/paste)

```powershell
# 1. You already extracted → files sit in data/raw_extracted/

# 2. Clean and split by hand, then copy into data/stories/
#    (one story = one .txt = one Title)

# 3. Build + enrich JSON
py scripts/step1_prepare_and_enrich.py

# 4. Open data/story_json/*.json and fill author/title; fix bad chunks/tags

# 5. Ingest
py scripts/records_to_ingest_manifest.py
py scripts/ingest_manifest.py
```

Same steps as orchestration `1_prepare_and_enrich_story_json` → `3_ingest_stories`. Do not skip the manual pass between extract and Step 1.
