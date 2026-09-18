# Data layout

After extract, **do not ingest the raw dump**. Clean it, then follow [`docs/DATA_PREP.md`](../docs/DATA_PREP.md).

## After extract (short checklist)

Extracted files land in `data/raw_extracted/` (scratch). Ingest reads `data/stories/`.

1. **Split** — one story per `.txt`. Filename stem = Chroma Title (`Lovecraft__The_Call_of_Cthulhu.txt`).
2. **Strip** — Gutenberg/Archive licenses, TOC, page numbers, running headers, “end of ebook”.
3. **Paragraphs** — blank line between paragraphs so chunking does not cut mid-sentence.
4. **Copy** cleaned files into `data/stories/`.
5. **Prepare** — `py scripts/step1_prepare_and_enrich.py`
6. **Open JSON** in `data/story_json/` — fill `meta.author` / `meta.title`, set `Is_series` only if true, fix empty or garbage chunks, rewrite a bad summary.
7. **Ingest** — `py scripts/records_to_ingest_manifest.py` then `py scripts/ingest_manifest.py`

If the `.txt` is junk, later generation will be junk. The model cannot un-glue two books or ignore a table of contents that you left in.

## Public (committed)

Only the `sample/` demo folders are in git — enough to run tests and show the pipeline shape:

```
data/
├── stories/sample/       demo_village.txt + README
├── story_json/sample/    demo_village.json + README
└── ingest/
    ├── sample/           2-line demo manifest + README
    ├── ingest_metadata_template.json
    └── TEMPLATE__*.json
```

## Private (local only, gitignored)

| Path | Contents |
|------|----------|
| `data/raw_extracted/` | Book extraction scratch — clean here first |
| `data/stories/*.txt` | Clean stories, one tale per file |
| `data/story_json/*.json` | Editable records (text, chunks, metadata) |
| `data/ingest/ingest_manifest.jsonl` | Full ingest manifest (rebuilt locally) |
| `data/chroma_db/` | Chroma vector database |
| `data/outputs/` | Generated / evaluated story runs |

Rebuild locally (only after the checklist above):

```powershell
py scripts/step1_prepare_and_enrich.py
py scripts/records_to_ingest_manifest.py
py scripts/reset_and_ingest.py
```

Portfolio and job-search write-ups live outside this repo in `../StoryForge-portfolio/`.
