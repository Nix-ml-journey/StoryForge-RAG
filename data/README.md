# Data layout

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
| `data/stories/*.txt` | Your full story text files |
| `data/story_json/*.json` | Enriched records (Step 1 output) |
| `data/ingest/ingest_manifest.jsonl` | Full ingest manifest (rebuilt locally) |
| `data/chroma_db/` | Chroma vector database |
| `data/outputs/` | Generated / evaluated story runs |
| `data/raw_extracted/` | Book extraction scratch |

Rebuild locally:

1. Add `.txt` files under `data/stories/`
2. `py scripts/step1_prepare_and_enrich.py`
3. `py scripts/records_to_ingest_manifest.py`
4. `py scripts/reset_and_ingest.py`

Portfolio and job-search write-ups live outside this repo in `../StoryForge-portfolio/`.
