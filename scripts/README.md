# Utility scripts

Optional CLI helpers. Run from the **repo root** unless noted.

## Story pipeline (Step 1 → ingest)

| Script | Purpose |
|--------|---------|
| `step1_prepare_and_enrich.py` | `.txt` → `data/story_json/*.json` |
| `enrich_story_records.py` | Re-run enrichment on existing records |
| `records_to_ingest_manifest.py` | `story_json` → `data/ingest/ingest_manifest.jsonl` |
| `ingest_manifest.py` | Upsert manifest into Chroma |
| `reset_and_ingest.py` | Wipe Chroma + re-ingest from `data/stories/` |

## Diagnostics

| Script | Purpose |
|--------|---------|
| `check_cuda_compatibility.py` | NVIDIA / PyTorch CUDA check |
| `list_gemini_models.py` | List Gemini models for your API key |
| `peek_vector_store.py` | Inspect Chroma collection contents |
| `retrieval_eval.py` | Measure retrieval top-k accuracy against fixture cases |
| `test_generation.py` | HTTP smoke tests (server must be running) |

## Legacy book pipeline

| Script | Purpose |
|--------|---------|
| `check_merged_data.py` | Validate `Stories/`, `Metadata/`, `Data_Merged/` alignment |

## Text prep

| Script | Purpose |
|--------|---------|
| `merge_paragraphs.py` | Merge `---`-separated blocks into single lines |

Example:

```powershell
py scripts/reset_and_ingest.py
py scripts/test_generation.py --test 1
```
