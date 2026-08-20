# Utility scripts

Optional CLI helpers. Run from the **repo root** unless noted.

## Story pipeline (Step 1 → ingest)

| Script | Purpose |
|--------|---------|
| `step1_prepare_and_enrich.py` | `.txt` → `data/story_json/*.json` |
| `prepare_story_records.py` | Build story_json without enrichment |
| `enrich_story_records.py` | Re-run enrichment on existing records |
| `records_to_ingest_manifest.py` | `story_json` → `data/ingest/ingest_manifest.jsonl` |
| `ingest_manifest.py` | Upsert manifest into Chroma (explicit BGE embeddings) |
| `reset_and_ingest.py` | Wipe Chroma + re-ingest from `data/stories/` |

Typical full ingest:

```powershell
py scripts/step1_prepare_and_enrich.py
py scripts/records_to_ingest_manifest.py
py scripts/ingest_manifest.py
```

## Chroma maintenance

| Script | Purpose |
|--------|---------|
| `refresh_chunk_embeddings.py` | Re-embed chunks after editing `chunks[].text` in story_json (keeps metadata) |
| `push_section_metadata.py` | Push updated section/metadata tags to Chroma without re-embedding |

Examples:

```powershell
py scripts/refresh_chunk_embeddings.py --glob "Lovecraft__*"
py scripts/push_section_metadata.py --glob "Jekyll_and_Hyde__*"
```

## Diagnostics

| Script | Purpose |
|--------|---------|
| `check_cuda_compatibility.py` | NVIDIA / PyTorch CUDA check |
| `list_gemini_models.py` | List Gemini models for your API key |
| `peek_vector_store.py` | Inspect Chroma collection contents |
| `retrieval_eval.py` | Measure retrieval top-k accuracy against fixture cases |
| `test_generation.py` | HTTP smoke tests (server must be running) |
| `debug_hf_grounded_facts_mode.py` | Probe Step 2 HF JSON mode vs fallback |

## Text prep

| Script | Purpose |
|--------|---------|
| `merge_paragraphs.py` | Merge `---`-separated blocks into single lines |

## Examples

```powershell
# Full reset + ingest
py scripts/reset_and_ingest.py

# Smoke test generation (server must be running)
py scripts/test_generation.py --test 1

# Check HF grounded-facts JSON mode
py scripts/debug_hf_grounded_facts_mode.py

# Retrieval quality report
py scripts/retrieval_eval.py --cases tests/fixtures/retrieval_eval_cases.example.json --k 3
```
