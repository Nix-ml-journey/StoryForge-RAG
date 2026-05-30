# Utility Scripts

These scripts are optional local helpers. They are not required for the lightweight CI suite.

## `list_gemini_models.py`

Lists available Gemini text models for the configured API key.

```bash
python scripts/list_gemini_models.py
```

Requires a Gemini API key in `setup.yaml`, `.env`, or environment variables.

## `check_cuda_compatibility.py`

Checks NVIDIA driver visibility and PyTorch CUDA compatibility.

```bash
python scripts/check_cuda_compatibility.py
```

Useful when tuning local generation on newer GPUs.

## `check_merged_data.py`

Checks stem alignment and content consistency across `Stories/`, `Metadata/`, and `Data_Merged/`.

```bash
python scripts/check_merged_data.py
python scripts/check_merged_data.py --verbose
```

Useful after adding or regenerating local story data.

## `reset_and_ingest.py`

Resets the Chroma collection and re-ingests stories (e.g. after switching embedding models).

```bash
python scripts/reset_and_ingest.py
```

## `test_generation.py`

Manual smoke tests against the running API (`python main.py` must be up).

```bash
python scripts/test_generation.py
python scripts/test_generation.py --test 1
```

## `generate_portfolio_pdf.py`

Builds `docs/portfolio/StoryForge_RAG_Portfolio.pdf` from the markdown source.

```bash
pip install fpdf2
python scripts/generate_portfolio_pdf.py
```

## `merge_paragraphs.py`

One-off helper to merge story `.txt` blocks separated by `---` lines.

