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
