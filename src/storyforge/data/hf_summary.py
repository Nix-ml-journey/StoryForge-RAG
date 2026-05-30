from __future__ import annotations

import json
import os
from typing import Any

import requests


def get_hf_token(cfg: dict[str, Any]) -> str:
    token = (
        os.environ.get("STORYFORGE_HF_API_KEY", "").strip()
        or os.environ.get("HUGGINGFACE_API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or str(cfg.get("facehugging_api") or "").strip()
    )
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set STORYFORGE_HF_API_KEY (recommended) or HUGGINGFACE_API_KEY / HF_TOKEN, "
            "or put facehugging_api in setup.yaml."
        )
    return token


def summarize_long_text(
    hf_token: str,
    model: str,
    text: str,
    *,
    timeout_s: int = 120,
    max_chars: int = 8000,
) -> str:
    """Summarize text via HF Inference API (story_json enrichment)."""
    text = (text or "").strip()
    if not text:
        return ""

    payload = {"inputs": text[:max_chars]}
    headers = {"Authorization": f"Bearer {hf_token}"}
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"HF inference error: status={r.status_code} body={r.text[:500]}")

    data: Any = r.json()
    # API may return list[dict], dict, or str
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("summary_text") or data[0].get("generated_text") or "").strip()
    if isinstance(data, dict):
        return str(data.get("summary_text") or data.get("generated_text") or "").strip()
    if isinstance(data, str):
        return data.strip()

    try:
        return json.dumps(data)[:1000]
    except Exception:
        return ""

