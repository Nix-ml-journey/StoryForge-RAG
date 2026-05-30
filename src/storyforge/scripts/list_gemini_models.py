"""
List Gemini models for the configured API key.

CLI wrapper exists at `scripts/list_gemini_models.py`.
"""

from __future__ import annotations


def list_gemini_models(*, api_key: str) -> list[str]:
    from google import genai

    client = genai.Client(api_key=api_key)
    EXCLUDE_PREFIXES = ("embedding", "text-embedding", "gemini-embedding", "imagen", "veo", "aqa", "deep-research")
    EXCLUDE_SUBSTRINGS = (
        "-tts",
        "image-generation",
        "-image",
        "native-audio",
        "robotics",
        "computer-use",
        "nano-banana",
    )
    try:
        names: list[str] = []
        models = client.models.list()
        for model in models:
            base = model.name.replace("models/", "", 1).lower()
            if any(base.startswith(p) for p in EXCLUDE_PREFIXES):
                continue
            if any(s in base for s in EXCLUDE_SUBSTRINGS):
                continue
            names.append(model.name)
        return names
    finally:
        client.close()

