"""
List Gemini models for the configured API key.

Run:
  .venv/Scripts/python.exe scripts/list_gemini_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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


def main() -> None:
    from storyforge.config.config import load_config

    api_key = str(load_config().get("Gemini_api_key") or "").strip()
    if not api_key:
        raise SystemExit("Missing Gemini_api_key in setup.yaml (or set it via env overlay).")
    for name in list_gemini_models(api_key=api_key):
        print(name)


if __name__ == "__main__":
    main()
