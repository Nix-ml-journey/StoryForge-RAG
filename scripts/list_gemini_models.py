from __future__ import annotations

"""
CLI wrapper for `storyforge.scripts.list_gemini_models`.

Run:
  py scripts/list_gemini_models.py
"""

from storyforge_config import load_config
from storyforge.scripts.list_gemini_models import list_gemini_models


def main() -> None:
    cfg = load_config()
    api_key = str(cfg.get("Gemini_api_key") or "").strip()
    if not api_key:
        raise SystemExit("Missing Gemini_api_key in setup.yaml (or set it via env overlay).")
    for name in list_gemini_models(api_key=api_key):
        print(name)


if __name__ == "__main__":
    main()
