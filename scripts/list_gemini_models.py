from __future__ import annotations

"""
CLI wrapper for `storyforge.scripts.list_gemini_models`.

Run:
  py scripts/list_gemini_models.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyforge.config.config import load_config  # noqa: E402
from storyforge.scripts.list_gemini_models import list_gemini_models  # noqa: E402


def main() -> None:
    cfg = load_config()
    api_key = str(cfg.get("Gemini_api_key") or "").strip()
    if not api_key:
        raise SystemExit("Missing Gemini_api_key in setup.yaml (or set it via env overlay).")
    for name in list_gemini_models(api_key=api_key):
        print(name)


if __name__ == "__main__":
    main()
