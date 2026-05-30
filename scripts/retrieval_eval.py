"""
CLI wrapper for `storyforge.evaluation.retrieval_eval`.

Run from repo root:
  py scripts/retrieval_eval.py --cases tests/fixtures/retrieval_eval_cases.example.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyforge.evaluation.retrieval_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
