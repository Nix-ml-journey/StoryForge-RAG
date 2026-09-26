"""
CLI wrapper for `storyforge.scripts.check_cuda_compatibility`.

Run:
  py scripts/check_cuda_compatibility.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyforge.scripts.check_cuda_compatibility import run_cuda_compatibility_check  # noqa: E402


if __name__ == "__main__":
    run_cuda_compatibility_check()
