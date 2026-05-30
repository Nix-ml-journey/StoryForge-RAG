from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is on sys.path so `import storyforge...` works in tests.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
