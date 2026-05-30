"""
DEPRECATED -- this file is no longer the entry point.

The canonical entry point is `main.py` at the project root.

Run the API with:
    python main.py          (Windows)
    python3 main.py         (Linux / macOS)

This file is kept so any tooling that still references `app/main.py`
gets a clear error message rather than a confusing ImportError.
"""

raise RuntimeError(
    "app/main.py is deprecated. "
    "Run the API from the project root: `python main.py`"
)
