from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def run_step1_prepare_and_enrich(
    *,
    root: str | Path | None = None,
    limit: int = 0,
    overwrite_summary: bool = False,
    overwrite_sections: bool = False,
    dry_run: bool = False,
    enable_tqdm: bool = True,
) -> None:
    """
    Step 1 convenience runner (script-backed):
    - Create/editable `data/story_json/*.json` from `data/stories/*.txt`
    - Enrich records with `summary` + per-chunk `section`

    This calls the existing CLI scripts via `runpy` so behavior stays identical,
    but is importable for orchestrator / API use.
    """
    base = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[3]

    prepare_path = base / "scripts" / "prepare_story_records.py"
    enrich_path = base / "scripts" / "enrich_story_records.py"

    if not prepare_path.exists():
        raise FileNotFoundError(f"Missing script: {prepare_path}")
    if not enrich_path.exists():
        raise FileNotFoundError(f"Missing script: {enrich_path}")

    # Avoid leaking caller argv into the child scripts.
    old_argv = sys.argv
    try:
        sys.argv = ["prepare_story_records.py"]
        runpy.run_path(str(prepare_path), run_name="__main__")
    finally:
        sys.argv = old_argv

    argv = ["enrich_story_records.py"]
    if limit and limit > 0:
        argv += ["--limit", str(limit)]
    if overwrite_summary:
        argv += ["--overwrite-summary"]
    if overwrite_sections:
        argv += ["--overwrite-sections"]
    if dry_run:
        argv += ["--dry-run"]

    if enable_tqdm:
        os.environ["STORYFORGE_TQDM"] = "1"

    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(enrich_path), run_name="__main__")
    finally:
        sys.argv = old_argv

