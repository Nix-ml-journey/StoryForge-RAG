from __future__ import annotations

from typing import Any

from storyforge.data.enrich_records import enrich_story_records
from storyforge.data.story_records import create_story_records


def run_step1_prepare_and_enrich(
    *,
    limit: int = 0,
    overwrite_summary: bool = False,
    overwrite_sections: bool = False,
    dry_run: bool = False,
    enable_tqdm: bool = False,
) -> dict[str, Any]:
    """
    Step 1 runner (importable; the `scripts/` CLI wraps this):
    - Create missing `data/story_json/*.json` from `data/stories/*.txt`
    - Enrich records with `summary` + per-chunk `section`
    """
    created = create_story_records()
    enriched = enrich_story_records(
        limit=limit,
        overwrite_summary=overwrite_summary,
        overwrite_sections=overwrite_sections,
        dry_run=dry_run,
        enable_tqdm=enable_tqdm,
    )
    return {"prepare": created, "enrich": enriched}
