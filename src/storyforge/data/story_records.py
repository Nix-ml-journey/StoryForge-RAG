from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from storyforge.config.config import load_config
from storyforge.vector_store.ingest_stories import _chunk_text as _chunk_text_impl


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def create_story_records(
    *,
    base_path: str | Path | None = None,
    story_input: str | None = None,
    out_dir: str | Path | None = None,
    template_path: str | Path | None = None,
    overwrite: bool = False,
    only: str = "",
    max_chars: int = 1800,
    overlap_chars: int = 200,
) -> dict[str, Any]:
    """
    Create editable `data/story_json/<Title>.json` from `data/stories/*.txt`.
    Uses the series template as the single base template.
    """
    cfg = load_config()
    base = Path(base_path or cfg.get("BASE_PATH") or ".").resolve()
    stories_dir = (base / (story_input or cfg.get("Story_input") or "data/stories")).resolve()
    out = (base / (out_dir or (Path("data") / "story_json"))).resolve()
    out.mkdir(parents=True, exist_ok=True)

    tpl_path = Path(template_path) if template_path else (base / "data" / "ingest" / "TEMPLATE__by_title_value__series.json")
    series_tpl = _load_json(tpl_path)
    if not series_tpl:
        raise FileNotFoundError(f"Missing series template: {tpl_path}")

    files = sorted([p for p in stories_dir.glob("*.txt") if p.is_file()], key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No .txt files found in {stories_dir}")

    created = 0
    skipped = 0
    for fp in files:
        title = fp.stem
        if only and title != only:
            continue
        out_path = out / f"{title}.json"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        raw = fp.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text_impl(raw, max_chars=max_chars, overlap_chars=overlap_chars)

        record: dict[str, Any] = dict(series_tpl)
        record.setdefault("Is_series", True)
        # Templates may include empty placeholder values; ensure required ids are populated.
        if not str(record.get("id") or "").strip():
            record["id"] = title
        record.setdefault("series", {"id": "", "series_name": ""})
        record.setdefault("volume", {"volume_id": "", "volume_name": "", "volume_number": 0})
        record.setdefault("meta", {"author": "", "title": ""})
        record.setdefault("chapter", {"chapter_id": "", "chapter_number": 1, "chapter_name": ""})
        record.setdefault("summary", "")

        record["raw_text"] = raw
        record["chunks"] = [
            {"chunk_id": f"{title}_chunk_{i}", "text": ch, "section": ""} for i, ch in enumerate(chunks, start=1)
        ]

        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        created += 1

    return {"created": created, "skipped_existing": skipped, "folder": str(out)}

