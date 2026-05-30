from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from storyforge.config.config import load_config


def _as_chroma_metadata(md: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (md or {}).items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float, str)):
            out[str(k)] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[str(k)] = ", ".join([x for x in v if (x or "").strip()])
        else:
            out[str(k)] = str(v)
    return out


def _dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def records_to_ingest_manifest(*, base_path: str | Path | None = None) -> dict[str, Any]:
    """
    Convert `data/story_json/*.json` into `data/ingest/ingest_manifest.jsonl` (one chunk per line).
    """
    cfg = load_config()
    base = Path(base_path or cfg.get("BASE_PATH") or ".").resolve()
    records_dir = (base / "data" / "story_json").resolve()
    out_dir = (base / "data" / "ingest").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ingest_manifest.jsonl"

    files = sorted([p for p in records_dir.glob("*.json") if p.is_file()], key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No record JSON files found in {records_dir}")

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for fp in files:
            rec = json.loads(fp.read_text(encoding="utf-8"))
            if not isinstance(rec, dict):
                continue

            title = fp.stem
            is_series = bool(rec.get("Is_series", False))
            story_id = str(rec.get("id") or title)

            meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
            chapter = rec.get("chapter") if isinstance(rec.get("chapter"), dict) else {}
            series = rec.get("series") if isinstance(rec.get("series"), dict) else {}
            volume = rec.get("volume") if isinstance(rec.get("volume"), dict) else {}

            chunks = rec.get("chunks") if isinstance(rec.get("chunks"), list) else []
            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                chunk_id = str(ch.get("chunk_id") or "")
                text = str(ch.get("text") or "")
                section = str(ch.get("section") or "")
                if not chunk_id:
                    continue

                md: dict[str, Any] = {
                    "Title": title,
                    "chunk_id": chunk_id,
                    "id": story_id,
                    "Is_series": is_series,
                    "query_type": "content",
                    "section": section,
                    "meta_json": _dumps_json(meta),
                    "chapter_json": _dumps_json(chapter),
                }
                if is_series:
                    md["series_json"] = _dumps_json(series)
                    md["volume_json"] = _dumps_json(volume)

                f.write(
                    json.dumps({"id": chunk_id, "text": text, "metadata": _as_chroma_metadata(md)}, ensure_ascii=False)
                    + "\n"
                )
                written += 1

    return {"written": written, "out_path": str(out_path), "records_dir": str(records_dir)}

