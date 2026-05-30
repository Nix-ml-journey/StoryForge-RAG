import json
import sys
from pathlib import Path
from typing import Any


def _chunk_text(text: str, *, max_chars: int = 1800, overlap_chars: int = 200) -> list[str]:
    sys.path.insert(0, "src")  # same chunker as ingest_stories_dir
    from storyforge.vector_store.ingest_stories import _chunk_text as _chunk

    return _chunk(text, max_chars=max_chars, overlap_chars=overlap_chars)


def _as_chroma_metadata(md: dict[str, Any]) -> dict[str, Any]:
    """Flatten metadata for Chroma: scalars as-is, list[str] joined, else str()."""
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


def _is_truthy_series(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y"}


def _dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_template(*, path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.exists():
        return (
            {
                "Author": "",
                "Summary": "",
                "query_type": "content",
                "Is_series": False,
            },
            {},
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ({"Is_series": False, "query_type": "content"}, {})

    # Template: {defaults, by_title} or legacy single object with optional by_title
    if "defaults" in data or "by_title" in data:
        defaults = data.get("defaults") or {}
        by_title = data.get("by_title") or {}
    else:
        strip_keys = {"_help", "_examples", "examples", "by_title"}
        defaults = {k: v for k, v in data.items() if k not in strip_keys}
        by_title = data.get("by_title") or {} if isinstance(data.get("by_title"), dict) else {}
    if not isinstance(defaults, dict):
        defaults = {}
    if not isinstance(by_title, dict):
        by_title = {}
    safe_defaults: dict[str, Any] = {str(k): v for k, v in defaults.items() if not str(k).startswith("_")}
    safe_by_title: dict[str, dict[str, Any]] = {}
    for title_key, v in by_title.items():
        if str(title_key).startswith("_"):
            continue  # doc keys like _help
        if isinstance(v, dict):
            safe_by_title[str(title_key)] = {str(k2): v2 for k2, v2 in v.items() if not str(k2).startswith("_")}
    return safe_defaults, safe_by_title


def _merge_base_metadata(
    *,
    title: str,
    chunk_id: str,
    standalone_defaults: dict[str, Any],
    series_defaults: dict[str, Any],
    extra_defaults: dict[str, Any],
    by_title: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    override = by_title.get(title) or {}
    is_series = _is_truthy_series(override.get("Is_series", False))

    base: dict[str, Any] = {}
    base.update(series_defaults if is_series else standalone_defaults)
    base.update(extra_defaults)
    base.update(override)

    for k in ("raw_text", "chunks", "extracted"):
        base.pop(k, None)
    is_series = _is_truthy_series(base.get("Is_series", is_series))

    if not str(base.get("id") or "").strip():
        base["id"] = title

    # Nested dicts → JSON strings for Chroma
    if isinstance(base.get("meta"), dict):
        base["meta_json"] = _dumps_json(base.get("meta"))
    base.pop("meta", None)

    if isinstance(base.get("chapter"), dict):
        base["chapter_json"] = _dumps_json(base.get("chapter"))
    base.pop("chapter", None)

    if is_series:
        if isinstance(base.get("series"), dict):
            base["series_json"] = _dumps_json(base.get("series"))
        if isinstance(base.get("volume"), dict):
            base["volume_json"] = _dumps_json(base.get("volume"))
    base.pop("series", None)
    base.pop("volume", None)
    if not is_series:
        base.pop("series_json", None)
        base.pop("volume_json", None)

    base["Title"] = title
    base["chunk_id"] = chunk_id
    base["Is_series"] = bool(is_series)
    return _as_chroma_metadata(base)


def main() -> None:
    """
    Build data/ingest/ingest_manifest.jsonl from data/stories/*.txt.

    Optional: data/ingest/ingest_metadata_template.json ({defaults, by_title}).
    Edit the JSONL after generation, then run ingest_manifest.py.
    """
    sys.path.insert(0, "src")
    from storyforge.config.config import load_config

    cfg = load_config()
    base = Path(cfg.get("BASE_PATH") or ".").resolve()
    stories_dir = (base / (cfg.get("Story_input") or "data/stories")).resolve()

    out_dir = (base / "data" / "ingest").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ingest_manifest.jsonl"
    template_path = out_dir / "ingest_metadata_template.json"

    standalone_defaults = _load_json_file(out_dir / "TEMPLATE__by_title_value__standalone.json")
    series_defaults = _load_json_file(out_dir / "TEMPLATE__by_title_value__series.json")
    extra_defaults, by_title = _load_template(path=template_path)

    files = sorted([p for p in stories_dir.glob("*.txt") if p.is_file()], key=lambda p: p.name)
    if not files:
        raise SystemExit(f"No .txt files found in {stories_dir}")

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for fp in files:
            raw = fp.read_text(encoding="utf-8", errors="replace")
            chunks = _chunk_text(raw)
            title = fp.stem
            for i, ch in enumerate(chunks, start=1):
                chunk_id = f"{title}_chunk_{i}"
                rec = {
                    "id": chunk_id,
                    "text": ch,
                    "metadata": _merge_base_metadata(
                        title=title,
                        chunk_id=chunk_id,
                        standalone_defaults=standalone_defaults,
                        series_defaults=series_defaults,
                        extra_defaults=extra_defaults,
                        by_title=by_title,
                    ),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} chunks to {out_path}")


if __name__ == "__main__":
    main()
