"""
push_section_metadata.py
-------------------------
After enrich_story_records.py fills in `chunks[].section` (and other metadata)
in data/story_json/*.json, this pushes the UPDATED metadata into Chroma for
matching chunk_ids -- without recomputing embeddings, since the chunk text
itself hasn't changed. This reuses each chunk's existing vector.

Usage (from project root, with venv active):
    py scripts/push_section_metadata.py --glob "Lovecraft__*"
    py scripts/push_section_metadata.py --glob "*"   # everything
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Same serializers the manifest / rebuild ingest paths use (were copy-pasted here).
from storyforge.data.records_to_manifest import _as_chroma_metadata, _dumps_json  # noqa: E402


def main() -> None:
    from storyforge.config.config import load_config
    from storyforge.vector_store.chromadb import get_or_create_collection

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="*",
        help='Glob (relative to data/story_json/) selecting which records to push, e.g. "Lovecraft__*"',
    )
    args = parser.parse_args()

    cfg = load_config()
    base = Path(cfg.get("BASE_PATH") or REPO_ROOT).resolve()
    records_dir = (base / "data" / "story_json").resolve()

    files = sorted(records_dir.glob(f"{args.glob}.json"))
    if not files:
        raise SystemExit(f"No story_json files matched '{args.glob}.json' in {records_dir}")

    collection_name = cfg.get("Chroma_collection_name") or "StoryForgeRag_v1"
    collection = get_or_create_collection(collection_name)
    print(f"Collection '{collection_name}' count before: {collection.count()}")

    total = 0
    for fp in files:
        title = fp.stem
        rec = json.loads(fp.read_text(encoding="utf-8"))
        chunks = rec.get("chunks") or []
        if not chunks:
            print(f"  SKIP {title}: no chunks in record")
            continue

        is_series = bool(rec.get("Is_series", False))
        story_id = str(rec.get("id") or title)
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        chapter = rec.get("chapter") if isinstance(rec.get("chapter"), dict) else {}
        series = rec.get("series") if isinstance(rec.get("series"), dict) else {}
        volume = rec.get("volume") if isinstance(rec.get("volume"), dict) else {}

        ids = [str(c.get("chunk_id") or "") for c in chunks]
        texts = [str(c.get("text") or "") for c in chunks]

        # Pull existing embeddings + metadata -- we only want to overlay the
        # freshly-enriched fields (section, meta/chapter/series in case those
        # changed too), reusing the vectors as-is.
        existing = collection.get(ids=ids, include=["embeddings", "metadatas"])
        existing_by_id = {i: (e, m) for i, e, m in zip(existing["ids"], existing["embeddings"], existing["metadatas"])}
        missing = [i for i in ids if i not in existing_by_id]
        if missing:
            print(f"  SKIP {title}: {len(missing)} chunk_ids not found in collection (e.g. {missing[:3]})")
            continue

        embeddings = []
        metadatas = []
        for ch, cid in zip(chunks, ids):
            emb, old_md = existing_by_id[cid]
            section = str(ch.get("section") or "")
            new_md = dict(old_md or {})
            new_md.update(
                {
                    "Title": title,
                    "chunk_id": cid,
                    "id": story_id,
                    "Is_series": is_series,
                    "section": section,
                    "meta_json": _dumps_json(meta),
                    "chapter_json": _dumps_json(chapter),
                }
            )
            if is_series:
                new_md["series_json"] = _dumps_json(series)
                new_md["volume_json"] = _dumps_json(volume)
            embeddings.append(emb)
            metadatas.append(_as_chroma_metadata(new_md))

        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        total += len(ids)
        print(f"  OK   {title}: {len(ids)} chunks metadata refreshed (embeddings reused, not recomputed)")

    print(f"\nTotal chunks refreshed: {total}")
    print(f"Collection count after: {collection.count()}")


if __name__ == "__main__":
    main()
