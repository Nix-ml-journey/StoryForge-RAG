"""
refresh_chunk_embeddings.py
----------------------------
Re-embed and upsert chunks for specific story_json records, WITHOUT touching
their existing Chroma metadata (Author, Is_series, section, meta_json, etc.).

Use this after editing a record's `chunks[].text` / `raw_text` in-place (e.g. a
text-cleanup pass) where the chunk count and chunk_ids are unchanged — only the
chunk text itself changed, so only the document + embedding need refreshing.

Usage (from project root, with venv active):
    py scripts/refresh_chunk_embeddings.py --glob "Lovecraft__*"
    py scripts/refresh_chunk_embeddings.py --glob "War_of_the_Worlds"
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


def main() -> None:
    from storyforge.config.config import load_config
    from storyforge.vector_store.chromadb import get_or_create_collection
    from storyforge.vector_store.ingest_stories import _embed_chunks, _get_embed_model

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="*",
        help='Glob (relative to data/story_json/) selecting which records to refresh, e.g. "Lovecraft__*"',
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

    embed_model_name = str(cfg.get("Vector_store_model") or "BAAI/bge-base-en-v1.5")
    is_bge = "bge" in embed_model_name.lower()
    embed_model = _get_embed_model(embed_model_name)
    if embed_model is None:
        raise SystemExit(f"Could not load embedding model '{embed_model_name}'")
    print(f"Embedding with: {embed_model_name}")

    total = 0
    for fp in files:
        title = fp.stem
        rec = json.loads(fp.read_text(encoding="utf-8"))
        chunks = rec.get("chunks") or []
        if not chunks:
            print(f"  SKIP {title}: no chunks in record")
            continue

        ids = [str(c.get("chunk_id") or "") for c in chunks]
        texts = [str(c.get("text") or "") for c in chunks]

        existing = collection.get(ids=ids, include=["metadatas"])
        existing_by_id = dict(zip(existing["ids"], existing["metadatas"]))
        missing = [i for i in ids if i not in existing_by_id]
        if missing:
            print(f"  SKIP {title}: {len(missing)} chunk_ids not found in collection (e.g. {missing[:3]}) "
                  f"-- run the normal ingest path first if this is a brand-new title")
            continue
        metadatas = [existing_by_id[i] for i in ids]

        embeddings = _embed_chunks(embed_model, texts, is_bge=is_bge)
        if embeddings is None:
            print(f"  SKIP {title}: embedding failed")
            continue

        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        total += len(ids)
        print(f"  OK   {title}: {len(ids)} chunks re-embedded (metadata preserved)")

    print(f"\nTotal chunks refreshed: {total}")
    print(f"Collection count after: {collection.count()}")


if __name__ == "__main__":
    main()
