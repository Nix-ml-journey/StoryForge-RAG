import json
import sys
from pathlib import Path


def main() -> None:
    """
    Step 1b (actual ingest):
    - Reads `data/ingest/ingest_manifest.jsonl`
    - Upserts ids/documents/metadatas into the configured Chroma collection.
    """
    sys.path.insert(0, "src")
    from storyforge.config.config import load_config
    from storyforge.vector_store.chromadb import get_or_create_collection

    cfg = load_config()
    base = Path(cfg.get("BASE_PATH") or ".").resolve()
    manifest = (base / "data" / "ingest" / "ingest_manifest.jsonl").resolve()
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest} (run scripts/prepare_ingest_manifest.py first)")

    collection_name = cfg.get("Chroma_collection_name") or "StoryForgeRag_v1"
    collection = get_or_create_collection(collection_name)

    ids: list[str] = []
    docs: list[str] = []
    mds: list[dict] = []

    written = 0
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.append(str(rec["id"]))
            docs.append(str(rec.get("text") or ""))
            md = rec.get("metadata") or {}
            if isinstance(md, dict) and "chunk_id" not in md:
                md["chunk_id"] = str(rec["id"])
            mds.append(md if isinstance(md, dict) else {})

            # Batch upserts to avoid huge memory for large corpora.
            if len(ids) >= 256:
                collection.upsert(ids=ids, documents=docs, metadatas=mds)
                written += len(ids)
                ids, docs, mds = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=mds)
        written += len(ids)

    print(f"Ingested {written} chunks into collection '{collection.name}' from {manifest}")


if __name__ == "__main__":
    main()

