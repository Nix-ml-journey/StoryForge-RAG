import json
import sys
from pathlib import Path


def main() -> None:
    """
    Step 1b (actual ingest):
    - Reads `data/ingest/ingest_manifest.jsonl`
    - Upserts ids/documents/metadatas into the configured Chroma collection.

    Embeddings are computed explicitly with the project's configured Vector_store_model
    (BGE by default) and passed to upsert(). Without this, Chroma silently falls back to
    its own default embedder (all-MiniLM-L6-v2, 384-dim), which does not match the
    768-dim BGE vectors the rest of the collection was ingested with — Chroma then
    rejects the whole batch with a dimension-mismatch error.
    """
    sys.path.insert(0, "src")
    from storyforge.config.config import load_config
    from storyforge.vector_store.chromadb import get_or_create_collection
    from storyforge.vector_store.ingest_stories import _embed_chunks, _get_embed_model

    cfg = load_config()
    base = Path(cfg.get("BASE_PATH") or ".").resolve()
    manifest = (base / "data" / "ingest" / "ingest_manifest.jsonl").resolve()
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest} (run scripts/records_to_ingest_manifest.py first)")

    collection_name = cfg.get("Chroma_collection_name") or "StoryForgeRag_v1"
    collection = get_or_create_collection(collection_name)

    embed_model_name = str(cfg.get("Vector_store_model") or "BAAI/bge-base-en-v1.5")
    is_bge = "bge" in embed_model_name.lower()
    embed_model = _get_embed_model(embed_model_name)
    if embed_model is None:
        raise SystemExit(
            f"Could not load embedding model '{embed_model_name}' "
            "(sentence-transformers missing?). Run: pip install sentence-transformers"
        )
    print(f"Embedding with: {embed_model_name}")

    ids: list[str] = []
    docs: list[str] = []
    mds: list[dict] = []

    written = 0

    def _flush() -> None:
        nonlocal ids, docs, mds, written
        if not ids:
            return
        embeddings = _embed_chunks(embed_model, docs, is_bge=is_bge)
        if embeddings is None:
            raise SystemExit("Embedding failed for a batch — aborting so the collection is not left inconsistent.")
        collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=mds)
        written += len(ids)
        ids, docs, mds = [], [], []

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
                _flush()

    _flush()

    print(f"Ingested {written} chunks into collection '{collection.name}' from {manifest}")


if __name__ == "__main__":
    main()

