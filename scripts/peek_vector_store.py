import json
import sys


def main() -> None:
    # Allow running from repo root without installing package.
    sys.path.insert(0, "src")

    from storyforge.config.config import load_config
    from storyforge.vector_store import chromadb as db

    cfg = load_config()
    collection = cfg.get("Chroma_collection_name") or "StoryForgeRag_v1"

    # Ensure we're looking at the configured collection.
    db.set_active_collection(collection)

    got = db.Collection.get(limit=5, include=["documents", "metadatas"])
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    mds = got.get("metadatas") or []

    out: list[dict] = []
    for i in range(len(ids)):
        out.append(
            {
                "id": ids[i],
                "text_preview": (docs[i] or "")[:500],
                "metadata": mds[i] or {},
            }
        )

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

