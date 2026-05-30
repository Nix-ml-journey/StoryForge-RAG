"""
reset_and_ingest.py
-------------------
Run this once after upgrading to BGE embeddings.

It will:
  1. Reset the Chroma collection via ChromaDB's API (safe even if DB is open)
  2. Delete the old DB directory if possible
  3. Re-ingest all stories from data/stories/ using the new BGE-base model (768-dim)

Usage (from project root, with venv active):
    py scripts/reset_and_ingest.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyforge.config.config import load_config  # noqa: E402
from storyforge.vector_store.ingest_stories import ingest_stories_dir  # noqa: E402


def _delete_dir(path: Path) -> bool:
    """Try to delete a directory. Returns True on success, False if locked."""
    try:
        shutil.rmtree(path)
        print(f"[DELETE] {path}  — Done.")
        return True
    except PermissionError as e:
        print(f"[WARN]   Could not delete {path}")
        print(f"         Reason: {e}")
        print(f"         The collection will be cleared via ChromaDB API instead.")
        return False


def _reset_via_chromadb(chroma_path: Path, collection_name: str) -> None:
    """Delete and recreate the collection using the ChromaDB client directly."""
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_path))
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"[RESET]  Collection '{collection_name}' deleted via ChromaDB API.")
    else:
        print(f"[SKIP]   Collection '{collection_name}' not found — nothing to reset.")


def main() -> None:
    cfg = load_config()
    base = Path(cfg.get("BASE_PATH") or REPO_ROOT).resolve()
    chroma_rel = cfg.get("Chroma_path") or "data/chroma_db"
    chroma_path = (base / chroma_rel).resolve()
    legacy_chroma = REPO_ROOT / "chroma_db"
    collection_name = str(cfg.get("Chroma_collection_name") or "StoryForgeRag_v1")

    print("=" * 60)
    print("StoryForge — Reset & Re-ingest")
    print("=" * 60)

    # --- Step 1: Reset old vector stores ---
    print()
    for path in [chroma_path, legacy_chroma]:
        if not path.exists():
            print(f"[SKIP]   {path}  (not found)")
            continue
        deleted = _delete_dir(path)
        if not deleted:
            # File is locked — use ChromaDB API to drop the collection instead
            _reset_via_chromadb(path, collection_name)

    # --- Step 2: Re-ingest ---
    stories_path = base / (cfg.get("Story_input") or "data/stories")

    print(f"\n[INGEST] Stories from : {stories_path}")
    print(f"         Collection   : {collection_name}")
    print(f"         Embedding    : {cfg.get('Vector_store_model') or 'all-MiniLM-L6-v2'}")
    print("\nThis will download the BGE model on first run (~440 MB) — please wait...\n")

    result = ingest_stories_dir(
        base_path=base,
        stories_dir=stories_path,
        collection_name=collection_name,
    )

    print("=" * 60)
    if result.success:
        print("[OK]  Ingest complete!")
        print(f"      Files seen    : {result.files_seen}")
        print(f"      Chunks written: {result.chunks_written}")
        print(f"      Collection    : {result.collection_name}")
    else:
        print(f"[FAIL] Ingest failed: {result.error}")
        sys.exit(1)
    print("=" * 60)
    print("\nYou can now start the server:  python main.py")


if __name__ == "__main__":
    main()
