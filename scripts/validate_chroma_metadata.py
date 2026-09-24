"""
Validate what an ingest actually wrote to Chroma (offline, no HF, no GPU).

Prints: total chunks, unique Titles, % non-empty Author / Summary /
Display_title / section, stories missing metadata, single-section stories,
hard problems (missing Title / chunk_id / empty text), and sample rows.

Usage (from repo root):
    py scripts/validate_chroma_metadata.py
    py scripts/validate_chroma_metadata.py --samples 10
    py scripts/validate_chroma_metadata.py --json > Evaluation/chroma_metadata_report.json
    py scripts/validate_chroma_metadata.py --strict   # exit 1 on hard problems

Read-only: never writes to the collection.
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

from storyforge.config.config import load_config  # noqa: E402
from storyforge.vector_store.metadata_report import (  # noqa: E402
    fetch_all,
    format_report,
    problems,
    summarize_metadata,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default=None, help="Collection name (default: Chroma_collection_name)")
    ap.add_argument("--samples", type=int, default=5, help="Sample rows to print")
    ap.add_argument("--json", action="store_true", help="Print the summary as JSON")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if hard problems are found")
    args = ap.parse_args()

    cfg = load_config()
    name = args.collection or cfg.get("Chroma_collection_name") or "StoryForgeRag_v1"
    base = Path(cfg.get("BASE_PATH") or REPO_ROOT)
    chroma_dir = (base / (cfg.get("Chroma_path") or "data/chroma_db")).resolve()

    from storyforge.vector_store.chromadb import get_or_create_collection

    collection = get_or_create_collection(name)
    ids, mds, docs = fetch_all(collection)
    summary = summarize_metadata(mds, ids=ids, documents=docs, samples=args.samples)
    summary["collection"] = name
    summary["chroma_path"] = str(chroma_dir)
    summary["embedding_model_config"] = str(cfg.get("Vector_store_model") or "")

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("=" * 70)
        print(f"Chroma metadata report -- collection '{name}'")
        print(f"Chroma path           : {chroma_dir}")
        print(f"Vector_store_model    : {summary['embedding_model_config']}")
        print("=" * 70)
        print(format_report(summary))

    return 1 if (args.strict and problems(summary)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
