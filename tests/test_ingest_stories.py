from __future__ import annotations

from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")

from storyforge.vector_store.chromadb import query_data, reset_vector_store_dir, set_active_collection
from storyforge.vector_store.ingest_stories import ingest_stories_dir


def test_ingest_stories_and_query(tmp_path: Path, monkeypatch):
    # Arrange: create a tiny Stories corpus.
    base = tmp_path
    stories = base / "Stories"
    stories.mkdir(parents=True, exist_ok=True)
    (stories / "dragon.txt").write_text(
        "A brave knight fought a dragon near the old bridge. The dragon fled into the mountains.\n\n"
        "The knight returned with a scar and a story.",
        encoding="utf-8",
    )
    (stories / "forest.txt").write_text(
        "A lost child found a magical forest. The trees whispered warnings and offered shelter.\n\n"
        "By dawn, the child learned the path home.",
        encoding="utf-8",
    )

    # Use an isolated chroma directory under tmp_path.
    chroma_dir = base / "chroma_test"

    # Act: reset + ingest.
    res_reset = reset_vector_store_dir(
        base_path=base, chroma_path=str(chroma_dir), new_collection_name="StoryForgeRag_v1"
    )
    assert res_reset["success"] is True
    set_active_collection("StoryForgeRag_v1")

    res_ingest = ingest_stories_dir(base_path=base, stories_dir=stories, collection_name="StoryForgeRag_v1")
    assert res_ingest.success is True
    assert res_ingest.files_seen == 2
    assert res_ingest.chunks_written >= 2

    # Assert: retrieval returns something relevant.
    q = query_data("dragon", n_results=3, query_type="content")
    assert q is not None
    docs = (q.get("documents") or [[]])[0]
    assert any("dragon" in (d or "").lower() for d in docs)
