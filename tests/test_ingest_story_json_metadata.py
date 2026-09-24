"""ingest_stories_dir() must honour reviewed story_json records (P0.1).

reset_and_ingest.py goes through ingest_stories_dir(), which used to read only
data/stories/*.txt and hardcode Author="" / Summary="" -- silently discarding
section tags, author/summary, and chunk-text fixes made in data/story_json/.
Chroma and the embedding model are faked; no GPU or DB needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

STORY = (
    "In a small village beside a grey lake, a baker named Mara kept a lantern lit every night "
    "so that travelers could find the road when the fog rolled in from the water.\n\n"
    "One winter, travelers stopped asking for bread and started asking for directions, and Mara "
    "wrote their answers in a notebook until it knew more paths than the village map."
)


class _FakeCollection:
    name = "TestCollection"

    def __init__(self):
        self.ids, self.docs, self.mds = [], [], []

    def upsert(self, ids, metadatas, documents, embeddings=None):
        self.ids += ids
        self.mds += metadatas
        self.docs += documents


@pytest.fixture
def ingest(stub_heavy_deps, monkeypatch, tmp_path):
    # storyforge.vector_store.chromadb opens a PersistentClient at import time;
    # swap in a fake module so importing ingest_stories never touches a real DB.
    import sys
    import types

    coll = _FakeCollection()
    fake_chroma_mod = types.ModuleType("storyforge.vector_store.chromadb")
    fake_chroma_mod.get_or_create_collection = lambda name=None: coll
    # Import a fresh ingest_stories bound to the fake; monkeypatch restores the
    # previous sys.modules entries afterwards so other test modules that use the
    # real chromadb (e.g. test_story_json_workflow) are unaffected.
    monkeypatch.setitem(sys.modules, "storyforge.vector_store.chromadb", fake_chroma_mod)
    monkeypatch.delitem(sys.modules, "storyforge.vector_store.ingest_stories", raising=False)
    import importlib

    ingest_mod = importlib.import_module("storyforge.vector_store.ingest_stories")

    monkeypatch.setattr(ingest_mod, "get_or_create_collection", lambda name: coll)
    monkeypatch.setattr(ingest_mod, "_get_embed_model", lambda name: None)
    monkeypatch.setattr(ingest_mod, "load_config", lambda: {})
    (tmp_path / "stories").mkdir()
    (tmp_path / "story_json").mkdir()

    def run(**kw):
        res = ingest_mod.ingest_stories_dir(
            base_path=tmp_path,
            stories_dir=tmp_path / "stories",
            records_dir=tmp_path / "story_json",
            **kw,
        )
        assert res.success, res.error
        return res, coll

    return run, tmp_path


def _write_story(tmp: Path, title: str, text: str = STORY) -> None:
    # write_bytes keeps exact newlines (Windows text-mode write_text would
    # turn an intentional CRLF string into CRCRLF and falsely mark story_json stale).
    (tmp / "stories" / f"{title}.txt").write_bytes(text.encode("utf-8"))


def _write_record(tmp: Path, title: str, *, raw_text: str = STORY, chunks=None) -> None:
    rec = {
        "id": title,
        "Is_series": False,
        "meta": {"author": "Sample Author", "title": "The Crossroads Lantern"},
        "summary": "A baker keeps a lantern lit for travelers.",
        "raw_text": raw_text,
        "chunks": chunks
        if chunks is not None
        else [
            {"chunk_id": f"{title}_chunk_1", "text": "Hand-fixed chunk one about Mara and the lantern.", "section": "setup"},
            {"chunk_id": f"{title}_chunk_2", "text": "   ", "section": "climax"},  # empty -> skipped
            {"chunk_id": f"{title}_chunk_3", "text": "Mara's notebook maps the village.", "section": "resolution"},
        ],
    }
    (tmp / "story_json" / f"{title}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_matching_story_json_supplies_chunks_sections_and_metadata(ingest):
    run, tmp = ingest
    _write_story(tmp, "demo_village")
    _write_record(tmp, "demo_village")

    res, coll = run()

    assert (res.from_story_json, res.stale_story_json, res.without_story_json) == (1, 0, 0)
    assert coll.ids == ["demo_village_chunk_1", "demo_village_chunk_3"]
    assert coll.docs[0] == "Hand-fixed chunk one about Mara and the lantern."
    assert [m["section"] for m in coll.mds] == ["setup", "resolution"]
    md = coll.mds[0]
    assert md["Title"] == "demo_village"  # stable filename-stem contract
    assert md["Display_title"] == "The Crossroads Lantern"
    assert md["Author"] == "Sample Author"
    assert md["Summary"] == "A baker keeps a lantern lit for travelers."
    assert md["chunk_id"] == "demo_village_chunk_1"
    assert md["Is_series"] is False
    assert all(v is not None and isinstance(v, (str, int, float, bool)) for v in md.values())


def test_crlf_only_difference_is_not_stale(ingest):
    run, tmp = ingest
    _write_story(tmp, "demo_village", STORY.replace("\n", "\r\n"))
    _write_record(tmp, "demo_village")
    res, _ = run()
    assert res.from_story_json == 1


def test_stale_story_json_uses_txt_chunks_but_keeps_story_metadata(ingest, caplog):
    run, tmp = ingest
    _write_story(tmp, "demo_village", STORY + "\n\nA new closing paragraph the record has never seen, long enough to chunk.")
    _write_record(tmp, "demo_village")

    res, coll = run()

    assert (res.from_story_json, res.stale_story_json) == (0, 1)
    assert "Hand-fixed chunk one" not in " ".join(coll.docs)
    assert all(m["Author"] == "Sample Author" for m in coll.mds)
    assert all(m["section"] == "" for m in coll.mds)
    assert coll.ids[0] == "demo_village_chunk_1"
    assert "stale" in caplog.text


def test_missing_story_json_keeps_old_behaviour(ingest):
    run, tmp = ingest
    _write_story(tmp, "Lovecraft__Cool_Air")

    res, coll = run()

    assert res.without_story_json == 1
    assert coll.ids[0] == "Lovecraft__Cool_Air_chunk_1"
    assert coll.mds[0]["Title"] == "Lovecraft__Cool_Air"
    assert coll.mds[0]["Author"] == "" and coll.mds[0]["Summary"] == ""


def test_use_story_json_false_ignores_records(ingest):
    run, tmp = ingest
    _write_story(tmp, "demo_village")
    _write_record(tmp, "demo_village")
    res, coll = run(use_story_json=False)
    assert res.without_story_json == 1
    assert coll.mds[0]["Author"] == ""


def test_record_with_only_empty_chunks_falls_back_to_txt(ingest):
    run, tmp = ingest
    _write_story(tmp, "demo_village")
    _write_record(tmp, "demo_village", chunks=[{"chunk_id": "x", "text": "", "section": "setup"}])
    res, coll = run()
    assert res.stale_story_json == 1 and len(coll.ids) >= 1


def test_unreadable_story_json_does_not_abort_ingest(ingest):
    run, tmp = ingest
    _write_story(tmp, "demo_village")
    (tmp / "story_json" / "demo_village.json").write_text("{not json", encoding="utf-8")
    res, _ = run()
    assert res.without_story_json == 1


def test_manifest_and_direct_ingest_write_same_story_metadata(stub_heavy_deps):
    from storyforge.data.records_to_manifest import story_level_metadata

    rec = {"id": "s1", "meta": {"author": " A ", "title": "T"}, "summary": "x" * 5000,
           "Is_series": True, "series": {"series_name": "Saga"}}
    md = story_level_metadata(rec, "file_stem")
    assert md["Title"] == "file_stem"
    assert md["Author"] == "A"
    assert len(md["Summary"]) == 2000
    assert md["Series_name"] == "Saga"
