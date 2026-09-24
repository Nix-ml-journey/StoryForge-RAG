"""Tests for storyforge.vector_store.metadata_report (no Chroma / GPU needed)."""
from __future__ import annotations

from storyforge.vector_store.metadata_report import fetch_all, format_report, problems, summarize_metadata


def _md(title, author="", summary="", section="", chunk_id="c", display=""):
    return {"Title": title, "Author": author, "Summary": summary, "section": section,
            "chunk_id": chunk_id, "Display_title": display}


def test_coverage_percentages_and_unique_titles():
    mds = [
        _md("A", author="X", summary="s", section="setup", display="The A"),
        _md("A", author="X", summary="s", section="climax"),
        _md("B"),
        _md("B"),
    ]
    s = summarize_metadata(mds, ids=["a1", "a2", "b1", "b2"], documents=["t"] * 4)
    assert s["total_chunks"] == 4
    assert s["unique_titles"] == 2
    assert s["pct_author"] == 50.0
    assert s["pct_summary"] == 50.0
    assert s["pct_section"] == 50.0
    assert s["pct_display_title"] == 25.0
    assert s["stories_without_author"] == ["B"]
    assert s["stories_without_summary"] == ["B"]
    assert problems(s) == []


def test_hard_problems_are_flagged():
    mds = [_md("", chunk_id="x"), _md("Unknown", chunk_id=""), _md("C")]
    s = summarize_metadata(mds, documents=["ok", "", "ok"])
    probs = problems(s)
    assert any("no Title" in p for p in probs)
    assert any("no chunk_id" in p for p in probs)
    assert any("empty text" in p for p in probs)


def test_empty_collection_is_a_problem():
    s = summarize_metadata([])
    assert s["total_chunks"] == 0
    assert problems(s) and "empty" in problems(s)[0]
    assert "PROBLEMS" in format_report(s)


def test_single_section_story_detected():
    mds = [_md("Lazy", section="setup") for _ in range(5)] + [_md("Good", section=s) for s in ("setup", "climax", "resolution", "epilogue")]
    s = summarize_metadata(mds)
    assert s["single_section_stories"] == ["Lazy"]


def test_report_hints_when_author_empty_everywhere():
    s = summarize_metadata([_md("A"), _md("B")], ids=["1", "2"], documents=["hello", "world"])
    text = format_report(s)
    assert "Non-empty Author      : 0.0%" in text
    assert "HINT" in text and "reset_and_ingest.py" in text
    assert "Sample rows" in text and "Title=A" in text


def test_none_metadata_rows_are_tolerated():
    s = summarize_metadata([None, _md("A", author="X")])
    assert s["total_chunks"] == 2 and s["pct_author"] == 50.0


def test_fetch_all_pages_through_collection():
    rows = [(f"id{i}", _md(f"T{i % 3}"), f"doc{i}") for i in range(7)]

    class _Coll:
        def get(self, limit, offset, include):
            page = rows[offset : offset + limit]
            return {"ids": [r[0] for r in page], "metadatas": [r[1] for r in page], "documents": [r[2] for r in page]}

    ids, mds, docs = fetch_all(_Coll(), page_size=3)
    assert ids == [r[0] for r in rows]
    assert len(mds) == 7 and docs[-1] == "doc6"
