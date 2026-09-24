"""Chroma metadata health report (pure functions, no Chroma import).

Used by ``scripts/validate_chroma_metadata.py`` after an ingest to confirm what
actually landed in the collection: how many chunks / stories, and how much of
the story_json metadata (Author, Summary, Display_title, section) made it in.
Kept free of chromadb / torch imports so it is unit-testable without a DB.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

_UNKNOWN_TITLES = {"", "unknown"}


def _s(v: Any) -> str:
    return str(v if v is not None else "").strip()


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total else 0.0


def fetch_all(collection: Any, *, page_size: int = 1000) -> tuple[list[str], list[dict], list[str]]:
    """Page through a Chroma collection: returns (ids, metadatas, documents)."""
    ids: list[str] = []
    mds: list[dict] = []
    docs: list[str] = []
    offset = 0
    while True:
        got = collection.get(limit=page_size, offset=offset, include=["metadatas", "documents"])
        page_ids = list(got.get("ids") or [])
        if not page_ids:
            break
        ids += page_ids
        mds += [m or {} for m in (got.get("metadatas") or [{}] * len(page_ids))]
        docs += [d or "" for d in (got.get("documents") or [""] * len(page_ids))]
        if len(page_ids) < page_size:
            break
        offset += page_size
    return ids, mds, docs


def summarize_metadata(
    metadatas: Iterable[Optional[dict]],
    *,
    ids: Optional[Iterable[str]] = None,
    documents: Optional[Iterable[str]] = None,
    samples: int = 5,
) -> dict[str, Any]:
    """Compute coverage stats + red flags for a list of chunk metadatas."""
    mds = [m or {} for m in metadatas]
    id_list = list(ids) if ids is not None else [""] * len(mds)
    doc_list = list(documents) if documents is not None else [""] * len(mds)
    total = len(mds)

    with_author = sum(1 for m in mds if _s(m.get("Author")))
    with_summary = sum(1 for m in mds if _s(m.get("Summary")))
    with_display = sum(1 for m in mds if _s(m.get("Display_title")))
    with_section = sum(1 for m in mds if _s(m.get("section")))
    missing_title = sum(1 for m in mds if _s(m.get("Title")).lower() in _UNKNOWN_TITLES)
    missing_chunk_id = sum(1 for m in mds if not _s(m.get("chunk_id")))
    empty_docs = sum(1 for d in doc_list if documents is not None and not _s(d))

    by_title: dict[str, list[dict]] = defaultdict(list)
    for m in mds:
        by_title[_s(m.get("Title")) or "<missing>"].append(m)

    stories_without_author = sorted(t for t, g in by_title.items() if not any(_s(m.get("Author")) for m in g))
    stories_without_summary = sorted(t for t, g in by_title.items() if not any(_s(m.get("Summary")) for m in g))
    # DATA_PREP: "a whole file tagged `setup` is a bad labeler run".
    single_section_stories = sorted(
        t
        for t, g in by_title.items()
        if len(g) >= 4 and len({_s(m.get("section")) for m in g}) == 1 and _s(g[0].get("section"))
    )

    sample_rows = []
    for i in range(min(samples, total)):
        m = mds[i]
        sample_rows.append(
            {
                "id": id_list[i],
                "Title": _s(m.get("Title")),
                "Display_title": _s(m.get("Display_title")),
                "Author": _s(m.get("Author")),
                "Summary": _s(m.get("Summary"))[:80],
                "section": _s(m.get("section")),
                "text": _s(doc_list[i])[:100],
            }
        )

    return {
        "total_chunks": total,
        "unique_titles": len(by_title),
        "pct_author": _pct(with_author, total),
        "pct_summary": _pct(with_summary, total),
        "pct_display_title": _pct(with_display, total),
        "pct_section": _pct(with_section, total),
        "missing_title_chunks": missing_title,
        "missing_chunk_id_chunks": missing_chunk_id,
        "empty_document_chunks": empty_docs,
        "stories_without_author": stories_without_author,
        "stories_without_summary": stories_without_summary,
        "single_section_stories": single_section_stories,
        "section_counts": dict(Counter(_s(m.get("section")) or "<none>" for m in mds).most_common(12)),
        "chunks_per_title_min": min((len(g) for g in by_title.values()), default=0),
        "chunks_per_title_max": max((len(g) for g in by_title.values()), default=0),
        "samples": sample_rows,
    }


def problems(summary: dict[str, Any]) -> list[str]:
    """Hard problems (break retrieval / attribution / eval), not just missing nice-to-haves."""
    out = []
    if summary["total_chunks"] == 0:
        out.append("collection is empty -- ingest did not run or wrote to another Chroma_path/collection")
    if summary["missing_title_chunks"]:
        out.append(f"{summary['missing_title_chunks']} chunk(s) have no Title (breaks diversity + retrieval_eval)")
    if summary["missing_chunk_id_chunks"]:
        out.append(f"{summary['missing_chunk_id_chunks']} chunk(s) have no chunk_id (breaks fact attribution)")
    if summary["empty_document_chunks"]:
        out.append(f"{summary['empty_document_chunks']} chunk(s) have empty text")
    return out


def format_report(summary: dict[str, Any], *, list_limit: int = 15) -> str:
    def _lst(items: list[str]) -> str:
        if not items:
            return "none"
        more = f" ... (+{len(items) - list_limit} more)" if len(items) > list_limit else ""
        return ", ".join(items[:list_limit]) + more

    lines = [
        f"Total chunks          : {summary['total_chunks']}",
        f"Unique Titles         : {summary['unique_titles']}  "
        f"(chunks/title min {summary['chunks_per_title_min']}, max {summary['chunks_per_title_max']})",
        f"Non-empty Author      : {summary['pct_author']}%",
        f"Non-empty Summary     : {summary['pct_summary']}%",
        f"Non-empty Display_title: {summary['pct_display_title']}%",
        f"Non-empty section     : {summary['pct_section']}%",
        f"Section counts        : {summary['section_counts']}",
        "",
        f"Stories without Author  : {_lst(summary['stories_without_author'])}",
        f"Stories without Summary : {_lst(summary['stories_without_summary'])}",
        f"Single-section stories  : {_lst(summary['single_section_stories'])}",
    ]
    probs = problems(summary)
    lines += ["", "PROBLEMS: " + ("none" if not probs else "")] + [f"  - {p}" for p in probs]
    if summary["pct_author"] == 0 and summary["total_chunks"]:
        lines.append(
            "HINT: Author is empty everywhere -- fill meta.author in data/story_json/*.json, "
            "then re-run scripts/reset_and_ingest.py (it reads story_json when present)."
        )
    lines += ["", "Sample rows:"]
    for r in summary["samples"]:
        lines.append(
            f"  {r['id']} | Title={r['Title']} | Display_title={r['Display_title'] or '-'} | "
            f"Author={r['Author'] or '-'} | section={r['section'] or '-'} | Summary={r['Summary'] or '-'}"
        )
        if r["text"]:
            lines.append(f"      text: {r['text']!r}")
    return "\n".join(lines)
