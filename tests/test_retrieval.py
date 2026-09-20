"""Regression tests for storyforge.rag.retrieval.retrieve_docs()'s pipeline order.

retrieve_docs() must rerank the full (hybrid-fused) candidate pool *before*
diversity selection narrows it down to a handful of titles -- not after.
Reranking after diversity means a correctly-relevant title that diversity
didn't happen to keep (because it ran on the raw dense/BM25 order) never
reaches the cross-encoder at all, and a title that did survive diversity is
only reranked against the few other titles diversity happened to keep, not
the full pool. See docs/PROJECT_JOURNEY.md for the retrieval_eval cases this
targets (Frankenstein, Jekyll_and_Hyde__Olalla, Jekyll_and_Hyde__The_Body_Snatcher,
Lovecraft__The_Statement_of_Randolph_Carter, Lovecraft__Cool_Air,
Lovecraft__The_Haunter_of_the_Dark).
"""

from __future__ import annotations


class _FakeDoc:
    def __init__(self, title: str, text: str):
        self.page_content = text
        self.metadata = {"Title": title, "chunk_id": f"{title}_c"}


def test_retrieve_docs_reranks_before_diversity_selection(stub_heavy_deps, monkeypatch):
    import storyforge.rag.retrieval as retrieval_mod

    pool = [_FakeDoc(f"Title{i}", f"text {i}") for i in range(8)]

    class FakeRetriever:
        def invoke(self, query):
            return list(pool)

    class FakeVectorstore:
        def as_retriever(self, search_kwargs=None):
            return FakeRetriever()

    monkeypatch.setattr(retrieval_mod, "_build_vectorstore", lambda cfg: FakeVectorstore())

    call_order: list[tuple[str, int]] = []

    def fake_rerank(query, docs, *, reranker_model_id, top_n, device):
        call_order.append(("rerank", len(docs)))
        return docs

    def fake_diversity(docs, *, n_stories, chunks_per_story, target=None):
        call_order.append(("diversity", len(docs)))
        return docs[:target] if target else docs

    monkeypatch.setattr(retrieval_mod, "_rerank_docs", fake_rerank)
    monkeypatch.setattr(retrieval_mod, "_select_diverse_stories", fake_diversity)

    cfg = {
        "Hybrid_search_enabled": False,
        "Reranker_enabled": True,
        "Story_generation_n_results": 10,
        "Story_generation_rerank_top_n": 10,
    }
    retrieval_mod.retrieve_docs(query="q", cfg=cfg)

    assert [step for step, _ in call_order] == ["rerank", "diversity"]
    # Rerank must see the full 8-doc pool, not one diversity already narrowed down.
    assert call_order[0] == ("rerank", 8)


def test_retrieve_docs_skips_diversity_narrowing_before_rerank_sees_it(stub_heavy_deps, monkeypatch):
    """A title diversity would have dropped (the pool is bigger than the target
    chunk count, and this title is far down the raw retrieval order) must still
    reach the reranker before diversity ever narrows the pool."""
    import storyforge.rag.retrieval as retrieval_mod

    # 12 distinct single-chunk titles, target_chunks=3: with the old
    # rerank-after-diversity order, diversity (n_stories=1) picks title "T0" plus
    # backfill up to target=3 in raw list order -- "T11" never survives to be
    # reranked. With rerank running first, the full 12-doc pool reaches it.
    pool = [_FakeDoc(f"T{i}", f"text {i}") for i in range(12)]

    class FakeRetriever:
        def invoke(self, query):
            return list(pool)

    class FakeVectorstore:
        def as_retriever(self, search_kwargs=None):
            return FakeRetriever()

    monkeypatch.setattr(retrieval_mod, "_build_vectorstore", lambda cfg: FakeVectorstore())

    seen_titles_at_rerank: list[str] = []

    def fake_rerank(query, docs, *, reranker_model_id, top_n, device):
        seen_titles_at_rerank.extend(d.metadata["Title"] for d in docs)
        return docs

    monkeypatch.setattr(retrieval_mod, "_rerank_docs", fake_rerank)

    cfg = {
        "Hybrid_search_enabled": False,
        "Reranker_enabled": True,
        "Story_generation_n_results": 3,
        "Story_generation_rerank_top_n": 12,
    }
    retrieval_mod.retrieve_docs(query="q", cfg=cfg, n_stories=1, chunks_per_story=1)

    assert "T11" in seen_titles_at_rerank
