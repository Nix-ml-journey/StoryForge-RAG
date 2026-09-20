from storyforge.evaluation.retrieval_eval import (
    RetrievalCase,
    evaluate_case,
    evaluate_retrieval,
    load_cases,
    normalize_results,
    summarize_results,
)


def test_normalize_results_accepts_chroma_shape():
    raw = {
        "documents": [["A child walked through the forest."]],
        "metadatas": [[{"Title": "Hansel and Gretel", "Author": "Brothers Grimm", "Summary": "forest house"}]],
        "distances": [[0.12]],
    }

    hits = normalize_results(raw)

    assert len(hits) == 1
    assert hits[0].title == "Hansel and Gretel"
    assert hits[0].distance == 0.12


def test_evaluate_case_tracks_topk_and_fact_coverage():
    case = RetrievalCase(
        query="lost children forest",
        expected_title="Hansel and Gretel",
        expected_facts=["children", "forest", "witch"],
    )
    raw_results = [
        {"metadata": {"Title": "Rapunzel", "Summary": "tower hair"}, "document": ""},
        {
            "metadata": {"Title": "Hansel and Gretel", "Summary": "children in forest"},
            "document": "The witch waited near the house.",
        },
    ]

    result = evaluate_case(case, raw_results, k=3)

    assert result["expected_rank"] == 2
    assert not result["top1_match"]
    assert result["top3_match"]
    assert result["fact_coverage"] == 1.0


def test_summarize_results_computes_accuracy_and_average_fact_coverage():
    results = [
        {"top1_match": True, "top3_match": True, "fact_coverage": 1.0},
        {"top1_match": False, "top3_match": True, "fact_coverage": 0.5},
    ]

    summary = summarize_results(results, k=3)

    assert summary["total_cases"] == 2
    assert summary["top1_accuracy"] == 0.5
    assert summary["top3_accuracy"] == 1.0
    assert summary["average_fact_coverage"] == 0.75


def test_evaluate_retrieval_uses_query_function():
    cases = [RetrievalCase(query="tower hair", expected_title="Rapunzel", expected_facts=["tower"])]

    def query_fn(query, k):
        assert query == "tower hair"
        assert k == 3
        return [{"metadata": {"Title": "Rapunzel", "Summary": "tower hair"}, "document": ""}]

    report = evaluate_retrieval(cases, query_fn, k=3)

    assert report["summary"]["top1_accuracy"] == 1.0
    assert report["cases"][0]["matched_facts"] == ["tower"]


def test_load_cases_accepts_object_with_cases_list(tmp_path):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        '{"cases": [{"query": "q", "expected_title": "Title", "expected_facts": ["fact"]}]}',
        encoding="utf-8",
    )

    cases = load_cases(cases_file)

    assert len(cases) == 1
    assert cases[0].query == "q"
    assert cases[0].expected_facts == ["fact"]


def test_example_fixture_has_realistic_diverse_cases():
    """tests/fixtures/retrieval_eval_cases.example.json should stay a meaningful
    retrieval regression fixture: enough cases, unique titles/queries, and
    verifiable facts -- not empty or degenerate after a future edit."""
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "retrieval_eval_cases.example.json"
    cases = load_cases(fixture)

    assert 20 <= len(cases) <= 40, f"expected ~20-30 cases, got {len(cases)}"

    titles = [c.expected_title for c in cases]
    assert len(set(titles)) == len(titles), "expected_title should be unique per case"

    queries = [c.query for c in cases]
    assert len(set(queries)) == len(queries), "query should be unique per case"

    for case in cases:
        assert len(case.query.split()) >= 5, f"query too short to be realistic: {case.query!r}"
        assert case.expected_facts, f"case for {case.expected_title!r} has no expected_facts"


def test_make_retrieve_docs_query_fn_routes_through_real_pipeline(stub_heavy_deps, monkeypatch):
    """The eval harness must query storyforge.rag.retrieval.retrieve_docs() -- the
    real Step 1 pipeline (hybrid BM25+dense fusion, diverse-title selection,
    cross-encoder reranking) -- not a bare dense-only Chroma query. Regression
    test for a bug where the harness called Orchestrator.query_vector_store()
    instead, silently bypassing hybrid search and reranking entirely."""
    import storyforge.rag.retrieval as retrieval_mod
    from storyforge.evaluation.retrieval_eval import make_retrieve_docs_query_fn

    captured: dict = {}

    def fake_retrieve_docs(query, cfg):
        captured["query"] = query
        captured["cfg"] = cfg
        return ["doc1", "doc2"]

    def fake_docs_to_chunks(docs):
        captured["docs"] = docs
        return [{"chunk_id": "c1", "title": "T", "metadata": {"Title": "T"}, "text": "hi"}]

    monkeypatch.setattr(retrieval_mod, "retrieve_docs", fake_retrieve_docs)
    monkeypatch.setattr(retrieval_mod, "_docs_to_chunks", fake_docs_to_chunks)

    cfg = {"Story_generation_n_results": 3}
    query_fn = make_retrieve_docs_query_fn(cfg)
    result = query_fn("a query", 3)

    assert captured["query"] == "a query"
    assert captured["cfg"] is cfg
    assert captured["docs"] == ["doc1", "doc2"]
    assert result == [{"chunk_id": "c1", "title": "T", "metadata": {"Title": "T"}, "text": "hi"}]


def test_retrieve_docs_query_fn_output_is_scorable_end_to_end(stub_heavy_deps, monkeypatch):
    """The chunk dicts retrieve_docs()/_docs_to_chunks() produce must be a shape
    normalize_results()/evaluate_case() can actually score (title + text)."""
    import storyforge.rag.retrieval as retrieval_mod
    from storyforge.evaluation.retrieval_eval import make_retrieve_docs_query_fn

    def fake_retrieve_docs(query, cfg):
        return ["ignored"]

    def fake_docs_to_chunks(docs):
        return [
            {
                "chunk_id": "Rapunzel_chunk_1",
                "title": "Rapunzel",
                "metadata": {"Title": "Rapunzel", "Author": "", "Summary": ""},
                "text": "A tower with no doors, only a high window.",
            }
        ]

    monkeypatch.setattr(retrieval_mod, "retrieve_docs", fake_retrieve_docs)
    monkeypatch.setattr(retrieval_mod, "_docs_to_chunks", fake_docs_to_chunks)

    query_fn = make_retrieve_docs_query_fn({})
    case = RetrievalCase(query="tower window", expected_title="Rapunzel", expected_facts=["tower"])
    result = evaluate_case(case, query_fn(case.query, 3), k=3)

    assert result["top1_match"]
    assert result["fact_coverage"] == 1.0
