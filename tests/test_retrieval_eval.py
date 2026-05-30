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
