from storyforge.rag.attribution import parse_grounded_facts_json
from storyforge.rag.evidence import EvidenceSearch


def _facts(*names):
    items = ",".join(f'{{"type":"who","fact":"{n} fights Zoruk","source_chunk_ids":["c1"]}}' for n in names)
    return parse_grounded_facts_json('{"facts":[' + items + "]}")


def _stages(monkeypatch, answers):
    """Scripted stages: each extraction call returns the next fact set; queries are recorded."""
    import storyforge.rag.extraction as extraction_mod
    import storyforge.rag.retrieval as retrieval_mod

    seen = []
    monkeypatch.setattr(retrieval_mod, "retrieve_docs", lambda q, cfg, **k: seen.append((q, k)) or ["doc"])
    monkeypatch.setattr(retrieval_mod, "_docs_to_chunks", lambda docs: [{"chunk_id": "c1", "text": "t"}])
    monkeypatch.setattr(retrieval_mod, "_docs_to_context", lambda docs: "ctx")
    it = iter(answers)
    monkeypatch.setattr(extraction_mod, "extract_grounded_facts", lambda q, chunks, cfg: ("raw", next(it)))
    return seen


def test_gather_merges_previous_facts_instead_of_replacing(stub_heavy_deps, monkeypatch):
    _stages(monkeypatch, [_facts("Alana"), _facts("Bora")])
    search = EvidenceSearch({})
    first = search.gather("q")
    both = search.gather("q again", first)
    assert [f.fact for f in both.facts.facts] == ["Alana fights Zoruk", "Bora fights Zoruk"]
    assert both.query == "q again" and both.context == "ctx"


def test_preflight_widens_then_reformulates_until_enough_facts(stub_heavy_deps, monkeypatch):
    seen = _stages(monkeypatch, [_facts("Alana"), _facts("Bora"), _facts("Cora", "Dora")])
    search = EvidenceSearch({"Agentic_loop_min_facts": 3, "Agentic_loop_preflight_tiers": 2})
    out = search.preflight("the duel", search.gather("the duel"))
    assert len(out.facts.facts) == 4  # merged across all three gathers
    assert seen[1][1]["k_boost"] == 2.0 and seen[1][0] == "the duel"  # tier 1: wider, same query
    assert seen[2][0].startswith("the duel ") and "Alana" in seen[2][0]  # tier 2: query gains names from the facts


def test_preflight_does_nothing_when_evidence_is_already_enough(stub_heavy_deps, monkeypatch):
    seen = _stages(monkeypatch, [_facts("A", "B", "C")])
    search = EvidenceSearch({"Agentic_loop_min_facts": 3})
    search.preflight("q", search.gather("q"))
    assert len(seen) == 1


class _Doc:
    def __init__(self, title, n):
        self.metadata = {"Title": title, "chunk_id": f"{title}_chunk_{n}"}
        self.page_content = f"{title} text {n}"


def _story_first_stages(monkeypatch, *, lead_hits):
    """First search returns a mixed list (top-1 = 'Lead'); a Title-filtered search returns lead_hits."""
    import storyforge.rag.extraction as extraction_mod
    import storyforge.rag.retrieval as retrieval_mod

    calls, extracted = [], []

    def fake_retrieve(q, cfg, **k):
        calls.append(k.get("filter_metadata"))
        if k.get("filter_metadata"):
            return lead_hits
        return [_Doc("Lead", 1), _Doc("Other", 1), _Doc("Other", 2), _Doc("Third", 1)]

    monkeypatch.setattr(retrieval_mod, "retrieve_docs", fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_docs_to_chunks", lambda docs: [{"chunk_id": d.metadata["chunk_id"]} for d in docs])
    monkeypatch.setattr(retrieval_mod, "_docs_to_context", lambda docs: "ctx")
    monkeypatch.setattr(
        extraction_mod, "extract_grounded_facts",
        lambda q, chunks, cfg: extracted.append([c["chunk_id"] for c in chunks]) or ("raw", _facts("A")),
    )
    return calls, extracted


def test_story_first_extracts_only_from_the_lead_story(stub_heavy_deps, monkeypatch):
    calls, extracted = _story_first_stages(monkeypatch, lead_hits=[_Doc("Lead", 1), _Doc("Lead", 2), _Doc("Lead", 3)])
    EvidenceSearch({"Story_first_retrieval": True, "Story_first_chunks": 2}).gather("q")
    assert calls == [None, {"Title": "Lead"}]  # one mixed search to find the lead, one filtered search for its chunks
    assert extracted == [["Lead_chunk_1", "Lead_chunk_2"]]  # capped at Story_first_chunks, nothing from other titles


def test_story_first_is_off_by_default(stub_heavy_deps, monkeypatch):
    calls, extracted = _story_first_stages(monkeypatch, lead_hits=[_Doc("Lead", 1)])
    EvidenceSearch({}).gather("q")
    assert calls == [None] and extracted == [["Lead_chunk_1", "Other_chunk_1", "Other_chunk_2", "Third_chunk_1"]]


def test_story_first_falls_back_to_the_mixed_list_when_the_filtered_search_is_empty(stub_heavy_deps, monkeypatch):
    _, extracted = _story_first_stages(monkeypatch, lead_hits=[])
    EvidenceSearch({"Story_first_retrieval": True}).gather("q")
    assert extracted == [["Lead_chunk_1", "Other_chunk_1", "Other_chunk_2", "Third_chunk_1"]]
