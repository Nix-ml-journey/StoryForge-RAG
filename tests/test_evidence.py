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
