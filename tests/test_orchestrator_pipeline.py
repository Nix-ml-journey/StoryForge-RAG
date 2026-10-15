import sys
import types


def test_run_pipeline_calls_match_the_orchestrator_signatures(stub_heavy_deps, monkeypatch):
    # Guards the internal calls: dropping a parameter once left run_pipeline calling download_book(urls=...) and
    # generate_story(generation_type=...), which ruff and the route tests could not see.
    seen = []
    ok = {"success": True}
    fake = types.ModuleType("storyforge.orchestrator.response_parameter")  # the real one opens Chroma on import
    fake.download_book_archive = lambda *a, **k: seen.append("download") or ok
    fake.extract_text_result = lambda *a, **k: seen.append("extract") or ok
    fake.generate_story_result = lambda *a, **k: seen.append("generate") or ok
    fake.generate_story_agentic_result = lambda *a, **k: seen.append("agentic") or ok
    import storyforge.orchestrator as package

    monkeypatch.setitem(sys.modules, "storyforge.orchestrator.response_parameter", fake)
    monkeypatch.setattr(package, "response_parameter", fake, raising=False)  # `from . import x` reads this attribute first
    monkeypatch.delitem(sys.modules, "storyforge.orchestrator.orchestrator", raising=False)

    from storyforge.orchestrator.orchestrator import Orchestrator

    result = Orchestrator().run_pipeline(
        "A title", steps=["0_fetch_and_extract", "4_generate_story_3step", "4_generate_story_agentic"]
    )

    assert result["success"], result
    assert seen == ["download", "extract", "generate", "agentic"]
