import pytest

from storyforge.orchestrator import response_parameter as rp


def test_only_output_folders_are_readable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"BASE_PATH": str(tmp_path), "Generated_story_output": "data/outputs/generated_stories"}
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: cfg)
    ok = tmp_path / "data/outputs/generated_stories/a.txt"
    assert rp._safe_output_path(str(ok)) == ok.resolve()
    assert rp._safe_output_path("data/outputs/generated_stories/a.txt") == ok.resolve()
    for bad in ("setup.yaml", "../secret.txt", str(tmp_path / "data/outputs/generated_stories/../../../setup.yaml")):
        with pytest.raises(ValueError):
            rp._safe_output_path(bad)
