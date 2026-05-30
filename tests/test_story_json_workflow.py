from __future__ import annotations

import json
from pathlib import Path

import pytest

from storyforge.data.records_to_manifest import records_to_ingest_manifest
from storyforge.data.story_records import create_story_records


def _write_series_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "Is_series": True,
                "id": "",
                "series": {"id": "", "series_name": ""},
                "volume": {"volume_id": "", "volume_name": "", "volume_number": 0},
                "chapter": {"chapter_id": "", "chapter_number": 1, "chapter_name": ""},
                "meta": {"author": "", "title": ""},
                "summary": "",
                "raw_text": "",
                "chunks": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _set_tmp_config(monkeypatch, *, base_path: Path) -> Path:
    cfg_path = base_path / "setup.yaml"
    # YAML double-quoted strings treat backslashes as escapes on Windows, so normalize to forward slashes.
    base_str = str(base_path).replace("\\", "/")
    cfg_path.write_text(
        "\n".join(
            [
                f'BASE_PATH: "{base_str}"',
                "Story_input: data/stories",
                "Chroma_path: data/chroma_db",
                "Chroma_collection_name: StoryForgeRag_v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STORYFORGE_CONFIG_PATH", str(cfg_path))
    return cfg_path


def test_create_story_records_writes_editable_json(monkeypatch, tmp_path: Path):
    _set_tmp_config(monkeypatch, base_path=tmp_path)

    stories_dir = tmp_path / "data" / "stories"
    stories_dir.mkdir(parents=True, exist_ok=True)
    (stories_dir / "Demo.txt").write_text(
        "This is a demo story.\n\nIt has two paragraphs, enough to chunk deterministically.\n" * 40,
        encoding="utf-8",
    )

    tpl_path = tmp_path / "data" / "ingest" / "TEMPLATE__by_title_value__series.json"
    _write_series_template(tpl_path)

    res = create_story_records(base_path=tmp_path, template_path=tpl_path, overwrite=True, max_chars=600, overlap_chars=80)
    assert res["created"] == 1

    out_path = tmp_path / "data" / "story_json" / "Demo.json"
    assert out_path.exists()

    rec = json.loads(out_path.read_text(encoding="utf-8"))
    assert rec["id"] == "Demo"
    assert rec["raw_text"]
    assert isinstance(rec["chunks"], list) and rec["chunks"]
    assert rec["chunks"][0]["chunk_id"].startswith("Demo_chunk_")
    assert rec["chunks"][0]["text"].strip()


def test_records_to_ingest_manifest_writes_jsonl(monkeypatch, tmp_path: Path):
    _set_tmp_config(monkeypatch, base_path=tmp_path)

    # Create one record file (simulate Step 1 output)
    records_dir = tmp_path / "data" / "story_json"
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / "Demo.json").write_text(
        json.dumps(
            {
                "id": "Demo",
                "Is_series": True,
                "meta": {"author": "A", "title": "Demo"},
                "chapter": {"chapter_id": "ch1", "chapter_number": 1, "chapter_name": "Start"},
                "series": {"id": "s1", "series_name": "Series"},
                "volume": {"volume_id": "v1", "volume_name": "Vol", "volume_number": 1},
                "chunks": [
                    {"chunk_id": "Demo_chunk_1", "text": "Some text.", "section": "setup"},
                    {"chunk_id": "Demo_chunk_2", "text": "More text.", "section": "climax"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    res = records_to_ingest_manifest(base_path=tmp_path)
    assert res["written"] == 2

    out_path = tmp_path / "data" / "ingest" / "ingest_manifest.jsonl"
    assert out_path.exists()

    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2

    row = json.loads(lines[0])
    assert row["id"] == "Demo_chunk_1"
    assert row["text"]
    md = row["metadata"]
    assert md["Title"] == "Demo"
    assert md["chunk_id"] == "Demo_chunk_1"
    assert md["query_type"] == "content"
    assert md["Is_series"] is True
    assert "meta_json" in md and md["meta_json"].startswith("{")
    assert "chapter_json" in md and md["chapter_json"].startswith("{")
    assert "series_json" in md and md["series_json"].startswith("{")
    assert "volume_json" in md and md["volume_json"].startswith("{")


def test_records_to_ingest_manifest_omits_series_fields_when_not_series(monkeypatch, tmp_path: Path):
    _set_tmp_config(monkeypatch, base_path=tmp_path)

    records_dir = tmp_path / "data" / "story_json"
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / "Solo.json").write_text(
        json.dumps(
            {
                "id": "Solo",
                "Is_series": False,
                "meta": {"author": "A", "title": "Solo"},
                "chapter": {"chapter_id": "", "chapter_number": 0, "chapter_name": ""},
                "chunks": [{"chunk_id": "Solo_chunk_1", "text": "One chunk.", "section": ""}],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    records_to_ingest_manifest(base_path=tmp_path)
    out_path = tmp_path / "data" / "ingest" / "ingest_manifest.jsonl"
    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    md = json.loads(lines[0])["metadata"]
    assert md["Is_series"] is False
    assert "series_json" not in md
    assert "volume_json" not in md

import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_story_json_to_manifest_workflow(tmp_path, monkeypatch):
    """
    New workflow test:
    - `.txt` stories -> `data/story_json/<title>.json` (manual stop stage)
    - `data/story_json/*.json` -> `data/ingest/ingest_manifest.jsonl`
    """
    # Arrange: temp project skeleton
    (tmp_path / "data" / "stories").mkdir(parents=True)
    (tmp_path / "data" / "ingest").mkdir(parents=True)
    (tmp_path / "data" / "story_json").mkdir(parents=True)

    (tmp_path / "data" / "stories" / "Amun.txt").write_text("A short story about Amun.", encoding="utf-8")

    # Minimal config expected by scripts via load_config()
    (tmp_path / "setup.yaml").write_text(
        "\n".join(
            [
                f'BASE_PATH: "{tmp_path.as_posix()}"',
                'Story_input: "data/stories"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STORYFORGE_CONFIG_PATH", str(tmp_path / "setup.yaml"))

    # Template file used by prepare_story_records + metadata module
    (tmp_path / "data" / "ingest" / "TEMPLATE__by_title_value__series.json").write_text(
        json.dumps(
            {
                "Is_series": True,
                "id": "id_01",
                "series": {"id": "series_01", "series_name": "Amun Chronicles"},
                "volume": {"volume_id": "vol_01", "volume_name": "Rise", "volume_number": 1},
                "meta": {"author": "", "title": ""},
                "chapter": {"chapter_id": "", "chapter_number": 1, "chapter_name": ""},
                "summary": "",
                "raw_text": "FULL CHAPTER TEXT HERE",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Act: run scripts as modules with cwd=temp
    import runpy
    import os
    import sys

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        sys.path.insert(0, str((Path(cwd) / "src").resolve()))
        sys.path.insert(0, str(tmp_path / "scripts"))

        old_argv = sys.argv
        try:
            sys.argv = ["prepare_story_records.py"]
            runpy.run_path(str(Path(cwd) / "scripts" / "prepare_story_records.py"), run_name="__main__")
            sys.argv = ["records_to_ingest_manifest.py"]
            runpy.run_path(str(Path(cwd) / "scripts" / "records_to_ingest_manifest.py"), run_name="__main__")
        finally:
            sys.argv = old_argv
    finally:
        os.chdir(cwd)

    # Assert: record created
    record_path = tmp_path / "data" / "story_json" / "Amun.json"
    assert record_path.exists()
    record = _read_json(record_path)
    assert "chunks" in record and isinstance(record["chunks"], list) and record["chunks"]

    # Assert: manifest created
    manifest_path = tmp_path / "data" / "ingest" / "ingest_manifest.jsonl"
    assert manifest_path.exists()
    first = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert set(first.keys()) == {"id", "text", "metadata"}
    assert "chunk_id" in first["metadata"]
    assert first["metadata"]["Title"] == "Amun"

