"""API contract tests for StoryForge FastAPI endpoints.

Uses FastAPI TestClient with monkeypatched RAG internals so no GPU, Chroma,
HF API, or Ollama Docker instance is required at test time.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# App fixture — mount only the orchestration router under test
# ---------------------------------------------------------------------------

@pytest.fixture()
def app():
    from storyforge.api.orchestration_routes import orchestration_router
    _app = FastAPI()
    _app.include_router(orchestration_router)
    return _app


@pytest.fixture()
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------

def _fake_docs(n: int = 3) -> list[Document]:
    return [
        Document(
            page_content=f"Chunk {i}: Amun walked through the desert.",
            metadata={"chunk_id": f"ch_{i}", "Title": "Amun Chronicles"},
        )
        for i in range(n)
    ]


def _fake_parsed_facts():
    """Minimal ParsedFacts stub."""
    from storyforge.rag.attribution import ParsedFacts
    return ParsedFacts(facts=(), raw={})


# ---------------------------------------------------------------------------
# /orchestration/status
# ---------------------------------------------------------------------------

def test_status_returns_200(client):
    resp = client.get("/orchestration/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["type"] == "orchestration"


# ---------------------------------------------------------------------------
# /orchestration/run_step — validation
# ---------------------------------------------------------------------------

def test_run_step_rejects_invalid_step(client):
    resp = client.post(
        "/orchestration/run_step",
        json={"step": "99_invalid_step", "title": "Test"},
    )
    assert resp.status_code == 400
    assert "Invalid step" in resp.json()["detail"]


def test_run_step_accepts_valid_step(client):
    """Ensure valid steps are accepted (orchestrator itself is mocked)."""
    with patch(
        "storyforge.api.orchestration_routes.orchestrator.run_pipeline",
        return_value={"success": True, "message": "ok", "steps_done": ["3_ingest_stories"]},
    ):
        resp = client.post(
            "/orchestration/run_step",
            json={"step": "3_ingest_stories", "title": "manual_step"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["step"] == "3_ingest_stories"


# ---------------------------------------------------------------------------
# /orchestration/generate_stream
# ---------------------------------------------------------------------------

@pytest.fixture()
def _mock_rag_for_stream(monkeypatch):
    """Patch retrieve_docs, extract_grounded_facts, and ChatOllama.astream."""
    # Patch retrieve_docs in the orchestration_routes module namespace.
    monkeypatch.setattr(
        "storyforge.api.orchestration_routes.retrieve_docs",
        lambda query, cfg, **kwargs: _fake_docs(3),
    )
    # Patch extract_grounded_facts similarly.
    monkeypatch.setattr(
        "storyforge.api.orchestration_routes.extract_grounded_facts",
        lambda query, chunks, cfg: ('{"facts": []}', _fake_parsed_facts()),
    )
    # Patch ChatOllama so no Ollama server is needed.
    fake_llm = MagicMock()

    async def _fake_astream(messages):
        """Yield a few fake tokens."""
        for token in ["Once ", "upon ", "a time."]:
            chunk = MagicMock()
            chunk.content = token
            yield chunk

    fake_llm.astream = _fake_astream

    monkeypatch.setattr(
        "storyforge.api.orchestration_routes.ChatOllama",
        lambda **_kwargs: fake_llm,
        raising=False,
    )


def test_generate_stream_returns_sse_content_type(client, _mock_rag_for_stream):
    resp = client.post(
        "/orchestration/generate_stream",
        json={"query": "A warrior monk's journey", "mode": "fast", "n_stories": 2},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_generate_stream_emits_step_events(client, _mock_rag_for_stream):
    resp = client.post(
        "/orchestration/generate_stream",
        json={"query": "A warrior monk's journey", "mode": "fast"},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200

    raw = resp.text
    events = [
        json.loads(line[len("data: "):])
        for line in raw.splitlines()
        if line.startswith("data: ") and not line.startswith("data: [DONE]")
    ]
    steps_seen = {(e.get("step"), e.get("status")) for e in events}

    assert ("retrieve", "start") in steps_seen
    assert ("retrieve", "done") in steps_seen
    assert ("extract", "start") in steps_seen
    assert ("extract", "done") in steps_seen
    assert ("generate", "start") in steps_seen
    assert ("generate", "done") in steps_seen


def test_generate_stream_emits_token_events(client, _mock_rag_for_stream):
    resp = client.post(
        "/orchestration/generate_stream",
        json={"query": "A warrior monk's journey", "mode": "fast"},
        headers={"Accept": "text/event-stream"},
    )
    raw = resp.text
    token_events = [
        json.loads(line[len("data: "):])
        for line in raw.splitlines()
        if line.startswith("data: ")
        and not line.startswith("data: [DONE]")
        and '"token"' in line
    ]
    assert len(token_events) >= 1
    # Tokens together should spell out the fake story.
    story = "".join(e["token"] for e in token_events)
    assert "Once" in story


def test_generate_stream_ends_with_done_sentinel(client, _mock_rag_for_stream):
    resp = client.post(
        "/orchestration/generate_stream",
        json={"query": "A warrior monk's journey", "mode": "fast"},
    )
    assert "data: [DONE]" in resp.text


def test_generate_stream_rejects_empty_query(client):
    resp = client.post(
        "/orchestration/generate_stream",
        json={"query": ""},
    )
    assert resp.status_code == 422  # Pydantic min_length=1 validation error
