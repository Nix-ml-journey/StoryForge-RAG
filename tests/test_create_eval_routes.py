import importlib
import sys
import types
from enum import Enum

from fastapi import FastAPI
from fastapi.testclient import TestClient


class _GenMode(str, Enum):
    FAST = "FAST"
    THINKING = "THINKING"
    SHORT = "SHORT"


class _FakeOrchestrator:
    def generate_story(self, **_kwargs):
        return {
            "success": True,
            "content": "A generated story.",
            "timestamp": "2026-01-01T00:00:00",
            "saved": False,
            "saved_path": None,
            "mode": "FAST",
            "gen_params": {"three_layer": True},
        }

    def evaluate_story(self, **_kwargs):
        return {
            "success": True,
            "average_score": 8.0,
            "scores": {"overall": {"score": 8}},
            "summary": "Good story.",
            "suggestions": ["Tighten the ending."],
            "metadata": {"provider": "test"},
            "conclusion": "Readable and coherent.",
        }

    def generate_summary(self, **_kwargs):
        return {
            "success": True,
            "summary": "Short summary.",
            "timestamp": "2026-01-01T00:00:00",
            "saved": False,
            "saved_path": None,
        }

    def evaluate_summary(self, **_kwargs):
        return self.evaluate_story()

    def evaluate_story_file(self, **_kwargs):
        return self.evaluate_story()


def _load_create_eval_routes(monkeypatch):
    sys.modules.pop("storyforge.api.create_eval_routes", None)

    fake_gen = types.ModuleType("storyforge.rag.generative_ai")
    fake_gen.Gen_mode = _GenMode
    fake_gen.parse_gen_mode = lambda mode: mode or "fast"
    fake_gen.parse_story_type = lambda story_type: story_type or "mix"

    fake_orchestrator = types.ModuleType("storyforge.orchestrator.orchestrator")
    fake_orchestrator.Orchestrator = _FakeOrchestrator

    monkeypatch.setitem(sys.modules, "storyforge.rag.generative_ai", fake_gen)
    monkeypatch.setitem(sys.modules, "storyforge.orchestrator.orchestrator", fake_orchestrator)

    return importlib.import_module("storyforge.api.create_eval_routes")


def _client_for(module):
    app = FastAPI()
    app.include_router(module.create_eval_router)
    return TestClient(app)


def test_story_generate_route_returns_generation_payload(monkeypatch):
    module = _load_create_eval_routes(monkeypatch)
    client = _client_for(module)

    response = client.post("/create-eval/story_generate", json={"query": "A child finds a door"})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["content"] == "A generated story."
    assert data["mode"] == "FAST"


def test_story_evaluate_route_returns_scores(monkeypatch):
    module = _load_create_eval_routes(monkeypatch)
    client = _client_for(module)

    response = client.post(
        "/create-eval/story_evaluate",
        json={"story": "Once upon a time, a child solved a problem and went home."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["average_score"] == 8.0
    assert data["suggestions"] == ["Tighten the ending."]


def test_status_route_is_available(monkeypatch):
    module = _load_create_eval_routes(monkeypatch)
    client = _client_for(module)

    response = client.get("/create-eval/status")

    assert response.status_code == 200
    assert response.json()["type"] == "create_eval"
