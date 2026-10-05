"""Integration tests: real Chroma (in-memory) + real langchain-ollama against a fake Ollama server.

No dependency stubs: retrieval runs on a live ``chromadb.EphemeralClient`` with a
deterministic hashing embedder, and Steps 2-3 plus evaluation talk HTTP to a tiny
fake Ollama (``/api/tags`` + ``/api/chat``). Nothing here needs a GPU, a model
download, or network access.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("langchain_chroma")
pytest.importorskip("langchain_ollama")

import chromadb  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

MODEL = "fake-qwen:9b"

CORPUS = [
    ("Cthulhu_Cult", False, "A scholar studies a secret cult that worships a sleeping god beneath the sea."),
    ("Frankenstein", False, "A young scientist at a university brings a dead body back to life in his laboratory."),
    ("Series_Ch1", True, "The warrior trains daily on the mountain to become as strong as the legendary heroes."),
    ("Doctor_Blizzard", False, "A country doctor is called out in a blizzard for an urgent house call at midnight."),
]


# --------------------------------------------------------------------------- fakes
class _HashEmbeddings(Embeddings):
    """Bag-of-words hashing embedder: deterministic, no model download."""

    DIM = 128

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in re.findall(r"[a-z]+", text.lower()):
            if len(tok) < 3:
                continue
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


_SENTENCE = "The quiet night settled over the old harbour town while the people waited and watched closely."


def _story_text() -> str:
    headers = [
        "[SECTION 1: WHO, WHERE, WHEN (The Setup)]",
        "[SECTION 2: WHAT (The Problem Starts)]",
        "[SECTION 3: TWIST/COMPLICATION (The Challenge)]",
        "[SECTION 4: HOW (The Big Action/Climax)]",
        "[SECTION 5: WHY/OUTCOME (The Moral and Conclusion)]",
    ]
    return "\n\n".join(f"{h}\n{' '.join([_SENTENCE] * 10)}" for h in headers)


class _FakeOllama(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, *_a):  # silence
        pass

    def _send(self, payload: bytes, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._send(json.dumps({"models": [{"name": MODEL}]}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        type(self).calls.append(body)
        prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
        fmt = body.get("format")
        if isinstance(fmt, dict):  # grounded-facts call (schema-constrained)
            ids = re.findall(r"\[CHUNK (\S+) \|", prompt)
            facts = [
                {"type": "who", "fact": f"Fact about {cid}", "source_chunk_ids": [cid], "quote": "a quote"}
                for cid in ids[:4]
            ]
            text = json.dumps({"facts": facts})
        elif fmt == "json":  # evaluation call
            text = json.dumps(
                {"faithfulness": {"score": 9}, "coherence": {"score": 8}, "conclusion": "Solid.", "suggestions": []}
            )
        else:  # story generation
            text = _story_text()
        line = {
            "model": body.get("model", MODEL),
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": "stop",
            "total_duration": 1,
            "load_duration": 1,
            "prompt_eval_count": 1,
            "prompt_eval_duration": 1,
            "eval_count": 1,
            "eval_duration": 1,
        }
        self._send((json.dumps(line) + "\n").encode(), "application/x-ndjson")


@pytest.fixture(scope="module")
def ollama_url():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture()
def cfg(ollama_url):
    return {
        "Generation_provider": "ollama",
        "Generative_model": MODEL,
        "Ollama_base_url": ollama_url,
        "Grounded_facts_provider": "local",
        "Evaluation_mode": "ollama",
        "Reranker_enabled": False,
        "Hybrid_search_enabled": True,
        "Hybrid_bm25_pool": 12,  # feature is off by default; the BM25-recovery test needs it on
        "Story_generation_n_results": 4,
        "Agentic_loop_max_iterations": 2,
    }


@pytest.fixture()
def store(monkeypatch):
    """Live in-memory Chroma wired in as retrieval's vector store."""
    import storyforge.rag.retrieval as retrieval_mod

    vs = Chroma(
        collection_name=f"it_{uuid.uuid4().hex[:8]}",
        embedding_function=_HashEmbeddings(),
        client=chromadb.EphemeralClient(),
    )
    vs.add_texts(
        texts=[c[2] for c in CORPUS],
        metadatas=[{"Title": c[0], "chunk_id": f"{c[0]}__c0", "Is_series": c[1]} for c in CORPUS],
        ids=[f"{c[0]}__c0" for c in CORPUS],
    )
    monkeypatch.setattr(retrieval_mod, "_build_vectorstore", lambda cfg: vs)
    return vs


# --------------------------------------------------------------------------- tests
def test_retrieval_returns_relevant_chunk_from_real_chroma(store, cfg):
    from storyforge.rag.retrieval import retrieve_docs

    docs = retrieve_docs("a scholar investigates a cult worshipping a sleeping god under the sea", cfg)

    assert docs
    assert docs[0].metadata["Title"] == "Cthulhu_Cult"


def test_story_type_filters_on_is_series(store, cfg):
    from storyforge.rag.retrieval import retrieve_docs

    q = "a warrior trains to become strong like the heroes"
    single = retrieve_docs(q, cfg, story_type="single")
    series = retrieve_docs(q, cfg, story_type="series")

    assert single and all(d.metadata["Is_series"] is False for d in single)
    assert series and all(d.metadata["Is_series"] is True for d in series)


def test_global_bm25_recovers_story_dense_retrieval_missed(store, cfg, monkeypatch):
    """Dense search returns only distractors; whole-corpus BM25 must still surface the cult story."""
    import storyforge.rag.retrieval as retrieval_mod

    class DenseMissesTarget:
        def as_retriever(self, search_kwargs=None):
            distractors = store.get(ids=["Frankenstein__c0", "Series_Ch1__c0"], include=["documents", "metadatas"])
            from langchain_core.documents import Document

            docs = [Document(page_content=t, metadata=m) for t, m in zip(distractors["documents"], distractors["metadatas"])]
            return type("R", (), {"invoke": staticmethod(lambda q: docs)})()

        def get(self, **kw):
            return store.get(**kw)

    monkeypatch.setattr(retrieval_mod, "_build_vectorstore", lambda c: DenseMissesTarget())
    docs = retrieval_mod.retrieve_docs("cult worships a sleeping god beneath the sea", cfg)

    assert "Cthulhu_Cult" in {d.metadata["Title"] for d in docs}


def test_ollama_evaluator_scores_story_via_fake_server(cfg, monkeypatch):
    from storyforge.evaluation import evaluation

    monkeypatch.setattr(evaluation, "_cfg", lambda: cfg)
    model = evaluation.evaluate_model()
    assert model["provider"] == "ollama" and model["model"] == MODEL

    result = evaluation.evaluate_story_text(model, _story_text())

    assert result["faithfulness"]["score"] == 9


def test_ollama_evaluator_fails_loudly_when_server_down(cfg, monkeypatch):
    from storyforge.evaluation import evaluation

    monkeypatch.setattr(evaluation, "_cfg", lambda: {**cfg, "Ollama_base_url": "http://127.0.0.1:9"})

    with pytest.raises(RuntimeError, match="Ollama is unreachable"):
        evaluation.evaluate_model()


def test_local_facts_extraction_cites_only_retrieved_chunks(store, cfg):
    from storyforge.rag.extraction import extract_grounded_facts
    from storyforge.rag.retrieval import _docs_to_chunks, retrieve_docs

    chunks = _docs_to_chunks(retrieve_docs("a doctor is called out in a blizzard", cfg))
    _raw, parsed = extract_grounded_facts("a doctor is called out in a blizzard", chunks, cfg)

    known = {c["chunk_id"] for c in chunks}
    assert parsed.facts
    assert all(set(f.source_chunk_ids) <= known for f in parsed.facts)


def test_three_step_pipeline_end_to_end(store, cfg):
    from storyforge.rag.langchain_rag import generate_story_3step_langchain

    out = generate_story_3step_langchain(
        "a scholar investigates a cult that worships something sleeping beneath the sea",
        cfg=cfg,
        length="short",
    )

    assert "[SECTION 1" in out.content and "[SECTION 5" in out.content
    assert out.grounded_facts, "Step 2 should have produced grounded facts"
    assert any(c["title"] == "Cthulhu_Cult" for c in out.retrieval_chunks)


def test_agentic_loop_end_to_end_uses_ollama_judge(store, cfg, monkeypatch):
    from storyforge.evaluation import evaluation
    from storyforge.rag.agentic_loop import run_agentic_story_loop

    monkeypatch.setattr(evaluation, "_cfg", lambda: cfg)
    seen = len(_FakeOllama.calls)  # calls is shared across tests; only inspect this run

    res = run_agentic_story_loop(
        "a scientist brings the dead back to life at a university", cfg=cfg, length="short"
    )

    assert res.content and "[SECTION 5" in res.content
    assert res.iterations and res.iterations[0]["has_eval"] is True
    assert res.final_average > 0
    judge_prompts = [
        str(c["messages"]) for c in _FakeOllama.calls[seen:] if c.get("format") == "json"
    ]
    assert judge_prompts and all("Grounded facts (source of truth)" in p for p in judge_prompts)
