from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Ensure `src/` is on sys.path so `import storyforge...` works in tests.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Heavy-dependency stubs
#
# Test files that need to import storyforge.rag.* without a GPU, Chroma, or
# HF/Ollama install can request the `stub_heavy_deps` fixture.  It registers
# lightweight stand-ins for all GPU/ML packages before the first import and
# tears them down after the test (by restoring sys.modules to its prior state).
#
# Usage:
#   def test_something(stub_heavy_deps):
#       from storyforge.rag.extraction import _hf_chat_extract_json
#       ...
# ---------------------------------------------------------------------------

_HEAVY_MODS = [
    "langchain_ollama",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.documents",
    "langchain_core.embeddings",
    "langchain_huggingface",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_openai",
    "langchain_chroma",
    "transformers",
    "torch",
    "huggingface_hub",
    "rank_bm25",
    "sentence_transformers",
    "chromadb",
    "tqdm",
]


def _build_stubs() -> dict[str, types.ModuleType]:
    """Return a dict of module-name → stub module for all heavy deps."""
    stubs: dict[str, types.ModuleType] = {}
    for name in _HEAVY_MODS:
        if name not in sys.modules:
            stubs[name] = types.ModuleType(name)

    # Populate the specific attributes that storyforge source files import by name.
    def _set(mod_name: str, **attrs: object) -> None:
        m = stubs.get(mod_name) or sys.modules.get(mod_name)
        if m is not None:
            for k, v in attrs.items():
                setattr(m, k, v)

    # langchain_core.messages
    _set(
        "langchain_core.messages",
        HumanMessage=type("HumanMessage", (), {"__init__": lambda self, content: None}),
        SystemMessage=type("SystemMessage", (), {"__init__": lambda self, content: None}),
    )

    # langchain_core.documents
    _set(
        "langchain_core.documents",
        Document=type(
            "Document",
            (),
            {"__init__": lambda self, page_content="", metadata=None: None},
        ),
    )

    # langchain_chroma
    _set("langchain_chroma", Chroma=type("Chroma", (), {}))

    # langchain_huggingface
    _set(
        "langchain_huggingface",
        HuggingFaceEmbeddings=type(
            "HuggingFaceEmbeddings", (), {"__init__": lambda self, **k: None}
        ),
        HuggingFacePipeline=type(
            "HuggingFacePipeline", (), {"__init__": lambda self, **k: None}
        ),
    )

    # huggingface_hub
    _set(
        "huggingface_hub",
        InferenceClient=type("InferenceClient", (), {"__init__": lambda self, token=None: None}),
    )

    return stubs


@pytest.fixture()
def stub_heavy_deps():
    """Register lightweight stubs for all GPU/ML packages.

    Safe to use in any test that imports storyforge.rag.* without a real GPU,
    Chroma, or HF/Ollama environment.  Stubs are inserted into sys.modules for
    the duration of the test and removed afterwards (restoring the prior state).
    """
    stubs = _build_stubs()
    # Insert all stubs that aren't already registered.
    inserted: list[str] = []
    for name, mod in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = mod
            inserted.append(name)
    yield
    # Clean up only the modules WE inserted (don't remove real installs).
    for name in inserted:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Shared factory helpers (importable by test files as plain functions)
# ---------------------------------------------------------------------------

def fake_hf_response(content: str):
    """Build a minimal InferenceClient chat_completion response stub."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def fake_docs(n: int = 3):
    """Return n minimal LangChain Document stubs for use in API / retrieval tests."""
    from langchain_core.documents import Document  # already stubbed in tests

    return [
        Document(
            page_content=f"Chunk {i}: A hero walked through the ancient forest.",
            metadata={"chunk_id": f"ch_{i}", "Title": "Test Story"},
        )
        for i in range(n)
    ]
