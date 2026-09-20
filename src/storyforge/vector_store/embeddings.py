"""Shared embedding helpers and the BGE query prefix.

Leaf module (no storyforge imports) so chromadb, ingest, and retrieval can all
use it.

BGE's documented recipe (BAAI/bge-base-en-v1.5) prefixes the *query* side only
for asymmetric retrieval -- passages/documents are embedded with no
instruction prefix. Do not change ``QUERY_PREFIX`` without re-ingesting: it
does not change stored passage vectors, but it does change where queries land
relative to them, so a changed prefix must match whatever convention the
corpus was embedded with.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

LOG = logging.getLogger(__name__)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# CPU default avoids competing with Ollama for GPU VRAM.
_DEFAULT_DEVICE = "cpu"

_EMBED_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def is_bge_model(model_name: str) -> bool:
    """True when the model follows the BGE query/passage instruction convention."""
    return "bge" in str(model_name or "").lower()


def get_embed_model(model_name: str, device: Optional[str] = None):
    """Load and cache a SentenceTransformer. Defaults to CPU; pass device=\"cuda\" if needed."""
    resolved_device = str(device or _DEFAULT_DEVICE)
    key = (str(model_name or ""), resolved_device)
    if key in _EMBED_MODEL_CACHE:
        return _EMBED_MODEL_CACHE[key]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        if resolved_device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    LOG.warning("Embedding device 'cuda' requested but unavailable; using cpu.")
                    resolved_device = "cpu"
                    key = (str(model_name or ""), resolved_device)
            except Exception:
                resolved_device = "cpu"
                key = (str(model_name or ""), resolved_device)

        LOG.info("Loading embedding model %s on %s", key[0], resolved_device)
        model = SentenceTransformer(key[0], device=resolved_device)
        _EMBED_MODEL_CACHE[key] = model
        return model
    except ImportError:
        LOG.warning(
            "sentence-transformers not installed — cannot embed. "
            "Run: pip install sentence-transformers"
        )
        return None
    except Exception as e:
        LOG.warning("Could not load embedding model %r: %s", key[0], e)
        return None


def embed_texts(
    model,
    texts: list[str],
    *,
    prefix: str = "",
) -> Optional[list[list[float]]]:
    """Embed texts with an optional prefix. Returns None on failure."""
    if model is None:
        return None
    try:
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vecs = model.encode(payload, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]
    except Exception as e:
        LOG.warning("Embedding failed: %s", e)
        return None


def embed_query(query: str, model_name: str, device: Optional[str] = None) -> Optional[list[float]]:
    """Embed one search query with the correct prefix for ``model_name``."""
    model = get_embed_model(model_name, device=device)
    if model is None:
        return None
    prefix = QUERY_PREFIX if is_bge_model(model_name) else ""
    vecs = embed_texts(model, [query], prefix=prefix)
    if not vecs:
        return None
    return vecs[0]
