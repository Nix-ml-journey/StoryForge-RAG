"""
Refresh the graphify knowledge graph and name its communities with local Gemma 4.

Steps:
  1. Make sure the Ollama model `gemma4-graphify` exists (pulls gemma4:12b and
     creates it from .graphify/gemma4-graphify.Modelfile if missing).
  2. `graphify update .`   -- code-only AST refresh, no LLM
     (`--full` wipes graphify-out/ and runs `graphify extract . --code-only`).
  3. `graphify label .`    -- community names via the `ollama-gemma4` provider
     in .graphify/providers.json (localhost only, thinking disabled).
  4. Unload the model so the RAG pipeline gets its VRAM back.

Usage (from repo root):
    py scripts/update_graph.py
    py scripts/update_graph.py --force       # after refactors that delete code
    py scripts/update_graph.py --full        # rebuild graphify-out/ from scratch
    py scripts/update_graph.py --no-label    # AST refresh only (Ollama not needed)

Requires the graphify CLI (`uv tool install graphifyy`) and a running Ollama.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVIDER = "ollama-gemma4"
MODEL = "gemma4-graphify"
BASE_MODEL = "gemma4:12b"
MODELFILE = REPO_ROOT / ".graphify" / "gemma4-graphify.Modelfile"
OLLAMA_URL = os.environ.get("STORYFORGE_OLLAMA_URL", "http://localhost:11434")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _ollama_models() -> set[str] | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            tags = json.load(resp)
    except OSError:
        return None
    names = {m["name"] for m in tags.get("models", [])}
    return names | {n.removesuffix(":latest") for n in names}


def _ensure_model(ollama: str) -> bool:
    models = _ollama_models()
    if models is None:
        print(f"[update_graph] Ollama is not reachable at {OLLAMA_URL}; skipping labeling.")
        return False
    if MODEL in models:
        return True
    if BASE_MODEL not in models:
        _run([ollama, "pull", BASE_MODEL])
    _run([ollama, "create", MODEL, "-f", str(MODELFILE)])
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="Overwrite graph.json even if it shrinks")
    ap.add_argument("--full", action="store_true", help="Rebuild graphify-out/ from scratch")
    ap.add_argument("--no-label", action="store_true", help="Skip LLM community naming")
    args = ap.parse_args()

    graphify = shutil.which("graphify")
    if not graphify:
        print("[update_graph] graphify CLI not found. Install: uv tool install graphifyy")
        return 1

    # Only the latest graph is kept: no dated backup folders in graphify-out/.
    env = dict(os.environ, GRAPHIFY_NO_BACKUP="1")
    out_dir = REPO_ROOT / "graphify-out"
    if args.full or not (out_dir / "graph.json").exists():
        shutil.rmtree(out_dir, ignore_errors=True)
        _run([graphify, "extract", ".", "--code-only"], env=env)
    else:
        _run([graphify, "update", ".", *(["--force"] if args.force else [])], env=env)
    if args.no_label:
        return 0

    ollama = shutil.which("ollama")
    if not ollama or not _ensure_model(ollama):
        print("[update_graph] Graph refreshed; community names left unchanged.")
        return 0

    env.update(GRAPHIFY_ALLOW_LOCAL_PROVIDERS="1", OLLAMA_API_KEY="ollama")
    try:
        _run([graphify, "label", ".", f"--backend={PROVIDER}"], env=env)
    finally:
        subprocess.run([ollama, "stop", MODEL], cwd=REPO_ROOT, capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
