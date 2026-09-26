"""
Read-only cleanup inventory for StoryForge-RAG (never modifies files).

Reports:
  1. Unused exports   -- names in storyforge/rag/__init__.py __all__ with no
                         call sites outside their definition / re-export lines.
  2. Duplicate helpers -- top-level functions defined in more than one file
                         (src/ + scripts/), compared by AST (docstrings
                         ignored): IDENTICAL vs DIFFERENT.
  3. Known look-alike pairs -- helpers with different names but overlapping
                         jobs, with a SAFE / CAREFUL / DO NOT TOUCH label.
  4. No-op aliases     -- module-level `_x = y` re-bindings and their uses.
  5. Script bootstraps -- scripts/*.py that import storyforge without adding
                         src/ to sys.path.
  6. runpy indirection -- library code that executes CLI scripts via runpy.

Usage (repo root):
    .\\.venv\\Scripts\\python.exe scripts\\debug_cleanup_inventory.py
"""
from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [ROOT / "src", ROOT / "scripts", ROOT / "tests"]
EXTRA_FILES = [ROOT / "main.py"]

# Hand-maintained risk labels for pairs a name-based scan can't judge.
KNOWN_PAIRS = [
    (("scripts/push_section_metadata.py", "_as_chroma_metadata"),
     ("src/storyforge/data/records_to_manifest.py", "_as_chroma_metadata"),
     "SAFE", "pure serializer; import the shared one if IDENTICAL"),
    (("scripts/push_section_metadata.py", "_dumps_json"),
     ("src/storyforge/data/records_to_manifest.py", "_dumps_json"),
     "SAFE", "pure serializer; import the shared one if IDENTICAL"),
    (("src/storyforge/vector_store/ingest_stories.py", "_get_embed_model"),
     ("src/storyforge/vector_store/embeddings.py", "get_embed_model"),
     "DO NOT TOUCH", "device policy differs (ingest auto-CUDA vs query-side configurable device); needs an explicit decision"),
    (("src/storyforge/rag/extraction.py", "_hf_token"),
     ("src/storyforge/data/hf_summary.py", "get_hf_token"),
     "DO NOT TOUCH", "token resolution order differs; unifying would change HF facts auth behaviour"),
    (("src/storyforge/rag/generation_backend.py", "strip_thinking_tags"),
     ("src/storyforge/rag/attribution.py", "_strip_think_blocks"),
     "DO NOT TOUCH", "unclosed <think> handling differs; local facts salvage depends on it"),
]
CAREFUL_NAMES = {
    "_chunk_text": "CAREFUL -- chunk boundaries decide chunk_ids + embeddings; move only without changing params",
}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def py_files() -> list[Path]:
    out: list[Path] = []
    for d in SCAN_DIRS:
        if d.exists():
            out += [p for p in d.rglob("*.py") if "__pycache__" not in p.parts]
    out += [p for p in EXTRA_FILES if p.exists()]
    return sorted(set(out))


def parse(p: Path):
    try:
        return ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
    except SyntaxError as e:
        print(f"  [WARN] cannot parse {rel(p)}: {e}")
        return None


def top_level_funcs(tree) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


class _StripAnnotations(ast.NodeTransformer):
    """Drop type annotations so `md: dict` and `md: dict[str, Any]` compare equal."""

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def body_fingerprint(fn) -> str:
    import copy

    fn = _StripAnnotations().visit(copy.deepcopy(fn))
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # ignore docstring
    return ast.dump(fn.args, include_attributes=False) + "|" + "".join(
        ast.dump(s, include_attributes=False) for s in body
    )


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> int:
    files = py_files()
    texts = {p: p.read_text(encoding="utf-8", errors="replace") for p in files}
    trees = {p: parse(p) for p in files}

    # ------------------------------------------------------------------ 1
    section("1. Unused exports (storyforge/rag/__init__.py __all__)")
    init = ROOT / "src/storyforge/rag/__init__.py"
    all_names: list[str] = []
    for node in trees.get(init).body if trees.get(init) else []:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
            all_names = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    for name in all_names:
        word = re.compile(rf"\b{re.escape(name)}\b")
        uses = []
        for p, t in texts.items():
            if p == init:
                continue
            for i, line in enumerate(t.splitlines(), 1):
                s = line.strip()
                if not word.search(line):
                    continue
                if s.startswith(("def ", "class ", "async def ")) or re.fullmatch(rf'["\']{re.escape(name)}["\'],?', s):
                    continue  # definition or an __all__ entry
                uses.append(f"{rel(p)}:{i}")
        label = "UNUSED -> SAFE to remove" if not uses else f"{len(uses)} use(s)"
        print(f"  {name:32s} {label}")
        for u in uses[:4]:
            print(f"      {u}")

    # ------------------------------------------------------------------ 2
    section("2. Same-named top-level helpers in >1 file (src/ + scripts/)")
    defs: dict[str, list[tuple[Path, ast.FunctionDef]]] = defaultdict(list)
    for p, tree in trees.items():
        if tree is None or rel(p).startswith("tests/"):
            continue
        for name, fn in top_level_funcs(tree).items():
            if name in {"main", "__getattr__"}:
                continue
            defs[name].append((p, fn))
    dupes = {n: v for n, v in defs.items() if len(v) > 1}
    if not dupes:
        print("  none")
    for name, locs in sorted(dupes.items()):
        fps = {body_fingerprint(fn) for _, fn in locs}
        verdict = "IDENTICAL" if len(fps) == 1 else "DIFFERENT"
        label = CAREFUL_NAMES.get(name, "SAFE (dedupe by import)" if verdict == "IDENTICAL" else "REVIEW (bodies differ)")
        print(f"  {name:28s} {verdict:9s} {label}")
        for p, fn in locs:
            print(f"      {rel(p)}:{fn.lineno}")

    # ------------------------------------------------------------------ 3
    section("3. Known look-alike pairs (different names / overlapping job)")
    for (fa, na), (fb, nb), label, why in KNOWN_PAIRS:
        ta, tb = trees.get(ROOT / fa), trees.get(ROOT / fb)
        fa_fn = top_level_funcs(ta).get(na) if ta else None
        fb_fn = top_level_funcs(tb).get(nb) if tb else None
        if not fa_fn or not fb_fn:
            print(f"  [{label}] {fa}::{na} / {fb}::{nb} -- one side not found (already cleaned up?)")
            continue
        same = "IDENTICAL" if body_fingerprint(fa_fn) == body_fingerprint(fb_fn) else "DIFFERENT"
        print(f"  [{label}] {same}")
        print(f"      {fa}:{fa_fn.lineno}  {na}")
        print(f"      {fb}:{fb_fn.lineno}  {nb}")
        print(f"      why: {why}")

    # ------------------------------------------------------------------ 4
    section("4. No-op module-level aliases (`_x = y`)")
    found = False
    for p, tree in trees.items():
        if tree is None or rel(p).startswith("tests/"):
            continue
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Name) and node.targets[0].id.startswith("_")
                    and not node.targets[0].id.startswith("__")):
                alias, real = node.targets[0].id, node.value.id
                word = re.compile(rf"\b{re.escape(alias)}\b")
                n_uses = sum(len(word.findall(t)) for t in texts.values()) - 1
                # A real cross-module use names this module too (import or monkeypatch string);
                # a same-named alias in an unrelated file doesn't count.
                parts = p.relative_to(ROOT / "src").with_suffix("").parts if rel(p).startswith("src/") else ()
                mod = ".".join(parts)
                importers = sorted(
                    rel(q) for q, t in texts.items() if q != p and mod and mod in t and word.search(t)
                )
                if importers:
                    label = "KEEP (used by other modules: " + ", ".join(importers) + ")"
                else:
                    label = "SAFE (local only)"
                print(f"  {rel(p)}:{node.lineno}  {alias} = {real}   ({n_uses} other reference(s))  -> {label}")
                found = True
    if not found:
        print("  none")

    # ------------------------------------------------------------------ 5
    section("5. scripts/*.py importing storyforge without adding src/ to sys.path")
    found = False
    for p in sorted((ROOT / "scripts").glob("*.py")):
        t = texts.get(p, "")
        if re.search(r"^\s*(from|import)\s+storyforge", t, re.M) and "sys.path.insert" not in t:
            print(f"  SAFE  {rel(p)} -- add the standard REPO_ROOT/src sys.path bootstrap")
            found = True
    if not found:
        print("  none")

    # ------------------------------------------------------------------ 6
    section("6. runpy indirection in library code")
    found = False
    for p, t in texts.items():
        if rel(p).startswith("src/") and "runpy.run_path" in t:
            lines = [i for i, line in enumerate(t.splitlines(), 1) if "runpy.run_path" in line]
            print(f"  CAREFUL  {rel(p)}:{','.join(map(str, lines))} -- replace with direct library calls only if CLI flags map 1:1")
            found = True
    if not found:
        print("  none")

    print("\n(read-only report; nothing was changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
