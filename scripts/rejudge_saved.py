"""
rejudge_saved.py
----------------
Separate judge noise from writer variance, using stories saved by measure_generation_length.py.

The same saved story is scored several times by the configured judge, with the full facts block and with
the facts capped to the first N. If one story swings (for example 9, 4, 4) the judge is the noise source;
if each story is stable but different stories score differently, the writer varies. If capping the facts
raises faithfulness for the same story, a long facts block is confusing the 4B judge.

Needs Ollama running with the judge model (same as the generation run). No retrieval or generation.

Usage (project root, venv active):
    .\\.venv\\Scripts\\python.exe scripts/rejudge_saved.py
    .\\.venv\\Scripts\\python.exe scripts/rejudge_saved.py --report Evaluation/generation_eval_report_arc.json --repeats 3 --cap 8
    .\\.venv\\Scripts\\python.exe scripts/rejudge_saved.py --limit 6      # quick look
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyforge.evaluation import evaluation as eval_mod  # noqa: E402
from storyforge.rag.agentic_loop import criterion_score  # noqa: E402
from storyforge.rag.attribution import format_facts_for_prompt  # noqa: E402


def _facts(record: dict[str, Any], cap: int | None) -> str:
    raw = list(record.get("grounded_facts") or [])
    if not raw:  # older reports only kept the fact text
        raw = [{"type": "fact", "fact": t, "source_chunk_ids": (), "quote": ""} for t in record.get("facts") or []]
    items = [
        types.SimpleNamespace(
            type=f.get("type", "fact"), fact=f.get("fact", ""),
            source_chunk_ids=tuple(f.get("source_chunk_ids") or ()), quote=f.get("quote") or "",
        )
        for f in raw
    ]
    if cap:
        items = items[:cap]
    return format_facts_for_prompt(types.SimpleNamespace(facts=tuple(items)))


def _scores(model: Any, story: str, facts: str, repeats: int) -> list[float | None]:
    return [criterion_score(eval_mod.evaluate_story_text(model, story, facts=facts), "faithfulness") for _ in range(repeats)]


def _fmt(values: list[float | None]) -> str:
    nums = [v for v in values if v is not None]
    spread = f" (spread {max(nums) - min(nums):.0f})" if len(nums) > 1 else ""
    return f"{values}{spread}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="Evaluation/generation_eval_report_arc.json")
    ap.add_argument("--repeats", type=int, default=3, help="judge calls per story and variant")
    ap.add_argument("--cap", type=int, default=8, help="facts kept in the capped variant")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records")
    ap.add_argument("--output", default="Evaluation/rejudge_saved.json")
    args = ap.parse_args()

    records = [r for r in json.loads(Path(args.report).read_text(encoding="utf-8"))["records"] if r.get("story")]
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("No saved stories in the report; rerun measure_generation_length.py first.")
        return 1
    model = eval_mod.evaluate_model()

    rows = []
    for r in records:
        full = _scores(model, r["story"], _facts(r, None), args.repeats)
        capped = _scores(model, r["story"], _facts(r, args.cap), args.repeats)
        n = len(r.get("grounded_facts") or r.get("facts") or [])
        rows.append({"query": r["query"], "run": r.get("run"), "n_facts": n, "reported": r.get("final_faithfulness"),
                     "full": full, "capped": capped})
        print(f"{r['query'][:34]:<34} run{r.get('run')} facts={n:<3} reported={r.get('final_faithfulness')}"
              f"\n    full  : {_fmt(full)}\n    cap {args.cap:<2}: {_fmt(capped)}", flush=True)

    def spread(key: str) -> float:
        out = [max(v) - min(v) for v in ([x for x in row[key] if x is not None] for row in rows) if len(v) > 1]
        return round(statistics.mean(out), 2) if out else float("nan")

    def mean(key: str) -> float:
        vals = [x for row in rows for x in row[key] if x is not None]
        return round(statistics.mean(vals), 2) if vals else float("nan")

    print(f"\nMean faithfulness  full={mean('full')}  capped={mean('capped')}")
    print(f"Mean spread per story (judge noise)  full={spread('full')}  capped={spread('capped')}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
