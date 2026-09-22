"""
measure_generation_length.py
-----------------------------
Phase 2 measurement tool: run a small batch of queries through the real
agentic generation path (Orchestrator.generate_story_agentic -- the same
code /orchestration/run_step and /orchestration/run_pipeline call for
"4_generate_story_agentic") and log, per query:

  - requested length target (name, target_words, min_words from LengthProfile)
  - actual word count of the accepted/best draft
  - whether it was ACCEPTed by the agentic loop, and the stop_reason
  - iterations run, and each iteration's action / word_count / faithfulness
  - whether the final draft is under LengthProfile.min_words

This calls the orchestrator directly (no HTTP server needed) so it gets the
full iteration history: neither /create-eval/story_generate (always the
non-agentic 3-step path) nor /orchestration/run_step /run_pipeline (agentic,
but only logs iterations server-side and returns a thin {success, steps_done}
response) exposes this data over HTTP today.

Requires a real environment: Ollama running (docker compose up -d) with the
configured model pulled, sentence-transformers/chromadb installed, an
ingested Chroma collection, and (with Evaluation_mode: "api") a reachable HF
token in setup.yaml -- i.e. run this on your machine, not in a sandbox.

Usage (from project root, with venv active):
    py scripts/measure_generation_length.py
    py scripts/measure_generation_length.py --length 13min --mode fast
    py scripts/measure_generation_length.py --queries-file my_queries.json --limit 3
    py scripts/measure_generation_length.py --mode thinking --length long   # tests empty-draft recovery path too
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyforge.config.config import load_config  # noqa: E402
from storyforge.orchestrator.orchestrator import Orchestrator  # noqa: E402
from storyforge.rag.generative_ai import Gen_mode, StoryType  # noqa: E402

# Small, varied default batch: distinct corpus-adjacent themes so retrieval has
# something real to ground on, without hand-picking exact titles (that's what
# tests/fixtures/retrieval_eval_cases.example.json is for).
DEFAULT_QUERIES = [
    "A scholar investigates a cult that worships something sleeping beneath the sea",
    "A scientist experiments with bringing the dead back to life at a university",
    "An orphaned huntress is raised by a tribe after losing her parents to a beast",
    "A doctor is called out in a blizzard for an urgent house call",
    "A goblin explosives maker builds a reputation in a bustling trade city's markets",
    "A man is besieged by whispering creatures from another world in his farmhouse",
    "A grieving lawyer investigates his old friend's connection to a violent alter ego",
    "A young warrior trains to become as strong as the heroes of legend",
]


def _run_one(orchestrator: Orchestrator, query: str, *, mode: Gen_mode, length: str) -> dict[str, Any]:
    started = time.monotonic()
    res = orchestrator.generate_story_agentic(
        query=query,
        save=False,
        mode=mode,
        story_type=StoryType.MIX,
        debug=False,
        length=length,
    )
    elapsed = round(time.monotonic() - started, 1)

    if not res.get("success"):
        return {
            "query": query,
            "success": False,
            "error": res.get("error") or "generation failed (empty draft)",
            "elapsed_seconds": elapsed,
        }

    length_info = (res.get("gen_params") or {}).get("length") or {}
    target_words = int(length_info.get("target_words") or 0)
    min_words = int(length_info.get("min_words") or 0)
    content = res.get("content") or ""
    actual_words = len(content.split())
    iterations = res.get("iterations") or []

    return {
        "query": query,
        "success": True,
        "elapsed_seconds": elapsed,
        "length_name": length_info.get("name"),
        "target_words": target_words,
        "min_words": min_words,
        "actual_words": actual_words,
        "under_min_words": actual_words < min_words if min_words else None,
        "accepted": bool(res.get("accepted")),
        "stop_reason": res.get("stop_reason"),
        "iterations_run": res.get("iterations_run"),
        "final_average": res.get("final_average"),
        "final_faithfulness": (iterations[-1].get("faithfulness") if iterations else None),
        "iterations": [
            {
                "iteration": it.get("iteration"),
                "action": it.get("action"),
                "word_count": it.get("word_count"),
                "average_score": it.get("average_score"),
                "faithfulness": it.get("faithfulness"),
                "completeness_ok": it.get("completeness_ok"),
                "facts_count": it.get("facts_count"),
                "missing_sections": it.get("missing_sections"),
                "reasons": it.get("reasons"),
            }
            for it in iterations
        ],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("success")]
    total = len(records)
    n_ok = len(ok)
    accepted = [r for r in ok if r.get("accepted")]
    under_min = [r for r in ok if r.get("under_min_words")]
    return {
        "total_queries": total,
        "generation_succeeded": n_ok,
        "generation_failed": total - n_ok,
        "accept_rate": round(len(accepted) / n_ok, 2) if n_ok else None,
        "under_min_words_rate": round(len(under_min) / n_ok, 2) if n_ok else None,
        "avg_iterations_run": (
            round(sum(r.get("iterations_run") or 0 for r in ok) / n_ok, 2) if n_ok else None
        ),
        "avg_actual_words": round(sum(r.get("actual_words") or 0 for r in ok) / n_ok, 1) if n_ok else None,
        "avg_target_words": round(sum(r.get("target_words") or 0 for r in ok) / n_ok, 1) if n_ok else None,
        "avg_elapsed_seconds": round(sum(r.get("elapsed_seconds") or 0 for r in records) / total, 1) if total else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure agentic-loop length accept rate over a small query batch.")
    parser.add_argument("--queries-file", default=None, help="JSON file: a list of query strings.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N queries.")
    parser.add_argument("--mode", default="fast", choices=["fast", "thinking"], help="Generation mode.")
    parser.add_argument(
        "--length",
        default="long",
        help='Length target: preset ("long"), duration ("13min"), or word count ("1800").',
    )
    parser.add_argument(
        "--output",
        default="Evaluation/generation_eval_report.json",
        help="Where to write the JSON report (gitignored under Evaluation/).",
    )
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
    if args.limit:
        queries = queries[: args.limit]

    mode = Gen_mode.THINKING if args.mode == "thinking" else Gen_mode.FAST
    load_config()  # fail fast on a broken config before spending any GPU time
    orchestrator = Orchestrator()

    records: list[dict[str, Any]] = []
    for i, q in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] mode={args.mode} length={args.length} :: {q[:70]}")
        rec = _run_one(orchestrator, q, mode=mode, length=args.length)
        records.append(rec)
        if rec.get("success"):
            print(
                f"    words={rec['actual_words']}/{rec['target_words']} "
                f"(min {rec['min_words']}) accepted={rec['accepted']} "
                f"stop={rec['stop_reason']} iters={rec['iterations_run']} "
                f"faithfulness={rec['final_faithfulness']} "
                f"({rec['elapsed_seconds']}s)"
            )
        else:
            print(f"    FAILED: {rec.get('error')} ({rec['elapsed_seconds']}s)")

    report = {
        "mode": args.mode,
        "length": args.length,
        "summary": summarize(records),
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote report to {out_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
