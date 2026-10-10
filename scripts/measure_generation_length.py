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
    py scripts/measure_generation_length.py --methods 5w1h arc              # A/B: one report per method + a table
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import storyforge.config.config as _config_mod  # noqa: E402
from storyforge.config.config import load_config  # noqa: E402
from storyforge.orchestrator.orchestrator import Orchestrator  # noqa: E402
from storyforge.rag.agentic_loop import criterion_score  # noqa: E402
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


# Generation methods to compare. Each is a config overlay applied on top of setup.yaml.
METHODS: dict[str, dict[str, str]] = {
    "5w1h": {"Story_generation_method": "5w1h", "Story_generation_mode": "single"},
    "5w1h-sectioned": {"Story_generation_method": "5w1h", "Story_generation_mode": "sectioned"},
    "arc": {"Story_generation_method": "arc"},
    "arc-rules": {"Story_generation_method": "arc", "Agentic_loop_rule_driven": True},  # ADR-0011 Step 2
}
_OVERRIDES: dict[str, Any] = {}


def _install_overrides() -> None:
    """Make every load_config() call (in any module) see ``_OVERRIDES`` on top of setup.yaml."""
    original = _config_mod._load_config_cached
    _config_mod._load_config_cached = lambda *a, **k: {**original(*a, **k), **_OVERRIDES}


def _run_one(orchestrator: Orchestrator, query: str, *, mode: Gen_mode, length: str) -> dict[str, Any]:
    started = time.monotonic()
    res = orchestrator.generate_story_agentic(
        query=query,
        save=False,
        mode=mode,
        story_type=StoryType.MIX,
        debug=True,  # needed for grounded_facts and final_scores in the payload
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
        "story": content,  # kept so rules can be calibrated offline against judge scores
        "facts": [f.get("fact") for f in (res.get("grounded_facts") or [])],
        "grounded_facts": list(res.get("grounded_facts") or []),  # full dicts, so rejudge_saved.py reproduces the judge prompt
        "iterations_run": res.get("iterations_run"),
        "final_average": res.get("final_average"),
        # Both numbers describe the returned (best) draft; the last iteration may be a different draft.
        "final_faithfulness": criterion_score(res.get("final_scores"), "faithfulness"),
        "last_iteration_faithfulness": (iterations[-1].get("faithfulness") if iterations else None),
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
        "avg_judge_average": _mean([r.get("final_average") for r in ok]),
        "avg_judge_faithfulness": _mean([r.get("final_faithfulness") for r in ok]),
    }


def _mean(values: list[Any]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


def summarize_runs(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Mean and spread of the headline metrics over repeated runs (the judge and sampling are noisy)."""
    per_run = [summarize(r) for r in runs]
    out: dict[str, Any] = {"runs": len(per_run)}
    for key in ("accept_rate", "avg_iterations_run", "avg_elapsed_seconds", "avg_judge_average", "avg_judge_faithfulness"):
        vals = [s[key] for s in per_run if s.get(key) is not None]
        if vals:
            out[key] = f"{statistics.mean(vals):.2f} +/- {statistics.pstdev(vals):.2f}"
    out["per_run"] = per_run
    return out


def format_comparison(summaries: dict[str, dict[str, Any]]) -> str:
    """Side-by-side table of the summary metrics, one column per method."""
    keys = list(next(iter(summaries.values())))
    width = max(len(k) for k in keys)
    labels = list(summaries)
    lines = [f"{'metric':<{width}}  " + "  ".join(f"{label:>14}" for label in labels)]
    lines += [f"{k:<{width}}  " + "  ".join(f"{str(summaries[label].get(k)):>14}" for label in labels) for k in keys]
    return "\n".join(lines)


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
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        default=None,
        help="Run the same queries once per generation method and print a comparison table "
        "(default: one run with whatever setup.yaml says).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the whole query set this many times per method and report mean +/- spread "
        "(the 4B judge shifts by about a point between runs, so one run cannot rank methods).",
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
    _install_overrides()
    load_config()  # fail fast on a broken config before spending any GPU time
    orchestrator = Orchestrator()

    base_out = Path(args.output)
    summaries: dict[str, dict[str, Any]] = {}
    for method in args.methods or [None]:
        label = method or "config"
        _OVERRIDES.clear()
        _OVERRIDES.update(METHODS.get(method, {}))
        records: list[dict[str, Any]] = []
        runs: list[list[dict[str, Any]]] = []
        plan = [(r, i, q) for r in range(1, max(1, args.repeat) + 1) for i, q in enumerate(queries, start=1)]
        for r, i, q in plan:
            if i == 1:
                runs.append([])
            print(f"[{label} run {r} {i}/{len(queries)}] mode={args.mode} length={args.length} :: {q[:70]}")
            rec = _run_one(orchestrator, q, mode=mode, length=args.length)
            rec["run"] = r
            records.append(rec)
            runs[-1].append(rec)
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
            "method": label,
            "mode": args.mode,
            "length": args.length,
            "summary": summarize(records),
            "records": records,
        }
        if args.repeat > 1:
            report["repeat_summary"] = summarize_runs(runs)
        out_path = base_out if method is None else base_out.with_name(f"{base_out.stem}_{label}{base_out.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries[label] = report["summary"]
        print(f"\nWrote report to {out_path}")
        print(json.dumps(report["summary"], indent=2))
        if args.repeat > 1:
            print("Mean +/- spread over runs:")
            print(json.dumps({k: v for k, v in report["repeat_summary"].items() if k != "per_run"}, indent=2))

    if len(summaries) > 1:
        print("\n" + format_comparison(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
