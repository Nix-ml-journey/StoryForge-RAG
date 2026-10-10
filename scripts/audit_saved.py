"""
audit_saved.py
--------------
Test the claim audit (evaluation.audit_story_claims) on stories saved by measure_generation_length.py,
before it goes anywhere near the loop.

The judge lists the story sentences the facts do not support. Questions this answers:
  1. Does the count of flagged sentences track the scalar faithfulness (low faith -> more flagged)?
  2. Is the count stable when the same story is audited twice?
  3. Are the flagged sentences really unsupported? Read the printed samples; that part is a human check.

Needs Ollama with the judge model. No retrieval or generation.

Usage (project root, venv active):
    .\\.venv\\Scripts\\python.exe scripts/audit_saved.py
    .\\.venv\\Scripts\\python.exe scripts/audit_saved.py --limit 6 --repeats 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rejudge_saved import _facts  # noqa: E402  (same facts block the loop's judge sees)

from storyforge.evaluation import evaluation as eval_mod  # noqa: E402


def _norm(text: str) -> str:
    return " ".join(text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'").split()).lower().strip(" .\"'")


def _in_story(sentence: str, story: str) -> bool:
    """A flagged sentence is only real if it appears in the story (whitespace and quote style ignored)."""
    return _norm(sentence) in _norm(story)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="Evaluation/generation_eval_report_arc.json")
    ap.add_argument("--repeats", type=int, default=3, help="audits per story (3 lets two sentence-intersections be compared)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default="Evaluation/audit_saved.json")
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
        runs = [eval_mod.audit_story_claims(model, r["story"], _facts(r, None)) for _ in range(args.repeats)]
        counts = [None if x is None else len(x) for x in runs]
        first = next((x for x in runs if x), [])
        flat = [x for run in runs for x in (run or [])]
        verbatim = sum(_in_story(x["sentence"], r["story"]) for x in flat)
        keys = [{_norm(x["sentence"]) for x in (run or [])} for run in runs]
        both = [len(keys[i] & keys[i + 1]) for i in range(len(keys) - 1)]  # sentences flagged in two audits in a row
        rows.append({"query": r["query"], "run": r.get("run"), "faithfulness": r.get("final_faithfulness"),
                     "counts": counts, "both": both, "flagged": first[:3], "listed": len(flat), "verbatim": verbatim,
                     "runs": runs})
        print(f"{r['query'][:34]:<34} run{r.get('run')} faith={r.get('final_faithfulness')} flagged={counts}", flush=True)

    ok = [x for x in rows if x["faithfulness"] is not None and x["counts"][0] is not None]
    lo = [x["counts"][0] for x in ok if x["faithfulness"] < 6]
    hi = [x["counts"][0] for x in ok if x["faithfulness"] >= 6]
    if lo and hi:
        print(f"\nFlagged sentences: faithfulness < 6 -> {statistics.mean(lo):.1f} (n={len(lo)}), >= 6 -> {statistics.mean(hi):.1f} (n={len(hi)})")
    if len(ok) > 2:
        try:
            print("Correlation (faithfulness vs flagged count):",
                  round(statistics.correlation([x["faithfulness"] for x in ok], [x["counts"][0] for x in ok]), 2))
        except statistics.StatisticsError:
            print("Correlation undefined (a column is constant).")
    same = [x for x in rows if len(set(x["counts"])) == 1]
    print(f"Same count on every repeat: {len(same)}/{len(rows)}; unparseable answers: {sum(c is None for x in rows for c in x['counts'])}")
    listed = sum(x["listed"] for x in rows)
    rate = sum(x["verbatim"] for x in rows) / listed if listed else 0.0
    moved = max((max(c) - min(c) for x in rows if (c := [v for v in x["counts"] if v is not None])), default=0)
    separates = bool(lo and hi and statistics.mean(lo) > statistics.mean(hi) + 1)
    print("\nGo/no-go (ADR-0011):")
    print(f"  listed sentences found verbatim in the story: {rate:.0%} of {listed}  (need >= 80%)  {'PASS' if rate >= 0.8 else 'FAIL'}")
    print(f"  faithfulness < 6 flags clearly more than >= 6 (by > 1):  {'PASS' if separates else 'FAIL'}")
    print(f"  largest change in count between repeats: {moved}  (need <= 1)  {'PASS' if moved <= 1 else 'FAIL'}")
    near = sum(1 for x in rows if len({v for v in x["counts"] if v is not None}) <= 1 or max(x["counts"]) - min(x["counts"]) <= 1)
    print(f"  (stories within 1 between repeats: {near}/{len(rows)})")
    # Keep only sentences flagged in two audits in a row: fewer one-off flags, so a steadier count.
    pairs = [x["both"] for x in rows if len(x["both"]) >= 2]
    if pairs:
        moved_b = max(abs(b[0] - b[1]) for b in pairs)
        lo_b = [x["both"][0] for x in rows if x["faithfulness"] is not None and x["faithfulness"] < 6 and x["both"]]
        hi_b = [x["both"][0] for x in rows if x["faithfulness"] is not None and x["faithfulness"] >= 6 and x["both"]]
        sep_b = bool(lo_b and hi_b and statistics.mean(lo_b) > statistics.mean(hi_b) + 1)
        print("Flagged in two audits in a row (intersection):")
        if lo_b and hi_b:
            print(f"  mean count: faithfulness < 6 -> {statistics.mean(lo_b):.1f}, >= 6 -> {statistics.mean(hi_b):.1f}  {'PASS' if sep_b else 'FAIL'}")
        print(f"  largest change between the two intersections: {moved_b}  (need <= 1)  {'PASS' if moved_b <= 1 else 'FAIL'}")
    print("\nSamples to read (lowest faithfulness first) -- are these really unsupported by the facts?")
    for x in sorted(ok, key=lambda x: x["faithfulness"])[:4]:
        print(f"- {x['query'][:40]} (faith {x['faithfulness']})")
        for f in x["flagged"]:
            print(f"    S{f.get('section')}: {str(f.get('sentence'))[:140]} -- {f.get('why')}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
