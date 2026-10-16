"""Group a measure_generation_length report by source-story length, to find the best story size to ingest.

Needs Evaluation/length_sample_map.json (query -> target title, words, style) and a report written by
`measure_generation_length.py --queries-file Evaluation/length_sample_queries.json`. Read-only.

    .\\.venv\\Scripts\\python.exe scripts/analyze_length_buckets.py [report.json]
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Evaluation" / "generation_eval_report_arc.json"
map_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "Evaluation" / "length_sample_map.json"  # e.g. Evaluation/wiki_test_map.json
sample = {m["query"]: m for m in json.loads(map_path.read_text(encoding="utf-8"))}
BUCKETS = [(0, 600), (600, 1000), (1000, 1500), (1500, 2600), (2600, 6000), (6000, 10**9)]


def bucket(words: int) -> tuple[int, str]:
    lo, hi = next(b for b in BUCKETS if b[0] <= words < b[1])
    return lo, (f"{lo}-{hi}" if hi < 10**9 else f">{lo}")


groups = defaultdict(list)
for r in json.loads(report.read_text(encoding="utf-8"))["records"]:
    m = sample.get(r["query"])
    if not m:
        continue
    facts = r.get("grounded_facts") or []
    own = sum(1 for f in facts if f["source_chunk_ids"] and re.sub(r"_chunk_\d+$", "", f["source_chunk_ids"][0]) == m["title"])
    lo, label = bucket(m["words"])
    groups[(m["style"], lo, label)].append(
        (bool(r["accepted"]), r["final_faithfulness"] or 0, r["final_average"] or 0, own / (len(facts) or 1), r["iterations_run"])
    )

print(f"{'style':8} {'words':10} {'n':>3} {'accept':>7} {'faith':>6} {'judge':>6} {'target-fact share':>18} {'iters':>6}")
for (style, _, label), v in sorted(groups.items()):
    print(
        f"{style:8} {label:10} {len(v):3} {sum(x[0] for x in v) / len(v):7.2f} {st.mean(x[1] for x in v):6.1f} "
        f"{st.mean(x[2] for x in v):6.2f} {st.mean(x[3] for x in v):18.2f} {st.mean(x[4] for x in v):6.2f}"
    )
