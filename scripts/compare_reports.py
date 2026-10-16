"""Side-by-side summary of measure_generation_length reports (also works on *.partial.json).

    .\\.venv\\Scripts\\python.exe scripts/compare_reports.py Evaluation/s2_short_arc-story-first.json Evaluation/s2_medium_arc-story-first.json
"""
import json
import statistics as st
import sys
from pathlib import Path

print(f"{'report':44} {'n':>3} {'accept':>6} {'faith':>5} {'judge':>5} {'words/target':>12} {'sec':>5} {'iters':>5} {'facts':>5}")
for path in sys.argv[1:]:
    records = [r for r in json.loads(Path(path).read_text(encoding="utf-8"))["records"] if r.get("success")]
    if not records:
        print(f"{Path(path).name[:44]:44} no finished stories")
        continue
    mean = lambda key: st.mean((r.get(key) or 0) for r in records)  # noqa: E731
    ratio = st.mean((r.get("actual_words") or 0) / (r.get("target_words") or 1) for r in records)
    facts = st.mean(len(r.get("grounded_facts") or []) for r in records)
    print(
        f"{Path(path).name[:44]:44} {len(records):3} {sum(bool(r.get('accepted')) for r in records) / len(records):6.2f} "
        f"{mean('final_faithfulness'):5.1f} {mean('final_average'):5.1f} {ratio:12.2f} {mean('elapsed_seconds'):5.0f} "
        f"{mean('iterations_run'):5.2f} {facts:5.1f}"
    )
