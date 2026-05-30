from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class RetrievalCase:
    query: str
    expected_title: str
    expected_facts: list[str] = field(default_factory=list)
    expected_author: str = ""


@dataclass
class RetrievalHit:
    title: str
    document: str = ""
    summary: str = ""
    author: str = ""
    distance: float | None = None

    @property
    def searchable_text(self) -> str:
        return " ".join([self.title, self.author, self.summary, self.document]).lower()


def _case_from_dict(data: dict[str, Any]) -> RetrievalCase:
    facts = data.get("expected_facts") or []
    if isinstance(facts, str):
        facts = [facts]
    return RetrievalCase(
        query=str(data.get("query", "")).strip(),
        expected_title=str(data.get("expected_title", "")).strip(),
        expected_author=str(data.get("expected_author", "")).strip(),
        expected_facts=[str(f).strip() for f in facts if str(f).strip()],
    )


def load_cases(path: str | Path) -> list[RetrievalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(raw_cases, list):
        raise ValueError("Retrieval eval cases must be a JSON list or an object with a 'cases' list.")
    cases = [_case_from_dict(item) for item in raw_cases if isinstance(item, dict)]
    return [case for case in cases if case.query and case.expected_title]


def normalize_results(raw_results: Any) -> list[RetrievalHit]:
    if raw_results is None:
        return []

    if isinstance(raw_results, dict) and "results" in raw_results:
        raw_results = raw_results.get("results") or []

    # Chroma shape: {"documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
    if isinstance(raw_results, dict) and "documents" in raw_results:
        docs = raw_results.get("documents") or [[]]
        metas = raw_results.get("metadatas") or [[]]
        distances = raw_results.get("distances") or [[]]
        docs0 = docs[0] if docs and isinstance(docs[0], list) else docs
        metas0 = metas[0] if metas and isinstance(metas[0], list) else metas
        distances0 = distances[0] if distances and isinstance(distances[0], list) else distances
        hits = []
        for i, doc in enumerate(docs0):
            meta = metas0[i] if i < len(metas0) and isinstance(metas0[i], dict) else {}
            distance = distances0[i] if i < len(distances0) else None
            hits.append(
                RetrievalHit(
                    title=str(meta.get("Title", "")),
                    author=str(meta.get("Author", "")),
                    summary=str(meta.get("Summary", "")),
                    document=str(doc or ""),
                    distance=distance if isinstance(distance, (float, int)) else None,
                )
            )
        return hits

    if isinstance(raw_results, list):
        hits = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") or item.get("metadatas") or {}
            hits.append(
                RetrievalHit(
                    title=str(meta.get("Title", item.get("Title", ""))),
                    author=str(meta.get("Author", item.get("Author", ""))),
                    summary=str(meta.get("Summary", item.get("Summary", ""))),
                    document=str(item.get("document", item.get("documents", "")) or ""),
                    distance=item.get("distance") if isinstance(item.get("distance"), (float, int)) else None,
                )
            )
        return hits

    return []


def _same_title(actual: str, expected: str) -> bool:
    return actual.strip().casefold() == expected.strip().casefold()


def _fact_hits(case: RetrievalCase, hits: list[RetrievalHit], k: int) -> list[str]:
    combined = " ".join(hit.searchable_text for hit in hits[:k])
    return [fact for fact in case.expected_facts if fact.casefold() in combined]


def evaluate_case(case: RetrievalCase, raw_results: Any, *, k: int = 3) -> dict[str, Any]:
    hits = normalize_results(raw_results)
    top_titles = [hit.title for hit in hits[:k]]
    expected_rank = None
    for idx, hit in enumerate(hits, start=1):
        if _same_title(hit.title, case.expected_title):
            expected_rank = idx
            break

    matched_facts = _fact_hits(case, hits, k)
    fact_total = len(case.expected_facts)
    fact_coverage = (len(matched_facts) / fact_total) if fact_total else None

    return {
        "query": case.query,
        "expected_title": case.expected_title,
        "top_titles": top_titles,
        "expected_rank": expected_rank,
        "top1_match": expected_rank == 1,
        f"top{k}_match": expected_rank is not None and expected_rank <= k,
        "matched_facts": matched_facts,
        "expected_facts": case.expected_facts,
        "fact_coverage": fact_coverage,
    }


def summarize_results(case_results: list[dict[str, Any]], *, k: int = 3) -> dict[str, Any]:
    total = len(case_results)
    if total == 0:
        return {"total_cases": 0, "top1_accuracy": 0.0, f"top{k}_accuracy": 0.0, "average_fact_coverage": None}

    fact_scores = [r["fact_coverage"] for r in case_results if isinstance(r.get("fact_coverage"), (float, int))]
    return {
        "total_cases": total,
        "top1_accuracy": sum(1 for r in case_results if r.get("top1_match")) / total,
        f"top{k}_accuracy": sum(1 for r in case_results if r.get(f"top{k}_match")) / total,
        "average_fact_coverage": (sum(fact_scores) / len(fact_scores)) if fact_scores else None,
    }


def evaluate_retrieval(
    cases: Iterable[RetrievalCase],
    query_fn: Callable[[str, int], Any],
    *,
    k: int = 3,
) -> dict[str, Any]:
    case_results = []
    for case in cases:
        raw_results = query_fn(case.query, k)
        case_results.append(evaluate_case(case, raw_results, k=k))
    return {
        "generated_at": datetime.now().isoformat(),
        "k": k,
        "summary": summarize_results(case_results, k=k),
        "cases": case_results,
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a lightweight retrieval evaluation against the configured vector store."
    )
    parser.add_argument(
        "--cases", default="Evaluation/retrieval_eval_cases.example.json", help="Path to retrieval eval cases JSON."
    )
    parser.add_argument(
        "--output", default="Evaluation/retrieval_eval_report.json", help="Where to write the JSON report."
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieval cutoff.")
    args = parser.parse_args()

    from Orchestrator.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    cases = load_cases(args.cases)
    report = evaluate_retrieval(
        cases,
        lambda query, k: orchestrator.query_vector_store(query=query, n_results=k),
        k=args.k,
    )
    output = write_report(report, args.output)
    print(f"Wrote retrieval evaluation report to {output}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
