# Architecture decisions

Short records of *why* the pipeline looks the way it does. New decisions get the next number;
superseded ones stay, with their status updated.

| ADR | Decision |
|---|---|
| [0001](0001-ollama-evaluation-no-api-fallback.md) | Evaluate with local Ollama, no API fallback |
| [0002](0002-global-bm25-candidates.md) | Whole-corpus BM25 candidates join the rerank pool (tested, no gain, off) |
| [0003](0003-story-type-filters-retrieval.md) | `story_type` filters retrieval on `Is_series` |
| [0004](0004-remove-vector-store-insert-update.md) | Remove the vector-store insert/update routes |
| [0005](0005-model-choices-16gb-vram.md) | Model choices for a 16 GB GPU |
| [0006](0006-agentic-loop-decision-order.md) | Agentic loop decision order, grounded judge, refine guard |
