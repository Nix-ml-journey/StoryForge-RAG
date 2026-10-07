# Retrieval decisions

Each ADR below keeps its original number. Newer decisions are appended; superseded ones keep their status.

---

## ADR-0002: Add whole-corpus BM25 candidates to the rerank pool

### Status
Implemented, **disabled by default** (`Hybrid_bm25_pool: 0`): the 2026-10-05 measurement showed no gain

### Date
2026-10-05

### Context
BM25 only re-ordered the 48-or-so chunks dense search had already returned, so it could never
recover a story dense search missed. The 30-case retrieval eval had 2 never-retrieved cases
(Frankenstein, Body Snatcher) and 4 low-rank cases. Offline BM25 over all 2,138 chunks placed
Frankenstein and Haunter of the Dark at rank 1, Cool Air at 3, Body Snatcher at 9, Olalla at 10.

### Decision
`retrieve_docs` takes the top `Hybrid_bm25_pool` (default 0 = off; 12 was tested) chunks from a BM25 search over the
whole (optionally filtered) collection, fuses them with the dense list via RRF, and passes the
union to the cross-encoder, which still decides the final order. Tokenization strips punctuation.
`Hybrid_bm25_pool: 0` is the old behaviour and is the shipped default.

### Alternatives Considered
- Bigger dense `k`: does not help when the embedding itself misses the story.
- Re-chunking / re-embedding: expensive, touches the locked chunking logic, and not shown to help.
- Query rewriting with an LLM: extra latency per request for an uncertain gain.

### Consequences
- Lexical matches on names and rare terms now reach the reranker; recall should rise.
- Risk: the reranker may still demote them; Randolph Carter (BM25 rank 27) and Olalla remain hard.
- The BM25 index is built from `vectorstore.get()` and cached by corpus ids; ~2k chunks is cheap.

### Result (2026-10-05, `scripts/retrieval_eval.py`, 30 cases, k=3)
| | top-1 | top-3 | fact coverage |
|---|---|---|---|
| Before (pool 0) | 0.80 | 0.833 | 0.733 |
| Pool 12 | 0.80 | 0.833 | 0.717 |

No case moved into the top 3 and one lost a fact (Whisperer in Darkness 1.0 -> 0.5), so the change
fails the repo's retrieval gate (fact_coverage floor 0.733) and is switched off. Frankenstein (BM25
rank 1 offline) and Body Snatcher are still never returned, even though BM25 now nominates them.

### Why it probably failed (hypothesis, not yet measured)
The candidate reaches the pool, but the cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores everything
from scratch and ranks the Herbert West / War of the Worlds chunks above it; title diversity then
keeps three titles. The recall fix is upstream of a reranker that overrides it.

### Next step
Add a per-stage diagnostic (dense rank, BM25 rank, rerank rank, diversity outcome for each expected
title) before changing anything else, then try fusing the rerank rank with the BM25/dense rank
instead of letting the cross-encoder decide alone.

Re-run with the pool off (2026-10-06): top-1 0.80, top-3 0.867, fact coverage 0.733. The fact-coverage floor is met again; the top-3 gain is unexplained.

---

## ADR-0003: `story_type` filters retrieval on `Is_series`

### Status
Accepted

### Date
2026-10-05

### Context
The API accepted `story_type` (single / series / mix) and echoed it back, but it never reached
retrieval, so it did nothing. Ingest already stores a boolean `Is_series` on every chunk.

### Decision
`single` adds `Is_series == False`, `series` adds `Is_series == True`, `mix` adds nothing. The
filter is combined with any caller `filter_metadata` using `$and`. If a filtered search returns
nothing, retrieval retries without it and logs a warning instead of returning no context.

### Alternatives Considered
- Remove the parameter: breaks API clients and discards a useful control.

### Consequences
- The parameter now does what its name says; the SSE streaming route has no `story_type` field.
