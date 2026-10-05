# ADR-0004: Remove the vector-store insert and update routes

## Status
Accepted

## Date
2026-10-05

## Context
`/vector_store/insert` and `/update` wrote through `Collection.add/update` on a collection created
without an embedding function. Rows written that way carry no usable embedding for BGE retrieval
and the update path could not re-embed, so data written through them was invisible or stale.

## Decision
Delete both routes, their request/response models, the Orchestrator methods, the
`response_parameter` helpers, and `update_data`. `/delete` stays (ID-based, needs no embedding).
Content enters the index only through `scripts/reset_and_ingest.py` / the ingest routes.

## Consequences
- One write path, always embedded with the configured model.
- Clients that called insert/update must re-ingest instead.
