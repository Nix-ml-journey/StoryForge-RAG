# ADR-0003: `story_type` filters retrieval on `Is_series`

## Status
Accepted

## Date
2026-10-05

## Context
The API accepted `story_type` (single / series / mix) and echoed it back, but it never reached
retrieval, so it did nothing. Ingest already stores a boolean `Is_series` on every chunk.

## Decision
`single` adds `Is_series == False`, `series` adds `Is_series == True`, `mix` adds nothing. The
filter is combined with any caller `filter_metadata` using `$and`. If a filtered search returns
nothing, retrieval retries without it and logs a warning instead of returning no context.

## Alternatives Considered
- Remove the parameter: breaks API clients and discards a useful control.

## Consequences
- The parameter now does what its name says; the SSE streaming route has no `story_type` field.
