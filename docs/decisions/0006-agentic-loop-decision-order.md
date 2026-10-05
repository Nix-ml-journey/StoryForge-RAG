# ADR-0006: Agentic loop decision order, grounded judge, refine guard

## Status
Accepted (implemented 2026-10-05; effect not yet re-measured)

## Date
2026-10-05

## Context
The first clean scored baseline (see `docs/PROJECT_JOURNEY.md`) showed four problems: the judge scored
faithfulness without seeing the facts (swinging 2-10); short or truncated drafts were scored low on
faithfulness and sent to RE_RETRIEVE instead of being finished; RE_RETRIEVE switched the reranker off and
thinned the evidence (27 -> 5 and 12 -> 2 facts); and a refine pass once shrank a draft from 1088 to 513 words.

## Decision
1. `evaluate_story_text(..., facts=...)` adds a "Grounded facts (source of truth)" section to the judge
   prompt; the loop passes `format_facts_for_prompt(parsed)`. Without facts the prompt is unchanged.
2. `decide_action` order is now: no facts -> RE_RETRIEVE; incomplete/short -> REFINE; low faithfulness ->
   RE_RETRIEVE; score >= bar -> ACCEPT; thin facts -> RE_RETRIEVE; else REFINE.
3. RE_RETRIEVE no longer disables the reranker.
4. `refine_regressed(prior, new)`: if the refined draft has under 80% of the prior word count or fewer
   sections, the loop keeps the prior draft and logs a warning.

## Alternatives Considered
- Judge sees only the story (status quo): faithfulness stays a guess.
- Raise `Agentic_loop_min_faithfulness`: treats the symptom and re-retrieves more often.
- Keep the reranker off on re-retrieve to widen the pool: measured to cut the facts available.

## Consequences
- Truncated drafts are refined before they can trigger a re-retrieve, so a bad judge score on a cut-off draft
  no longer discards good sections.
- The judge prompt is longer (the facts list), costing a few hundred tokens per iteration.
- A rejected refine re-evaluates the same text, wasting one judge call; accepted as simpler than caching.
- Re-run `scripts/measure_generation_length.py` to confirm accept rate and iterations.
