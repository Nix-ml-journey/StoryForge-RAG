# Evaluation and agentic-loop decisions

Each ADR below keeps its original number. Newer decisions are appended; superseded ones keep their status.

---

## ADR-0001: Evaluate drafts with local Ollama, with no API fallback

### Status
Accepted (supersedes the in-process Transformers evaluator from the 2026-09 "local evaluation" session)

### Date
2026-10-05

### Context
`Evaluation_mode: "local"` loaded `Qwen3-4B-Instruct` through Transformers on CPU and, on any
failure, silently retried through HF then Gemini. On the target machine that meant a slow CPU
judge, a second model download, and API calls (and credits) the user believed were switched off.
Ollama already serves the generation model on the GPU.

### Decision
`Evaluation_mode` accepts `ollama` (alias `local`). The judge is `Ollama_evaluation_model`
(default `Generative_model`), reached through the existing `load_ollama_llm` loader. If Ollama is
unreachable or the model is not pulled, `evaluate_model()` raises `RuntimeError`; in this mode
there is **no** HF/Gemini fallback. `Evaluation_mode: "api"` is unchanged.

### Alternatives Considered
- Keep Transformers-on-CPU: slow, duplicate model, hidden fallback. Rejected.
- Transformers-on-GPU: competes with Ollama for 16 GB VRAM. Rejected.
- Keep the fallback but log it: the failure mode (silently spending credits) stays. Rejected.

### Consequences
- Fully offline pipeline; each judgement costs one Ollama call.
- The judge is usually the same model as the generator, so scores can run generous. Prompts were
  recalibrated (strict 1-10 anchors) and `Ollama_evaluation_model` lets you pick a different judge.
- If Ollama dies mid-run the agentic loop logs an ERROR and falls back to completeness-only
  decisions (it still never ACCEPTs a draft with zero grounded facts).

---

## ADR-0006: Agentic loop decision order, grounded judge, refine guard

### Status
Accepted (implemented 2026-10-05; re-measured 2026-10-06: accept 0.75 -> 0.88, iterations 1.75 -> 1.38, with caveats in `docs/PROJECT_JOURNEY.md`)

### Date
2026-10-05

### Context
The first clean scored baseline (see `docs/PROJECT_JOURNEY.md`) showed four problems: the judge scored
faithfulness without seeing the facts (swinging 2-10); short or truncated drafts were scored low on
faithfulness and sent to RE_RETRIEVE instead of being finished; RE_RETRIEVE switched the reranker off and
thinned the evidence (27 -> 5 and 12 -> 2 facts); and a refine pass once shrank a draft from 1088 to 513 words.

### Decision
1. `evaluate_story_text(..., facts=...)` adds a "Grounded facts (source of truth)" section to the judge
   prompt; the loop passes `format_facts_for_prompt(parsed)`. Without facts the prompt is unchanged.
2. `decide_action` order is now: no facts -> RE_RETRIEVE; incomplete/short -> REFINE; low faithfulness ->
   RE_RETRIEVE; score >= bar -> ACCEPT; thin facts -> RE_RETRIEVE; else REFINE.
3. RE_RETRIEVE no longer disables the reranker.
4. `refine_regressed(prior, new)`: if the refined draft has under 80% of the prior word count or fewer
   sections, the loop keeps the prior draft and logs a warning.

### Alternatives Considered
- Judge sees only the story (status quo): faithfulness stays a guess.
- Raise `Agentic_loop_min_faithfulness`: treats the symptom and re-retrieves more often.
- Keep the reranker off on re-retrieve to widen the pool: measured to cut the facts available.

### Consequences
- Truncated drafts are refined before they can trigger a re-retrieve, so a bad judge score on a cut-off draft
  no longer discards good sections.
- The judge prompt is longer (the facts list), costing a few hundred tokens per iteration.
- A rejected refine re-evaluates the same text, wasting one judge call; accepted as simpler than caching.
- Re-run `scripts/measure_generation_length.py` to confirm accept rate and iterations.

---

## ADR-0007: Close the silent-accept gaps in the agentic loop

### Status
Accepted (implemented 2026-10-06; not yet re-measured)

### Date
2026-10-06

### Context
The Oct 6 baseline showed an unscored accept (judge returned nothing), accepts on 1-3 facts, three identical
rejected refines on one query, and Step 2 yielding 0-3 facts on four of eight queries (9-25 on the others).
Step 2 only retried when it parsed 0 facts, so a parsed-but-thin answer (schema-constrained output stopping the
list early) went through unchallenged. Truncation is unlikely here: `HF_grounded_facts_max_new_tokens` is 3200
and `Model_max_prompt_tokens` 12288.

### Decision
1. `evaluate_story_text` retries once with a stricter "reply with only the JSON object" suffix when the judge
   output is not valid JSON, and logs a warning; the loop also warns when an iteration is left unscored.
2. `decide_action` checks `Agentic_loop_min_facts` before ACCEPT: thin grounding goes to RE_RETRIEVE even with
   high scores.
3. After a rejected refine, the next refine feedback says "do NOT shorten: add about N words". A second
   consecutive rejection stops the loop (`stop_reason: refine_rejected`) and keeps the best draft.
4. Local Step 2 retries when it parses fewer than `Local_grounded_facts_min_facts` (default 6) facts, asking for
   coverage ("at least N facts, one per entity"), and keeps the larger answer. The attempt log now includes chunk
   count and prompt size to separate "model stopped early" from "chunks were sparse".

### Alternatives Considered
- Treat an unscored judge as a hard failure: would discard finished drafts on a flaky 4B judge.
- Regenerate from facts after a rejected refine: costs a full generation pass; asking for growth is cheaper.
- Raise `HF_grounded_facts_max_new_tokens`: not the cause at 3200.

### Consequences
- Fewer accepts on thin evidence, so accept rate may dip while quality rises; a thin-facts query that stays thin
  ends as `max_iterations` instead of accepted.
- One extra Step 2 call on thin answers and one extra judge call on unparseable output.
- Re-run `scripts/measure_generation_length.py` and read the new `Local grounded-facts attempt` log lines.

---

## ADR-0008: Refine-guard minimum, refine-once on low faithfulness, lower token floor

### Status
Accepted (implemented 2026-10-06; not yet re-measured)

### Context
The ADR-0007 re-measure (accept 0.62) showed: a cut-off 2688-word draft whose tighter refines were rejected as
"under 80% of the prior length"; a complete 13-fact draft re-retrieved because a 4B judge scored faithfulness 4
(later 7 and 8 on the same facts); and drafts averaging 31% over target because `Single_pass_fast_max_tokens: 3200`
sat above the ~2700-token budget derived from a 1500-word target.

### Decision
1. `refine_regressed(..., min_words=...)`: a shorter refine is accepted when it still meets the minimum word count
   and keeps all sections.
2. On iteration 1, low faithfulness with at least `Agentic_loop_min_facts` facts gives REFINE (judge notes as
   feedback) instead of RE_RETRIEVE; still low on iteration 2+, or thin facts, re-retrieves.
3. ~~`Single_pass_fast_max_tokens` lowered to 2700~~ -- reverted to 3200 after the re-run (see below).

### Consequences
- Fewer discarded good drafts; one extra refine pass on genuinely unfaithful drafts before re-retrieving.
- A hard token ceiling can still truncate a draft, which the refine then finishes.
- Re-run `scripts/measure_generation_length.py` and compare accept rate, words and time.

### Re-run result and corrections (2026-10-06)
Accept fell further to 0.38 (3/8), 2.25 iterations, 260 s per query. Five iterations were unscored (`no eval provider`, average 0.0)
and two drafts lost SECTION 5. Causes found:
- `Local_evaluation_max_new_tokens: 700` is too small for the judge JSON (five feedbacks, a conclusion and 2-4 per-section
  suggestions), so the output was cut off, failed to parse, and the retry hit the same limit. Raised to 1200.
- The 2700-token floor cut off SECTION 5 on drafts the model writes ~60% over target; those drafts completed under 3200. Reverted to 3200.
  Overshoot itself is still open (lever: stronger per-section length guidance, not a hard cap).

---

## ADR-0009: Short-output judge, loop helpers, opt-in sectioned writer

### Status
Accepted (implemented 2026-10-06; not yet re-measured)

### Context
The 0.38 re-run showed unscored judge iterations (judge JSON cut off), drafts 30-60% over target, SECTION 5 lost to the
token limit, and refines that could not repair a cut-off draft. One long loop function carried all of it.

### Decision
1. `evaluation.loop_judge` prompt (used only by `evaluate_story_text`): flat integer scores with faithfulness first, one
   sentence, 2 short suggestions. `_parse_json_response` salvages scores from truncated JSON. The detailed `with_story`
   prompt stays for the API evaluate routes. Completeness and length are already rule-checked in code, so the judge is
   not asked about them.
2. `agentic_loop.py`: `_iteration_record` replaces three copies of the iteration dict, and one `retrieve_and_extract`
   closure replaces the duplicated retrieve/extract block.
3. `Story_generation_mode: "sectioned"` (default `single`): `generate_from_facts` writes five short calls, each with a
   word budget and token cap, seeing the sections already written. A refine keeps sections that are present, long enough
   and not over 1.6x budget, and rewrites only the rest (all five if none is structurally wrong). `_invoke_nonempty`
   unifies the empty-draft retry. The SSE streaming route still uses the single-pass prompt.

### Alternatives Considered
- Full loop rewrite as a state machine: deferred until the sectioned mode has been measured.
- Raising only the judge token limit: done too (1200), but does not protect against other truncation.

### Consequences
- Judge output is about a third the length; scores arrive even if the reply is cut off.
- Sectioned mode costs five calls per full draft but removes most whole-story refines; compare accept rate, words and time
  against `single` on the same 8 queries.

---

## ADR-0010: Story-arc generation method alongside 5W1H (for A/B comparison)

### Status
Accepted (implemented 2026-10-07; measured 2026-10-07; a second-model review found the ranking not supported at n=8, so the default stays `5w1h`; see Result and Review)

### Context
ADR-0009 added a sectioned writer, but it still used the 5W1H outline, so a comparison against the single-pass baseline
could not separate "one call per section" from "a different story structure".

### Decision
`Story_generation_method: arc` (default `5w1h`) selects `rag/generation_arc.py`: a story-arc outline (setup, inciting
incident, rising action, climax, resolution), a separate prompt set (`generation_arc` in `prompts.yaml`, with one "job" line
per section), always written one section per call. It reuses the section loop in `generation.py` (`_generate_sectioned`,
now parameterised by headers, prompts and roles), so budgets and the rewrite-only-weak-sections refine are identical.
`scripts/measure_generation_length.py --methods 5w1h 5w1h-sectioned arc` runs the same queries once per method via a
config overlay, writes one report per method, and prints a comparison table.

### Alternatives Considered
- Copy the sectioned loop into the new module: duplicates the part that should stay identical for a fair comparison.
- Weight section budgets (longer climax): changes two things at once; try after the first comparison.

### Consequences
- Three variants to compare: `5w1h` (single pass), `5w1h-sectioned` (isolates the method), `arc` (isolates the outline).
- Judge, loop, completeness check and Step 2 facts are shared, so differences come from generation only.
- A full three-way run is roughly three times the single-run time.


### Result (2026-10-07, same 8 queries, `--length long`, fast mode, judge `qwen3.5:4b` with the short `loop_judge` prompt)
| Metric | 5w1h | 5w1h-sectioned | arc |
|---|---|---|---|
| Accept rate | 0.50 | 0.50 | 0.62 |
| Avg iterations | 2.0 | 2.25 | 2.0 |
| Avg words (target 1500) | 1957.6 | 1856.6 | 1787.8 |
| Avg seconds per query | 104.6 | 123.1 | 123.1 |
| Final drafts left incomplete | 3 of 8 | 0 | 0 |
| Iterations with no score (a "refine rejected twice" stop, not a judge failure) | 2 | 0 | 0 |
| Avg final score | 7.62 | 6.97 | 7.40 |

Reading the table (corrected after review, see below):
- The accept rate differs by one or two queries (counts 4, 4, 5 of 8), well inside run-to-run noise.
- `5w1h` ended with a cut-off or section-less final draft on 3 queries (for example the whispering draft at 2831 words). The two
  sectioned variants had none, so avoiding cut-off drafts is a property of writing section by section, not specifically of `arc`.
- `arc` stays closest to the length target (mean distance from 1500 words: 288 for `arc`, 357 sectioned, 460 `5w1h`).
- The +18 s per query for `arc` is almost all one query (lawyer, +142 s); without it the gap is under 1 s.

### Review (2026-10-07, second model, Opus, per the multi-model review pattern)
Verdict: partly agree. The first draft of this record was wrong in three places; each was checked against the reports and fixed above.
- **Lawyer query.** I had written that it stalled under every method. It was accepted at iteration 1 by `5w1h` (12 facts, 7.4, faithfulness 9)
  and failed under both sectioned variants: faithfulness 4 on every iteration, and the last re-retrieve returned **0 facts** (25 -> 25 -> 0 for `arc`).
  So for that query the sectioned runs did worse, and the re-retrieve path that drops the facts is the likely cause. Only scholar (0 facts from Step 2)
  and huntress (faithfulness 4-6 whatever the facts) stall under all three.
- **"Unscored judge iterations."** The two `5w1h` cases (goblin, whispering, iteration 3) are the "refine regressed twice" stop, where nothing is
  re-scored; they are not judge failures, so the earlier row overstated the judge problem. In both, the refine never produced a draft that passed the
  guard (goblin stays at 1723 words, whispering at 2831).
- **Judge noise is large.** The same 2831-word whispering draft scored 6.2 and then 8.0 on consecutive iterations; doctor scored 7.4 / 9.0 under all three
  methods. Differences of 0.2-0.6 in average score mean nothing here. Scholar scored 8.2-8.4 with 0 facts, so faithfulness is not checking grounding.
- **Inputs were not held fixed.** First-iteration fact counts differ by method for the same query (huntress 12 / 17 / 18, lawyer 25 / 12 / 12), so
  retrieval and Step 2 variance is mixed into every comparison.
- **Reporting bug.** `measure_generation_length.py` takes `final_average` from the best draft but `final_faithfulness` from the last iteration, so
  the pair can describe different drafts (arc scholar shows 8.2 / 6.0; the kept draft was 8.2 / 9.0).
- **Not supported at n=8 with one run per method:** any ranking by score, faithfulness or accept rate. `arc` is not better on first drafts (mean
  iteration-1 score 6.97 vs 7.40 for `5w1h`); its extra accepts come from the refine loop.

### Follow-up
1. Freeze the retrieval and Step 2 output once per query and feed the same facts to all three methods; run 3-5 seeds each. This separates generation from
   retrieval and judge noise and settles the lawyer question.
2. Fix the reporting pair (`final_faithfulness` from the kept draft) before the next run.
3. Investigate why a refine in `5w1h` keeps being rejected, and why a re-retrieve can return 0 facts for a reformulated query (lawyer, iteration 3).
4. Keep `Story_generation_method: "5w1h"` as the default until then. Not yet tried: weighted section budgets (longer climax).

---

## ADR-0011: Target architecture: pre-flight gate, section-by-section arc writer, rule checks first

### Status
Proposed (set as the goal on 2026-10-07). Step 1 implemented 2026-10-07, not yet measured; steps 2 and 3 not started.

### Date
2026-10-07

### Context
The 2026-10-07 comparison (ADR-0010) and its review showed where time and quality go on one 16 GB GPU:
- a full story is written before anyone checks the evidence (scholar: 2000+ words from 0 facts, then a re-retrieve);
- Step 2 can return 0-3 facts, and a re-retrieve can replace 25 facts with 0 (lawyer, iteration 3);
- the 4B judge steers the loop but is noisy (the same draft scored 6.2, then 8.0), so decisions and iterations are noise-driven;
- single-pass drafts overshoot the length target and lose SECTION 5; writing section by section fixes that.

### Decision
Head toward this flow. Each step is a small change on code that already exists:
```
Query -> hybrid retrieval -> facts (LLM, extractive fallback)
      -> GATE: enough facts? no -> widen (tier 1: more results; tier 2: reformulated query)
                                   merge facts across tiers, re-check, max 2 tiers, then extractive fallback
      -> write sections 1-5 (arc, each sees the story so far)
      -> rule checks per section: length, complete sentence, names not in the facts
      -> any fail? yes -> rewrite only those sections (max 2 rounds) -> re-check
      -> one judge call -> report score
```
Design rules:
1. The gate runs before any writing. "Enough" is a fact count and a source-diversity threshold (to be tuned; start from `Agentic_loop_min_facts`).
2. Widening merges facts; it never replaces them. After two tiers, build extractive facts (chunk id plus a short passage, no LLM) and write anyway.
3. Accept means the rule checks pass. The judge runs once at the end and only reports a score; it does not steer the loop. Its score stays visible
   because name checks miss invented events.
4. Rewrite touches only failing sections (the sectioned refine, which exists).

### What exists and what is new
- Exists: hybrid retrieval, widening via `k_boost` and `reformulate_query`, the sectioned/arc writer with per-section rewrite, the attribution gate.
- New: the pre-flight gate, fact merging across tiers, the extractive fallback, per-section rule checks (including novel names), and a loop driven
  by rules instead of judge scores.

### Alternatives Considered (and why not)
- **Arc planner (per-story outline call):** one extra 9B call, it can invent names, and the arc headers are already fixed. Assign facts to sections
  by rule only if prompt size becomes a measured problem.
- **Context buffers (summaries between sections):** the writer already passes the full story so far (about 1500 tokens); summaries add calls and lose detail.
- **"Adaptive" judge with several calls:** more noise and more GPU time for no defined gain.
- **5W1H sub-queries with a categorical matrix:** speculative. Test sub-query retrieval alone with `scripts/retrieval_eval.py` before building it.
- **Full rewrite:** unnecessary; everything above is incremental.

### Consequences
- Roughly 8-12 short 9B calls plus one judge call per story, instead of repeated long drafts plus a judge each time. VRAM is unchanged
  (9B ~6.6 GB + 4B ~3.3 GB + KV cache, about 11-12 GB). Time per query is an expectation, not a measurement.
- Thin-evidence queries stop early (before writing) instead of after a full draft.
- Risk: rule checks accept stories that invent events without new names. Mitigation: report the judge score; compare rule decisions with judge
  scores on saved drafts before trusting them for control.

### Plan and acceptance
1. Gate + fact merging + extractive fallback. Done when scholar-type queries never write from 0-2 facts and a re-retrieve never lowers the fact count.
2. Per-section rule checks drive the loop (completeness, length, novel names). Done when decisions agree with human reading on saved drafts.
3. Judge once at the end. Done when judge calls per story drop to 1.
After each step: `pytest`, `ruff`, `scripts/retrieval_eval.py` (floors top-1 0.80, top-3 0.833, fact coverage 0.733) and
`scripts/measure_generation_length.py --methods arc` against the 2026-10-07 `arc` run. Because n=8 and one run is noisy, add frozen facts per query
and 3-5 seeds before claiming a gain (ADR-0010 follow-up).

### Progress
- **Step 1 (implemented and measured once, 2026-10-07):**
  - `extractive_facts` and `merge_facts` in `attribution.py`; Step 2 tops up to `Local_grounded_facts_min_facts` facts with extractive facts when the LLM
    returns too few (`Facts_extractive_fallback`, default on).
  - Pre-flight gate in the agentic loop (`Agentic_loop_preflight_tiers`, default 2): thin facts widen retrieval, then reformulate the query, before any
    story is written. Facts are merged across tiers, and an in-loop RE_RETRIEVE now merges too, so the 25 -> 0 drop cannot recur.
  - Known limit: `retrieval_chunks` in the result lists only the latest retrieval, while merged facts may cite chunks from an earlier one.
  - **Result (`--methods arc`, 8 queries, one run):** accept 0.88 (7/8) vs 0.62 for the earlier `arc` run; average iterations 2.0 (unchanged);
    average words 1829 (target 1500, under-min 0.0); average 122.0 s (was ~123 s); 8/8 generations succeeded.
  - Per query: 4 accepted on iteration 1 (scientist, goblin, lawyer, warrior; 58-100 s), 3 accepted on iteration 3 (scholar, doctor, whispering),
    huntress hit `max_iterations`.
  - Scholar: Step 2 returned `no_json` / `no_facts`; the extractive top-up supplied 6 facts and the story was accepted on iteration 3. This is the
    clearest effect of Step 1.
  - Huntress still stalls: faithfulness 4.0 on all three iterations despite 20-30 facts. Thin facts are not the cause; suspect judge noise or content
    drift, which Steps 2-3 (rule checks, judge reports only) are meant to address.
  - Caveats: n=8, a single run, `arc` only (no 5w1h comparison re-run), and the 4B judge is noisy (same text scored 6.2 then 8.0). Treat 0.62 -> 0.88
    as encouraging, not proven. Reporting note: `final_average` comes from the best draft while `final_faithfulness` comes from the last iteration
    (e.g. doctor shows average 6.8 with faithfulness 9.0), so the two columns are not a matched pair; not yet fixed.
  - Next: Step 2 (per-section rule checks drive the loop) once confirmed; optionally repeat Step 1 with 3 seeds first to size the noise.
