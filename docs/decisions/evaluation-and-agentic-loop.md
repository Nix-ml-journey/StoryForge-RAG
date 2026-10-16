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
   word budget and token cap, seeing the sections already written. A refine keeps sections that are present and long enough (since 2026-10-09 over-long ones are kept too and trimmed to whole sentences upstream; see ADR-0011), and rewrites only the rest (all five if none is structurally wrong). `_invoke_nonempty`
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
Accepted (implemented 2026-10-07; measured 2026-10-07; a second-model review found the ranking not supported at n=8, so the default stayed `5w1h` at the time; ADR-0011 later made `arc` the default; see Result and Review)

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

## ADR-0011: Target architecture: pre-flight gate, section-by-section arc writer, judge-driven loop

### Status
Accepted, revised 2026-10-08. Step 1 is the default and measured. Step 2 (rule-driven loop) is built, measured three times, and kept opt-in: it
is faster but its stories score lower, so the judge-driven loop stays in control.

### Date
2026-10-07 (set as the goal), revised 2026-10-08.

### Context
The 2026-10-07 comparison (ADR-0010) showed where time and quality go on one 16 GB GPU:
- a full story was written before anyone checked the evidence (scholar: 2000+ words from 0 facts, then a re-retrieve);
- Step 2 could return 0-3 facts, and a re-retrieve could replace 25 facts with 0 (lawyer, iteration 3);
- the 4B judge steers the loop but is noisy (the same draft scored 6.2, then 8.0);
- single-pass drafts overshoot the length target and lose SECTION 5; writing section by section fixes that.

### Decision
Build on the `arc` writer (now the default `Story_generation_method`) with this flow:
```
Query -> hybrid retrieval -> facts (LLM, extractive top-up)
      -> GATE: enough facts? no -> widen (tier 1: more results; tier 2: reformulated query), merge facts, max 2 tiers
      -> write sections 1-5 (arc, each sees the story so far)
      -> judge -> ACCEPT | REFINE (rewrite only weak sections) | RE_RETRIEVE (merge facts), max 3 iterations
```
Rules:
1. The gate runs before any writing; widening merges facts and never replaces them (`Agentic_loop_min_facts`, `Agentic_loop_preflight_tiers`).
2. Thin facts are topped up with extractive facts (chunk sentences, no LLM, `Facts_extractive_fallback`).
3. The judge decides accept/refine/re-retrieve. Its noise is limited by refine-once-first on low faithfulness, a thin-facts block on ACCEPT, and a
   late accept slack.
4. Opt-in fast mode: `Agentic_loop_rule_driven: true` replaces the judge with per-section rule checks (`section_rules.py`: missing, unfinished
   sentence, too few sentences, over/under word budget, names not in the facts), rewrites only failing sections (max 2 rounds), trims over-long
   sections to whole sentences, and calls the judge once to report a score.

### Alternatives Considered (and why not)
- **Rules as the default controller:** about 2x faster (71-113 s vs 132-155 s per query) but judge average 5.8 vs 7.17 and faithfulness 5.5 vs 6.62
  in the latest run; rules check shape, not faithfulness, so weak first drafts were accepted.
- **Lexical grounding score as a faithfulness gate:** tried 2026-10-08; ratios (0.02-0.55) did not track judge faithfulness. Code removed.
- **Arc planner (per-story outline call):** one extra 9B call, it can invent names, and the arc headers are already fixed.
- **Context buffers (summaries between sections):** the writer already passes the full story so far (about 1500 tokens).
- **"Adaptive" judge with several calls:** more noise and more GPU time for no defined gain.
- **5W1H sub-queries with a categorical matrix:** speculative. Test sub-query retrieval alone with `scripts/retrieval_eval.py` first.

### Consequences
- VRAM unchanged (9B ~6.6 GB + 4B ~3.3 GB + KV cache, about 11-12 GB). The default loop still pays for up to 3 drafts plus 3 judge calls on hard queries.
- Thin-evidence queries are fixed before writing, not after a full draft.
- A judge-gated rewrite on top of the rule loop is the open way to get speed without losing quality; not built.

### Evidence
Eight queries per run (`scripts/measure_generation_length.py`), one run each; n=8 and the 4B judge are noisy, so differences under about one judge point
are not claims.

| Run | Accept | Iterations | Seconds | Judge avg / faithfulness |
|---|---|---|---|---|
| `arc`, before Step 1 | 0.62 | 2.0 | 123 | - |
| `arc`, Step 1 | 0.88 | 2.0 | 122 | - |
| `arc` (2026-10-08 compare) | 0.62 | 2.0 | 155 | 7.17 / 6.62 |
| `arc-rules`, three runs | 1.0, 0.62, 0.88 | 2.38, 2.5, 1.25 | 87, 113, 71 | 6.53 / 6.0 (run 1), 5.8 / 5.5 (run 3) |

- Step 1 (gate, merging, extractive top-up): scholar went from stalling to accepted after the top-up supplied 6 facts. Huntress still stalls at
  faithfulness 4 with 20-30 facts, so thin facts are not the whole story.
- Step 2 fixes found from saved rounds: sentence-initial words (Every, Now, They) were counted as names, and rewriting an over-long section did not
  shorten it (486 -> 498 words vs a 300 budget); now skipped and trimmed.
- Reporting bug, not fixed: the report's `final_average` is from the best draft but `final_faithfulness` is from the last iteration.
- Known limit: `retrieval_chunks` lists only the latest retrieval while merged facts may cite earlier chunks.

### Changes after the 2026-10-08 runs (not yet measured)
- Over-long sections are trimmed to whole sentences (1.5x budget) in the writer for `arc` and sectioned output; the attribution check skips
  sentence-initial words.
- Stalls (huntress 17-30 facts, doctor 11-17: faithfulness 4.0 on every draft, even after re-retrieves with more facts): low faithfulness with at least
  `Agentic_loop_rich_facts` (8) facts now REFINEs instead of RE_RETRIEVE. The refine feedback names the unsupported names per section
  (`section_failures`) and rewrites only those sections. Whispering (6 facts, faithfulness 2.0 -> 9.0 after re-retrieval) still re-retrieves.
- `Agentic_loop_stop_on_no_progress` (default on): two REFINEs in a row that do not beat the best draft by 0.5 stop the loop with
  `stop_reason: no_progress` (lawyer: 5.4 -> 4.6 -> 5.0 over 186 s).
- Measure script: `--repeat N` (mean +/- spread over runs), per-run judge averages, and `final_faithfulness` now comes from the returned draft
  (the old last-iteration value is `last_iteration_faithfulness`). Facts and stories are only saved because the script now runs with `debug=True`.
- To check: `--methods arc --repeat 3`; compare accept, seconds, judge average and faithfulness with the `arc` row above; huntress, doctor and lawyer
  should stop earlier or reach faithfulness >= 6.

### Result of the stall changes (`--methods arc --repeat 3`, 24 stories, 2026-10-08)
- accept 0.46 +/- 0.11, 1.67 iterations, 100 s, judge average 6.54 +/- 0.18, faithfulness 5.63 +/- 0.20. Stop reasons: 11 accepted, 12 `no_progress`, 1 max_iterations.
  Time fell from ~155 s, but accept did not improve (0.62 earlier, n=8), so the changes saved time, not quality.
- Outcomes are per query, not random: scholar, scientist and warrior were accepted on the first draft in 3/3 runs; huntress, doctor and goblin
  stopped on `no_progress` in 3/3; whispering and lawyer flipped. 10 of 24 stories were accepted on the first draft; of the 14 that went on, a refine pass rescued
  only 1 of 14 (the loop's later iterations mostly cost 40-60 s each for nothing).
- Hypothesis "unsupported names cause low faithfulness" is not supported: names missing from the facts show no relationship with faithfulness
  (sentence starts skipped: 1.8 for faithfulness < 6 vs 2.5 for >= 6; counting sentence starts: 18.8 vs 17.2, the opposite direction; goblin had 0-1).
  The differences are small against the spread, so read it as 'no evidence', not 'fewer'. The name feedback does not target the cause for these queries.
- Observed, not explained: stories scoring faithfulness < 6 had more facts (mean 13.7 vs 7.6; accepted queries had 6-9 facts). Within one query the
  score still swings (lawyer 9 / 4 / 4, doctor 2 / 6 / 2), so judge noise and fact count are confounded.
- Next measurement: `scripts/rejudge_saved.py` re-scores the saved stories several times, with the full facts and with the first 8 only. It separates
  judge noise (same story, different scores) from writer variance, and tests whether a long facts block hurts the 4B judge.

### Judge re-scoring of the saved stories (`scripts/rejudge_saved.py`, 24 stories x 3 calls x 2 fact sets, 2026-10-08)
- Mean faithfulness 5.62 with all facts vs 5.42 with the first 8 only: a long facts block is not what hurts the judge.
- Repeat calls on the same story agree most of the time (mean spread 0.42 full, 0.12 capped). The judge's faithfulness uses only four values
  (2, 4, 6, 9), so a flip moves 2-3 points, and flips happen near the boundaries (scientist 6/6/9, whispering 6/9/9, lawyer 4/6/4).
- The in-loop score disagreed with the re-score (median of 3) by 2-3 points on 8 of 24 stories, and matched on the other 16. A single in-loop
  score at the min-faithfulness threshold (6) therefore decides accept or reject partly by chance.
- Always-failing queries are stably low on re-scoring (huntress 2-4, goblin 2-4, doctor 4): the writer's drafts for those queries really score low;
  the judge is not inventing it. Whispering and lawyer are borderline queries.
- Reading: the earlier "judge noise" was mostly a coarse scalar plus boundary flips, not random scores. The judge says how bad but not where, which
  is why refine (generic suggestions) rescues so few drafts.
- Candidate next step (experiment script written, not yet run): `scripts/audit_saved.py`. A claim audit instead of a scalar, where the judge lists the sentences not supported by the facts, faithfulness
  comes from the count, and the refine rewrites exactly those sections quoting those sentences. Go/no-go on the saved stories: >= 80% of listed sentences occur verbatim in the story, stories scored < 6 get clearly more listed sentences than those >= 6, and the count moves <= 1 between repeats. Only then wire it into the loop.

- **Claim audit, first run (`scripts/audit_saved.py`, 24 stories x 2 audits, 2026-10-09):** 94% of 124 listed sentences appear verbatim in the story (PASS, need >= 80%);
  stories with faithfulness < 6 had 4.0 flagged sentences on average vs 1.4 for >= 6 (PASS; correlation -0.58); the count was identical on both audits for 17/24 stories
  and within 1 for 19/24, but one story moved by 4 (FAIL on the strict "<= 1" rule; mean change 0.58). Read by eye, the flags were mixed: wrong names and places
  were caught (huntress: a fox and a different camp; doctor: an apartment when the patient is elsewhere), while some goblin flags ("contradicts the ring fact")
  were weak.
- **Claim audit, second run (3 audits, 2026-10-09): no-go for count-driven control.** Verbatim 95% of 190 (PASS); faithfulness < 6 flagged 4.0 vs 1.3 (PASS, correlation
  -0.53); but the raw count still moved by up to 5 between audits (18/24 within 1) and keeping only sentences flagged twice in a row did not help (largest change between
  the two intersections: 6; separation 2.0 vs 0.8). Same 24 stories, same weak goblin flags. The audit is useful to read, not stable enough to accept or reject a draft.
  Not tested: using the flagged sentences only as refine feedback after the judge has already chosen REFINE.
- **Prompt review (2026-10-09, prompt-architect rubric, `prompts.yaml`):** changes only where a prompt had a measured gap. (1) Writers (`generation_arc.section_system`
  and the 5W1H system prompts): the huntress audit samples mixed four protagonists' names from facts retrieved from several stories, so the prompts now say to build one story
  around one protagonist and drop facts that do not fit, no new animals/rooms/objects/relatives/sub-plots, and the arc section prompt asks to use at least two fitting facts.
  (2) Judges (`loop_judge`, `with_story`): faithfulness is anchored on counts of unsupported sentences (none / 1-2 / 3-5 / 6-10 / most) instead of "8+ only if", aimed at the
  2/4/6/9 clustering seen in the re-scoring. (3) `claim_audit`: flag only with a cited fact number or a named new element, and do not flag elaboration. Placeholders and the
  strings asserted in `tests/test_config.py` are unchanged. Judge scores before and after this change are not comparable; re-score the saved stories with
  `scripts/rejudge_saved.py` before comparing a new `--repeat 3` run.
- **Prompt review, round 2 (2026-10-09):** the second pass was a clarity/completeness pass, not a response to a measured gap. Facts extraction (`generation.grounded_facts_*`)
  states its purpose, asks for stand-alone facts (one claim from one chunk, query-relevant first) and returns `{"facts": []}` when nothing fits. Writer and refine prompts
  share one priority order (never contradict a fact; add no new names or events; end on a complete sentence; then match the length) and ask for consistent names and roles.
  `ingest.section_label_*` defines each tag. `loop_judge` suggestions must name the section and quote the sentence start; `with_summary` gives 4 or less for an invented or
  missing ending. The rubric scores are my estimates, not measurements. The extraction prompt feeds Step 2, so run `scripts/retrieval_eval.py` against its floors first.

- **Audit fixes (2026-10-11, no effect on prompts or scores):** the file-based evaluate/summarise routes only read files inside the story, summary and evaluation output folders (`_safe_output_path`); loop thresholds accept `0` (`Agentic_loop_min_facts`, `_min_faithfulness`, `_accept_score`, `_rich_facts`, `_rule_max_novel_names` used to fall back to their defaults); a failed re-retrieve now stops with `stop_reason: retrieval_failed_using_best_so_far` and keeps the best draft; the unused `debug` and `eval_data` parameters were dropped from the loop. A second pass (same day): `Reranker_enabled` and `Hybrid_search_enabled` now default to on when the key is missing (a missing key used to switch them off silently; `retrieval_eval.py` not re-run yet); the model-loading helpers share one lock (`_load_lock.serialized`) so two threads cannot load the same model twice; the unused `cfg` parameter was dropped from `build_story_prompt` / `build_refine_prompt`.

- **Extraction-order finding (2026-10-12, extraction step only, not yet acted on):** with the control prompts, Step 2 took 0% of its facts from the top-ranked story when the chunks were listed in retrieval order (goblin to Cirilo, huntress to Zelea; 3 repeats each) and 31-43% when the list was reversed. The model also cites titles ("Blaze") instead of chunk ids on attempt 1, so the resolver drops that whole answer and the compact retry keeps only late-chunk facts. The three queries that always fail match short stories (2-3 chunks), which is where this matters; the queries that pass match one long story that fills all 10 chunks. The full-pipeline effect is not measured. Planned changes and order: `docs/INGEST_PLANNING.md`, section 7. **Root cause found 2026-10-15:** the chunk-order effect was prompt truncation. `build_chat_ollama` passed `options={"num_ctx": ..., "repeat_penalty": ...}` to `ChatOllama`, which langchain-ollama 0.3.10 silently ignores (verified: `num_ctx` and `repeat_penalty` stayed `None`), so Ollama ran at its default 4096-token context and dropped the head of longer prompts (the Ollama log showed `truncating input prompt`; the facts prompt is about 4,300+ tokens). Both are now real constructor fields (`/api/ps` shows 12288, 0 truncation warnings). `repeat_penalty` was also never applied before. A/B after the fix: retrieval order gave the target story 24-33% of facts (goblin) and 0-56% (huntress), against 0% before; reversing no longer helped (huntress 11-20%), so the reversal workaround was dropped. **All earlier baselines (0.46, 0.42, 0.34, 0.12) ran with this truncation and without a repeat penalty, so they are not comparable with runs from now on; re-measure the control first.** The length experiment that follows is described in `docs/INGEST_PLANNING.md` (finding the best story length).

- **Architecture pass (2026-10-15, behaviour unchanged; 230 tests pass):** (1) `generation_backend` warns when a prompt would not fit `num_ctx` (`_warn_if_prompt_too_long`) instead of letting Ollama truncate it silently. (2) `config.flag` and `config.num` replace the hand-written flag parsers and the `or <default>` idiom (every config flag now reads the same way: missing, empty or an unknown word give the default; `0` and `false` stay set). (3) New `rag/evidence.py` (`EvidenceSearch.gather / widen / preflight`, `Evidence`, `reformulate_query`): the retrieve + extract + merge + pre-flight steps that lived inside `run_agentic_story_loop` now sit behind one interface shared by both loops and the 3-step path. (4) Ingest, the maintenance scripts and retrieval use `vector_store/embeddings.py` (`get_embed_model`, `embed_texts`, `embed_passages`, `embed_query`); `ingest_stories._embed_chunks` / `_get_embed_model` and retrieval's `HuggingFaceEmbeddings` wrapper are gone. Checked: vectors are identical to the old code (cosine 1.000000 on queries and passages), so no re-ingest; `retrieval_eval.py` gives 0.80 / 0.867 / 0.733 with `Hybrid_bm25_pool: 0` (the documented baseline) and 0.80 / 0.833 / 0.717 with the live `setup.yaml` value of 12. (5) `Orchestrator` lost six parameters it accepted and ignored (`urls`, `generation_type`, `mode` on summary, `context` / `save` on `evaluate_story`, `context_path`); the API request models still accept those fields. Two internal calls in `run_pipeline` and `scripts/refresh_chunk_embeddings.py` still passed removed arguments and would have raised `TypeError`; both are fixed, and `tests/test_orchestrator_pipeline.py` now guards `run_pipeline`.

- **Judge correction (2026-10-15):** the live `setup.yaml` has no `Ollama_evaluation_model`, so `_build_ollama_evaluator` falls back to `Generative_model`: the judge is the same `qwen3.5:9b` as the writer (setup.yaml was last edited 2026-10-08 14:15, before the 0.46 baseline; earlier runs cannot be verified, the old logs do not record the judge). Wherever these notes say "4B judge" for the runs measured with this config, read "the 9B judging its own output"; a same-model judge is expected to be lenient and to share the writer's blind spots (unmeasured here). All runs since 10-08 share this setup, so comparisons between them are still like-for-like. The 4B was removed from Ollama; the example config now leaves `Ollama_evaluation_model` empty (= the writer model).
- **Control after the context fix (2026-10-15, old prompts, `--methods arc --repeat 3`, 24 stories):** accept 0.29 +/- 0.06, 2.04 iterations, 153 s, judge average 6.12 +/- 0.07, faithfulness 5.04 +/- 0.21; against 0.42 +/- 0.15, 1.79, 139 s, 6.43, 5.67 before the fix. Not better: now every prompt (facts, writer, judge) really gets 12288 tokens and the repeat penalty is applied. Which of the two causes it is not separated.

- **Story-first retrieval (2026-10-15, implemented, not yet measured):** `Story_first_retrieval: true` makes `EvidenceSearch.gather` keep only the lead story (the title of the top-ranked chunk), re-fetched with a Title filter for up to `Story_first_chunks` (8) chunks. Checked on the 8 default queries without the LLM: every query leads with the right story (Cirilo, Zelea, Country Doctor, Boris, and the long classics), one title each, short stories get their whole text (2-3 chunks). Motivation: after the context fix the extractor uses the right chunks (target-story fact share for huntress 0.00 to 0.63, goblin 0.00 to 0.31) but facts per story doubled to about 28 and short-story queries are dominated by other titles (warrior 1.00 to 0.22). To check: `--methods arc-story-first --repeat 3` against the control above (accept 0.29, faithfulness 5.04); right if target-fact share approaches 1.0 and facts per story fall, wrong if the share rises but accept does not move (then the writer or judge is the limit). Known limit: a wrong top-1 title builds the story from the wrong story.

- **A stuck Ollama runner hung a run for 1h40m (2026-10-16):** during the 27-story wiki dev run the Ollama log showed a request accepted and its prompt read (task 154803, 3,263 tokens) and then nothing; the script waited on the open connection with no timeout and nothing had been saved. A game was using the same GPU (VRAM 14.5 GB vs 9.5 GB in earlier runs); the cause is not proven. Fixes: every Ollama client now has a request timeout (`Ollama_request_timeout_s`, default 300 s) and a timeout becomes a `RuntimeError`, which the loops already handle (generation failure keeps the best draft; a judge failure falls back to completeness); `measure_generation_length.py` writes `<report>.partial.json` after every story and removes it when the method finishes. Close GPU-heavy apps before a long run.

- **Stage 1 result, story-first vs mixed list (2026-10-16, wiki-only store, 10 dev stories from 371 to 1,448 words, one run each, 9B judge, `--length long`):** accept 0.20 to 0.90, faithfulness 5.2 to 8.2, judge average 6.63 to 7.70, iterations 2.0 to 1.2, 131 s to 85 s per story; share of facts from the target story 0.46 to 1.00 (all 10 stories). Facts per story barely changed (22.8 to 19.0), so the gain is facts that are all from the right story, not fewer facts (the earlier guess of 6-15 facts was wrong). Caveats: n=10 and a single run; the store holds only the 37 wiki tales and each query is written from its own story (an easy setup); the judge is the writer model (lenient: Boris was accepted at faithfulness 9 with 4 facts behind a roughly 1,800-word story, and the generated Muriel story adds dialogue and scenery that are not in the 493-word source); the holdout set is untouched. Output is about 1.2x the target length in both arms. Next: output length (stage 2) with `arc-story-first`.

- **Stage 2 result, output length with story-first (2026-10-16, same 10 dev stories, one run each, 9B judge):** short (450 words): accept 1.00, faithfulness 8.4, judge 8.0, 60 s, 1.1 iterations; **medium (900): accept 1.00, 8.4, 8.0, 64 s, 1.0 iterations**; long (1,500): accept 0.80, 7.9, 7.2, 124 s, 1.7 iterations (Panko and Joe not accepted). Medium costs only about 4 s more than short for twice the words, because roughly 50 s per story is fixed (retrieval, facts, judge). Judge-independent check: capitalised names in the story that do not occur in its source story: 0.0 (short), 0.1 (medium), 0.1 (long) per story; the only offender is Boris (4 facts), one invented name at medium and one at long. Faithfulness did not vary with output/source word ratio (8.2, 8.4, 7.9, 8.5 for 0-1x, 1-1.5x, 1.5-2.5x, over 2.5x), so this judge cannot tell us how much invention a longer story hides. Recommended for the wiki tales: `--length medium` (source median 814 words). Not yet checked on the holdout set; the judge is the writer model.

- **Holdout and claim audit of the story-first setup (2026-10-16, 10 holdout stories, `--length medium`, 9B judge):** accept 0.80, faithfulness 7.1, judge 7.3, 87 s, 1.4 iterations (dev medium was 1.00 / 8.4 / 8.0 / 64 s, so the dev numbers were optimistic); 100% of facts come from the right story in all 10; Jess and Leo (the longest sources) were not accepted. Fact counts are bimodal and fixed per story: 18 of 50 story-first runs have 4-6 facts, 32 have 19 or more, none 8-17, and the same story gives the same count every run (9 of 20 stories are "thin"); thin stories score as well as rich ones (accept 0.94 vs 0.88). `audit_saved.py` on the holdout (2 audits each): 98% of the 55 flagged sentences occur verbatim in the story, about 2-4 flagged per story (roughly 3-7% of sentences), and thin-fact stories are not flagged more than rich ones (2.75 vs 2.75), so the worry that thin facts hide a lot of invention is **not supported**. The flagged sentences are mostly scene embellishment (steam, sparks) and a few direct contradictions of a fact (Talia, Jess). The audit count is still unstable (Jess 2 vs 7, max change 5) and does not separate faithfulness < 6 from the rest (n=2), so it stays a tool to read, not to gate on. Note: `claim_audit` exists only in the revised `prompts.yaml` on `main`; with the old control prompts the audit sends an empty prompt and returns nothing.

- **Story-first with the 35 classics back as distractors (2026-10-16, full 72-story store, same 10 holdout stories, `--length medium`):** lead story correct for all 37 wiki queries (37/37, same as the wiki-only store); generation accept 0.80, faithfulness 7.4, judge 7.5, 98 s, 1.6 iterations, against 0.80 / 7.1 / 7.3 / 86 s / 1.4 on the wiki-only store. Same stories accepted in both runs (Jess and Leo fail in both), 100% of facts from the right story in all 10. So the 35 distractors cost nothing here, and chunk headers or story cards have no measured need at this corpus size. Two slow stories (Layla 192 s, Leo 180 s) account for the extra seconds. The full store is `data/chroma_db`; the wiki-only store is kept at `data/chroma_db_wiki_only_2026-10-16`.

- **Revised prompts tested and reverted (2026-10-16):** the prompt-architect revision (writers, judges, facts, `claim_audit`) was run on the 10 holdout stories (full store, story-first, medium) against the control prompts: accept 0.40 vs 0.80, faithfulness 5.5 vs 7.4, 120 s vs 98 s, 2.0 vs 1.6 iterations; facts per story (17.6 vs 15.5) and length were about the same. Because the revised set also changes the judge, both story sets were re-scored with both judges (2 calls each, mean faithfulness): control stories 7.55 (revised judge) / 7.10 (control judge); revised stories 5.65 / 6.65. The revised judge is not simply stricter (it scored the control stories higher), and the revised-prompt stories score lower under both judges (-1.9 and -0.45; the second is within noise at n=10). Which edit hurts is not isolated (suspects: the writer wording on one protagonist, "at least two fitting facts", "checked sentence by sentence"). `prompts.yaml` is back to the f6578d3 prompts plus the `claim_audit` block (not used by the loop, needed by `audit_saved.py`); every other prompt is identical to f6578d3 (checked by comparing the parsed YAML). The revised version stays in git history (commit 08db11f).

- **Iteration cap 2 vs 3 (2026-10-16, 10 holdout stories, full store, story-first, medium):** accept 0.70 vs 0.80, faithfulness 7.7 vs 7.4, 91 s vs 98 s, 1.5 vs 1.6 iterations. Per story the cap saved Leo 80 s, Layla 77 s, Jess 13 s, and Layla was still accepted (on iteration 2). The one lost accept (Molly) was accepted on iteration 1 in the baseline and flipped on a borderline faithfulness of 6, so it is run-to-run noise, not the cap (the cap only affects iteration 3). The 7 stories that took one iteration in both runs went from 71 s to 86 s with the same setting, so the 7 s average saving is inside the noise; the per-story saving is about 17 s. Adopted `Agentic_loop_max_iterations: 2` (departing from the pre-set rule "keep 3 if the saving is under 10 s", on the per-story evidence). n=10, one run each.

### Next
1. Measure the changes above with `--repeat 3`.
2. Look at the `max_iterations` stalls (lawyer, huntress, doctor) with saved stories and facts.
3. Repeat `arc` over 3-5 seeds with frozen facts before claiming any gain (ADR-0010 follow-up).
After each change: `pytest`, `ruff`, `scripts/retrieval_eval.py` (floors top-1 0.80, top-3 0.833, fact coverage 0.733).
