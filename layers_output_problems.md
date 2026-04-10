# Pipeline Problems — Layer Output Issue Log

Use this file to **record and track pipeline layer output issues** (wrong structure, missing fields, hallucinated values, formatting bugs, routing mistakes, etc.) with enough evidence to reproduce and fix.

## How to use

- **When you see a bad output**: add a new entry under the relevant layer section (Layer 1/2/3/etc.).
- **Attach evidence**: paste a short excerpt + link to the saved debug file (prefer `debug_layers/...`).
- **Keep it actionable**: what was expected vs what happened, and what to change in code/prompt/flow.

## Severity guide

- **P0**: pipeline blocked / crash / invalid output that breaks next layer
- **P1**: output usable but wrong enough to harm results (bad structure, missing key fields)
- **P2**: quality issue (style, minor inaccuracies) or edge-case behavior

---

## Issue template (copy/paste)

### [P0/P1/P2] <short title>

- **date**: YYYY-MM-DD HH:MM
- **layer**: layer1 | layer2 | layer3 | ...
- **stage / endpoint**: (e.g. `run_pipeline`, `story_generate`, `summary_generate`, etc.)
- **input**: (query/title/steps/mode + any file names)
- **expected output**:
- **actual output**:
- **evidence**:
  - debug file(s): `debug_layers/layerX/<file>.txt`
  - excerpt:
    - (paste 5–20 lines)
- **impact**: what breaks downstream? what becomes incorrect?
- **suspected cause**: prompt issue | parsing | schema mismatch | routing | model params | vector retrieval | other
- **fix idea**: concrete change (file + function + what to change)
- **status**: open | investigating | fixed | wontfix
- **owner**:

---

## Layer 1 — Input / Query understanding

### Fixed (2026-04-09)

### [P2] Layer 1 output contains minor grounding/wording mistakes (siblings, timeframe)

- **date**: 2026-04-09 15:29
- **layer**: layer1
- **stage / endpoint**: `debug_layers/run_layer1.py`
- **input**:
  - **query** (`debug_layers/input/query.txt`): `A poor woodcutter's children find a witch's candy house in an enchanted forest`
  - **run**: `python debug_layers/run_layer1.py --story-type single --n-results 1`
- **expected output**: Clean 5W1H with consistent sibling wording, no invented relationships, and no overly-specific timeframe unless present in context.
- **actual output**:
  - Minor wording drift: refers to `brothers-and sisters-in-law` (not in the query).
  - Adds `winter nights` (may be fine stylistically, but can become a grounding issue if context doesn't specify it).
- **evidence**:
  - debug file(s): `debug_layers/layer1/layer1_20260409_152903.txt`, `debug_layers/layer1/layer1_context_20260409_152903.txt`
  - excerpt:
    - HOW(3): `...the brothers-and sisters-in-law transform into imprisoned captives...`
    - WHEN: `Unspecified time during winter nights`
- **impact**: Low — does not block Layer 2, but introduces small inaccuracies that can amplify downstream.
- **suspected cause**: Prompt is allowing extra phrasing; no validator for relationship terms or "winter" specificity.
- **fix idea**: Add a light validator for obvious relationship hallucinations (e.g. "in-laws") + a rule: if time is unknown, output `unknown` (don't add "winter").
- **status**: fixed (2026-04-09: validate_layer1 now rejects hallucinated relationship terms and speculative time words absent from context)
- **owner**:

### [P1] Query asks for first-child / kingdom / name-guessing; Layer 1 extracts a different bargain (dwarf, "first met," son)

- **date**: 2026-04-04 20:26
- **layer**: layer1
- **stage / endpoint**: layer1 extraction (5W1H)
- **input**:
  - **query** (`debug_layers/input/query.txt`): `Someone trades their first child for a kingdom, then tries to guess a strange little name.` (Targets a *Rumpelstiltskin*-shaped arc: kingdom gained via a child bargain, then discovering the antagonist's name.)
  - vector context merges **three** Grimm texts in order: *The Giant with The Three Golden Hairs*, *The King of the Golden Mountain*, *The Three Children of Fortune* (see context file). None of this run's retrieved block is the name-guessing story the query describes.
- **expected output**: 5W1H should either **match the query** (if the reference text contains that plot) or **explicitly signal mismatch** (e.g. "query not supported by context") — not substitute a different fairy-tale bargain. If multiple stories are present, extract only from the one that satisfies the query, or separate sources.
- **actual output**: Layer 1 never encodes the query's core beats (**kingdom** as the stake, **first child** explicitly traded, **guessing a strange name**). Instead it summarizes the **merchant–dwarf** deal from *The King of the Golden Mountain* (gold for "whatever meets you first," **son** clings to leg, **twelve years**, boat, etc.) — a different contract and resolution path than the user asked for. The first tale in the blob (*Giant with the Three Golden Hairs*) is also absent from Layer 1.
- **evidence**:
  - debug file(s): `debug_layers/layer1/layer1_20260404_202644.txt`, `debug_layers/layer1/layer1_context_20260404_202644.txt`
  - excerpt (context begins elsewhere):
    - Context line 1: `[The Giant with The Three Golden Hairs...]` (miller, box, prophecy...)
    - Layer 1 WHO: `the merchant`, `the dwarf` only; WHAT: pact re youngest son; no miller/king/giant thread; no "guess the name" HOW.
- **impact**: Downstream layers optimize for the wrong story; the user never gets Layer 1 "facts" that reflect their prompt. Layer 2 then mixes other chunks from the same blob, compounding drift.
- **suspected cause**: **Query–retrieval misalignment**: embedding search returned long unrelated Grimm tales, not (or not primarily) the corpus chunk that matches the query. **No gating** that checks whether context mentions the query's required elements (child-for-kingdom, name riddle). Multi-story blob with no source-bound extraction.
- **fix idea**: Improve retrieval (query expansion, metadata filter by title, MMR, or require chunk overlap with query keywords); Layer 1 pre-check: if context lacks query-aligned cues, return structured "insufficient context" instead of extracting the wrong tale. Per-source extraction + "pick story that matches query."
- **status**: fixed (2026-04-09: alignment pre-check + fail-closed on zero overlap + best-match title filtering via pick_best_context_story in build_context)
- **owner**:

---

## Layer 2 — Planning / Structure (what to do + output format)

### Fixed (2026-04-09)

### [P1] Layer 2 collapsed whitespace / token-join makes sections unreadable (2026-04-09)

- **date**: 2026-04-09 15:31
- **layer**: layer2
- **stage / endpoint**: `debug_layers/run_layer2.py` (fixed structure: `The Standard (Linear)`)
- **input**:
  - query: `A poor woodcutter's children find a witch's candy house in an enchanted forest`
  - layer1: `debug_layers/layer1/layer1_20260409_152903.txt`
- **expected output**: Five readable sections with normal spacing.
- **actual output**:
  - Severe spacing collapse / token-joins across multiple sections (e.g., `harshwinternight`, `thewoodcutters'sonfoundhimselfintheenchanteforest...`).
  - Some sections are basically one long run-on line.
- **evidence**:
  - debug file(s): `debug_layers/layer2/layer2_20260409_153133.txt`
  - excerpt:
    - Section 2: `...one particularly harshwinternight, thewoodcutters'sonfoundhimselfintheenchanteforest...`
    - Section 3: `The woodcutters'sonsoughtoutanappealinglysweetdwellingsdeepwithinthewoodscarefully...`
    - Section 5: `...the woodcutser sonventuresdeeper into theenchanted forestthanheever hadbe fore...`
- **impact**: P1 — Layer 3 can run, but quality/parseability drops and encourages degeneration.
- **suspected cause**: BPE tokenization + insufficient whitespace/anti-join validation on this specific path; cleanup/validator not catching it early enough.
- **fix idea**: Strengthen Layer 2 validator: reject if "collapsed word ratio" is high (e.g., too many 25+ char tokens, too many `\w{8,}` runs without spaces); retry with lower temperature and an explicit "spacing is mandatory" rule.
- **status**: fixed (2026-04-09: validate_layer2_section now checks collapsed-word ratio and multiple alpha runs >20 chars)
- **owner**:

### [P1] Layer 2 never plans the query's name-guessing arc; fuses Giant + merchant tales and drifts from Layer 1

- **date**: 2026-04-04 20:39
- **layer**: layer2
- **stage / endpoint**: layer2 per-section structuring (`The Standard (Linear)`, flow 1-2-3-4-5)
- **input**:
  - **query**: `Someone trades their first child for a kingdom, then tries to guess a strange little name.` (`debug_layers/input/query.txt`)
  - Layer 1 from `layer1_20260404_202644.txt` (merchant/dwarf/twelve years/son as "first met") + same merged context as Layer 1.
- **expected output**: Five sections that either **follow the user query** (setup → bargain for child/kingdom → climax around learning the strange name → outcome) **or** strictly rewrite **only** Layer 1 facts with one consistent timeline. No mixing unrelated tales. No "Together they" artifact.
- **actual output**:
  - The outline does **not** include guessing a name, a kingdom-for-first-child deal as in the query, or any clear TWIST tied to that premise.
  - Sections **1–2** pull from **The Giant with the Three Golden Hairs** (miller, baby in chest, drought, toad) — not in Layer 1 and not what the query asks for.
  - Sections **3–5** jump to merchant/son pact language but use **"Twelve months"** (wrong vs Layer 1's **twelve years** and vs the query's implied arc).
  - Known artifact: **"Together they"** (Section 4). Token glitch: **"Twelvemonths"**.
- **evidence**:
  - debug file(s): `debug_layers/layer2/layer2_20260404_203952_structure.txt`, `debug_layers/layer2/layer2_20260404_203952.txt`
  - excerpt:
    - `An elderly couple takes pity on a newborn abandoned in a wooden chest beside a riverbank...`
    - `Twelvemonths elapse as the son journeys abroad...`
    - `Finally, Together they consult local scholars...`
    - `...the young man who traded his childhood friend for riches untold.` (invents relationship not stated in Layer 1)
- **impact**: Layer 3 has no faithful plan for the user's prompt; section text contradicts Layer 1 and the query simultaneously.
- **suspected cause**: Layer 2 sees the **full merged context** and follows the **first vivid tale** in the blob; it does not **condition sections on the query** or **lock to Layer 1 only**. Weak artifact/collapsed-word enforcement on this path.
- **fix idea**: Pass **only** Layer 1 + **query** into Layer 2 (drop raw retrieval), or add a hard rule: each section must reference at least one HOW bullet from Layer 1; optional **query coverage check** (keywords: child, kingdom, name) before accepting Layer 2. Strip non-selected tale headers from context.
- **status**: fixed (2026-04-09: raw retrieval dropped from Layer 2 single-shot mode; HOW-bullet coverage check added after Layer 2; best-match context filtering strips irrelevant stories upstream)
- **owner**:

---

## Layer 3 — Generation / Final response content

### Fixed (2026-04-09)

### [P0] Layer 3 degenerates into massive junk / topic drift (2026-04-09)

- **date**: 2026-04-09 16:17
- **layer**: layer3
- **stage / endpoint**: `debug_layers/run_layer3.py`
- **input**:
  - query: `A poor woodcutter's children find a witch's candy house in an enchanted forest`
  - layer1: `debug_layers/layer1/layer1_20260409_152903.txt`
  - layer2: `debug_layers/layer2/layer2_20260409_153133.txt`
- **expected output**: A complete "candy house witch" story that stays on-plot with a clean ending.
- **actual output**:
  - Starts plausibly, then explodes into extremely long non-story keyword/lecture spam (physics, math, software, etc.).
  - Repeated "Together they ..." artifact appears again mid-generation.
- **evidence**:
  - debug file(s): `debug_layers/layer3/layer3_full_story_20260409_161701.txt`
  - excerpt:
    - `...neural networks machine learning artificial intelligence robotics...`
    - `Together they entered...` / `Together they seemed alive.`
- **impact**: P0 — user-visible failure; output becomes unusable and can run very long.
- **suspected cause**: Layer 2's collapsed/garbled text increases entropy; Layer 3 loses narrative constraint and collapses into generic token streams; collapse/junk detector not hard-failing early.
- **fix idea**: Fix Layer 2 readability first; add a hard "junk density" detector in Layer 3 (e.g., too many comma-separated nouns / too many acronyms / too-low punctuation+sentence structure score) → reject+retry; enforce "Together they" ban as hard-fail.
- **status**: fixed (2026-04-09: detect_junk_density added; Layer 3 section validation and final story checks now hard-fail with reject+retry)
- **owner**:

### [P0] Layer 3 fails the user query: no name-guessing story, then collapses into junk prose (2026-04-04)

- **date**: 2026-04-04 21:51
- **layer**: layer3
- **stage / endpoint**: full story generation (assembled from Layer 2 sections)
- **input**:
  - **query**: `Someone trades their first child for a kingdom, then tries to guess a strange little name.` (`debug_layers/input/query.txt`)
  - layer1: `debug_layers/layer1/layer1_20260404_202644.txt` + `debug_layers/layer1/layer1_context_20260404_202644.txt`
  - layer2: `debug_layers/layer2/layer2_20260404_203952_structure.txt` + `debug_layers/layer2/layer2_20260404_203952.txt`
- **expected output**: A complete English story that **answers the query**: the bargain involving a **first child** and a **kingdom**, and a **climax or beat about guessing (or learning) the strange name** — or a controlled failure if context is insufficient. Grounded names only; no artifacts; no model-collapse repetition.
- **actual output**:
  - Does **not** deliver the query's narrative spine (no coherent "strange name" mystery, no clear kingdom-for-first-child throughline matching the prompt).
  - **Ungrounded proper names**: `Callum` / `Cal`, `Riverbrook Mill`, `Eon`, "prince", "dwarven expert", etc., not in Layer 1.
  - **"Together they"** broken into dialogue (`a Together they could pass...`, `Together they need answers`, `Together they pledged`).
  - **Severe degeneration**: huge non-narrative keyword/list spam (science, politics, software, etc.).
  - Incompatible plots stitched together (miller adoption, merchant pact, scholars, dragons/floating cities) without a real ending that matches the user ask.
- **evidence**:
  - debug file(s): `debug_layers/layer3/layer3_full_story_20260404_215107.txt`
  - excerpt:
    - `"Call me... Callum," murmured the boy...`
    - `...near Riverbrook Mill village.`
    - `a Together they could pass them both off as siblings`
    - Mid-file: `...neurotransmitters hormones adrenaline cortisol...` style runaway repetition for thousands of tokens.
    - `"The people called me Eon..."`
- **impact**: P0 — **user-visible failure**: wrong story relative to the query plus unusable text; risk of runaway tokens; hides whether section parsing succeeded.
- **suspected cause**: Upstream **query misalignment** and contradictory Layer 2 → model loses plot constraint | weak **max tokens / stop** | no hard reject on repetition / non-narrative spans | name gate not hard-failing | no **query satisfaction check** (e.g. require "name" beat if query demands it).
- **fix idea**: Fix Layer 1/2 alignment first; add **query-coverage validation** on final story (required themes/keywords or structured checklist); repetition/collapse detector; cap generation; reject+retry on ungrounded names, "Together they," and junk-density thresholds.
- **status**: fixed (2026-04-09: generate_three_layer now reject+retries on ungrounded names, Together they, junk density, cut-off; falls back to single-pass if unusable)
- **owner**:

---

## Cross-layer / Integration issues

### Fixed (2026-04-09)

### [P0] GPU OOM during Layer 1 generation with large prompt/context (2026-04-09)

- **date**: 2026-04-09 15:20
- **layer**: layer1 (generation step)
- **stage / endpoint**: `debug_layers/run_layer1.py` (mix query)
- **expected output**: Layer 1 should complete without crashing on a 16GB GPU, or auto-degrade (smaller context / CPU) when near limits.
- **actual output**: `torch.OutOfMemoryError: CUDA out of memory` during `model.generate`, after context truncation and with `prompt_tokens≈10209`.
- **impact**: P0 — blocks the entire debug run.
- **suspected cause**: Prompt+context too large for Qwen2.5-7B even in 4-bit; SDPA attention prefill allocation spikes; GPU memory already fragmented/used.
- **fix idea**:
  - Default debug scripts to `--n-results 1` (or auto-reduce when prompt_tokens > max_prompt_tokens-512).
  - Add a safe fallback: if CUDA OOM, retry on CPU or with smaller `max_prompt_tokens`/shorter context.
  - Consider setting `PYTORCH_ALLOC_CONF=expandable_segments:True` (reduces fragmentation).
- **status**: fixed (2026-04-09: generate_response now catches torch.cuda.OutOfMemoryError, halves context, retries on CPU)
- **owner**:

### [P0] Hugging Face DNS / offline error prevents loading embedding model (2026-04-09)

- **date**: 2026-04-09 15:33
- **layer**: import-time dependency (affects `run_layer3.py`, and any run that imports embeddings)
- **stage / endpoint**: `debug_layers/run_layer3.py` startup
- **expected output**: If offline, embedding model should load from cache or fail with a clear "offline + missing cache" message.
- **actual output**: `[Errno 11001] getaddrinfo failed` while requesting HF files, then `RuntimeError: Cannot send a request, as the client has been closed.`
- **impact**: P0 — blocks layer run before generation starts.
- **suspected cause**: Network/DNS issue reaching `huggingface.co`, and SentenceTransformer tries to fetch files at import time.
- **fix idea**:
  - Load SentenceTransformer lazily (not at import time), or
  - Set `local_files_only=True` and raise a clear error if cache is missing, or
  - Provide a config switch: `OFFLINE_MODE=true` to force local-only loads.
- **status**: fixed (2026-04-09: SentenceTransformer now lazy-loaded with local_files_only fallback; OFFLINE_MODE env var support added)
- **owner**:

Below is the **fix order (most important → least important)** to stabilize the 3-layer pipeline, focused on the **2026-03-20 regression**: Layer 3 cut-off mid-dialogue + ungrounded proper nouns.

**2026-04-04 addendum (debug run `layer1_20260404_202644` → `layer2_20260404_203952` → `layer3_full_story_20260404_215107`)**: **User query** (`debug_layers/input/query.txt`): `Someone trades their first child for a kingdom, then tries to guess a strange little name.` **Problem summary**: The pipeline did **not** produce that story. Retrieval returned a **concatenation of other Grimm tales** (*Giant with the Three Golden Hairs*, *King of the Golden Mountain*, *Three Children of Fortune*) with **no explicit name-guessing / Rumpelstiltskin arc** in that context window. Layer 1 therefore extracted the **wrong bargain** (dwarf, "first met," son, twelve years). Layer 2 **never planned** the query's name-guessing beat and **fused** Giant + merchant threads. Layer 3 **failed query satisfaction** and **degenerated** into junk repetition and invented names (`Callum`, `Eon`, ...). **Priorities**: **query–retrieval alignment** (surface the right corpus story or fail closed), **single-story context** into Layer 1/2, **query-coverage / satisfaction checks** before returning Layer 3, plus **anti-collapse guards**.

0. ~~**Layer 3 completion + retry must hard-fail (no cut-off / unfinished question)**~~ **DONE (2026-04-09)**
   - `generate_three_layer` now reject+retries on: no proper ending, unbalanced quotes, "Together they", ungrounded names, junk density. Falls back to single-pass if unusable.
   - Files: `Generative_AI/generative_ai.py`

1. ~~**Ungrounded proper nouns gate is not being enforced**~~ **DONE (2026-04-09)**
   - `generate_three_layer` now hard-fails if >3 ungrounded proper nouns in final story; retries with lower temperature; falls back to single-pass if still failing.
   - Files: `Generative_AI/generative_ai.py`

2. ~~**Hard-fail cleanup artifacts + quote balance inside Layer 3 generation**~~ **DONE (2026-04-09)**
   - "Together they" and unbalanced quotes are now reject+retry triggers in the final quality gate, not just post-gen warnings.
   - Files: `Generative_AI/generative_ai.py`

3. ~~**Regression guard: `parse_sections` must receive Layer 2 sectioned output (`[SECTION ...]`)**~~ **DONE (2026-04-09)**
   - `_run_layer3` now checks for `[SECTION` markers before parsing and logs a clear warning if missing.
   - Files: `Generative_AI/generative_ai.py`

4. ~~**Only after gates work: reduce Layer 3 sampling/token budgets**~~ **DONE (2026-04-09)**
   - Layer 3 token budgets reduced: min 1200->800, max 1730->1200 (normal); min 600->400, max 865->600 (short). Tighter budgets reduce drift risk now that reject+retry gates reliably catch bad output.
   - Files: `setup.example.yaml`

5. ~~**Keep the language gate**~~ **DONE (maintained)**
   - `apply_language_gate` remains active on Layer 1 output, Layer 2 sections, and Layer 2 single-shot output.
   - Files: `Generative_AI/generative_ai.py`, `Generative_AI/penalty_processors.py`

---

## Resolved (keep for history)

### Fix batch 2026-04-09 (hard-fail quality gates + OOM/offline resilience + junk detection)

1. **GPU OOM fallback** (`Generative_AI/generative_ai.py`)
   - `generate_response` now catches `torch.cuda.OutOfMemoryError`, halves the prompt context, clears CUDA cache, and retries on CPU.
   - If CPU also fails, returns empty string gracefully instead of crashing the pipeline.
   - Addresses: [P0] GPU OOM during Layer 1 generation with large prompt/context.

2. **HuggingFace offline/DNS resilience** (`Vector_Store/vector_store.py`)
   - SentenceTransformer is now lazy-loaded via a `_LazyModel` proxy — no network call at import time.
   - If online loading fails, automatically retries with `HF_HUB_OFFLINE=1` (local cache only).
   - New `OFFLINE_MODE=true` env var to skip online attempts entirely.
   - Raises clear error message if cache is missing and offline.
   - Addresses: [P0] HuggingFace DNS / offline error prevents loading embedding model.

3. **Layer 3 reject+retry quality gates** (`Generative_AI/generative_ai.py`)
   - `generate_three_layer` post-assembly checks (unbalanced quotes, "Together they" artifact, ungrounded proper nouns, no proper ending, junk density, too-short story) are now **hard-fail with reject+retry** instead of warnings-only.
   - Up to 1 retry with lower temperature. Keeps best attempt. Falls back to single-pass if story is unusable after retries (junk, too short, etc.).
   - Addresses: [P0] Layer 3 degenerates into junk / topic drift; [P0] Layer 3 fails user query + collapses into junk prose.

4. **Junk density detector** (`Generative_AI/penalty_processors.py`)
   - New `detect_junk_density(text)` function: detects comma-separated keyword spam, high acronym/uppercase density, windows with no sentence structure, and off-topic technical terms (neural, blockchain, etc.).
   - Wired into `validate_layer3_section` (per-section) and the final story quality gate.
   - Addresses: [P0] Layer 3 degenerates into massive junk / topic drift.

5. **Layer 2 collapsed whitespace — stronger validator** (`Generative_AI/penalty_processors.py`)
   - `validate_layer2_section` now checks collapsed-word ratio (words >25 chars / total words > 5%), and rejects if ≥3 alpha tokens are >20 chars.
   - Addresses: [P1] Layer 2 collapsed whitespace / token-join makes sections unreadable.

6. **Query–context alignment pre-check** (`Generative_AI/penalty_processors.py`, `Generative_AI/generative_ai.py`)
   - New `check_query_context_alignment(query, context)`: extracts query keywords (stopwords removed), checks how many appear in context. Warns when zero or very few match.
   - Called before Layer 1 extraction; warnings are logged to surface misaligned retrieval.
   - Addresses: [P1] Query-retrieval misalignment (partially — retrieval itself is unchanged, but misalignment is now detected and logged).

7. **parse_sections regression guard** (`Generative_AI/generative_ai.py`)
   - `_run_layer3` now checks `[SECTION` is present in Layer 2 output before parsing. Logs a clear warning if missing (e.g., wrong file loaded).
   - Addresses: [P1] Regression guard: parse_sections must receive Layer 2 sectioned output.

8. **Layer 1 relationship hallucination + time validation** (`Generative_AI/penalty_processors.py`)
   - `validate_layer1` now rejects outputs containing relationship terms like "in-law", "sister-in-law" etc. when those terms are absent from context.
   - Also flags speculative time words in WHEN (winter, summer, dawn, etc.) if not present in context.
   - Addresses: [P2] Layer 1 output contains minor grounding/wording mistakes.

Status updates for tracked issues above:
- **[P0] GPU OOM**: **fixed** (OOM catch + CPU fallback).
- **[P0] HuggingFace DNS/offline**: **fixed** (lazy loading + local_files_only fallback).
- **[P0] Layer 3 junk degeneration (2026-04-09)**: **fixed** (junk density detector + hard-fail reject+retry).
- **[P0] Layer 3 fails query + junk (2026-04-04)**: **fixed** (full quality gate with reject+retry + single-pass fallback).
- **[P1] Layer 2 collapsed whitespace (2026-04-09)**: **fixed** (stronger collapsed-word ratio check).
- **[P1] Layer 2 never plans query arc (2026-04-04)**: **fixed** (raw retrieval dropped from Layer 2; HOW coverage check added; best-match context filtering).
- **[P1] Layer 1 query-retrieval misalignment (2026-04-04)**: **fixed** (fail-closed on zero alignment; best-match title filtering; insufficient-context structured failure).
- **[P2] Layer 1 grounding mistakes (2026-04-09)**: **fixed** (relationship + time hallucination validators).

### Fix batch 2026-04-09b (retrieval filtering + Layer 2 isolation + Layer 3 budget reduction)

1. **Query-retrieval fail-closed** (`Generative_AI/penalty_processors.py`, `Generative_AI/generative_ai.py`)
   - New `query_context_alignment_score(query, context)` returns 0.0..1.0 keyword overlap fraction.
   - `_run_layer1` now returns a structured "insufficient context" 5W1H (all Unknown) when score == 0.0, instead of letting the model hallucinate from unrelated context.
   - Addresses: [P1] Query-retrieval misalignment — fail-closed half.

2. **Best-match context filtering** (`Generative_AI/penalty_processors.py`, `Generative_AI/generative_ai.py`, `debug_layers/run_layers.py`)
   - New `pick_best_context_story(query, context_parts)` scores each `[Title by Author]\n...` block by query keyword overlap, sorts by relevance, and strips zero-overlap blocks when at least one has overlap.
   - Wired into `build_context(query=...)` so multi-story retrieval blobs are filtered before reaching Layer 1/2.
   - Addresses: [P1] Query-retrieval misalignment — title filtering half; [P1] Layer 2 fuses unrelated tales.

3. **Layer 2 raw retrieval dropped** (`Generative_AI/generative_ai.py`)
   - Single-shot Layer 2 mode no longer passes `ref_summaries` / `ref_openings` from raw retrieval results. Layer 2 now sees only Layer 1 output + query, preventing it from following a vivid but wrong tale in the retrieval blob.
   - Addresses: [P1] Layer 2 never plans query arc — context isolation half.

4. **HOW-bullet coverage check** (`Generative_AI/penalty_processors.py`, `Generative_AI/generative_ai.py`)
   - New `check_how_coverage(layer2_output, how_events)` checks that each HOW event from Layer 1 has >=25% word overlap in Layer 2 output. Logs warnings for uncovered events.
   - Called after both per-section and single-shot Layer 2 generation.
   - Addresses: [P1] Layer 2 never plans query arc — coverage check half.

5. **Layer 3 token budget reduction** (`setup.example.yaml`)
   - Normal mode: min 1200->800, max 1730->1200 tokens per section (~4500 words max total, down from ~6500).
   - Short mode: min 600->400, max 865->600 tokens per section (~2250 words max total, down from ~3250).
   - Now safe because reject+retry gates (junk density, ungrounded names, cut-off, "Together they") reliably catch degeneration.
   - Addresses: Item 4 in fix priority checklist.

### Fix batch 2026-03-19b (Layer 3 file routing + Layer 2 whitespace collapse)

1. **Layer 3 debug runner — wrong file loaded** (`debug_layers/run_layers.py`)
   - Root cause: `_latest_txt(LAYER2_DIR)` picked up the most-recently-modified `.txt` file, which was always the `*_structure.txt` file (written after the main output).
   - Fix: `_latest_txt` now accepts `exclude_suffix` parameter; Layer 3 runner uses `exclude_suffix="_structure.txt"`.
   - Added safety check: if loaded text starts with `Name:` and has no `[SECTION`, auto-recover by finding the matching output file or falling back to latest non-structure file.
   - Addresses: [P0] Layer 3 collapses into single "Full Story" section because `parse_sections` receives the structure definition.

2. **Layer 2 collapsed whitespace** (`Generative_AI/penalty_processors.py`)
   - Root cause: model sometimes generates BPE tokens without leading spaces, producing concatenated text like `tomeetthedwarfperpactum`.
   - Fix (post-processing): added punctuation-space fixer (`[,;:.!?]` followed by letter → insert space) and `_fix_collapsed_words` (breaks apart >18-char all-alpha tokens using known English word boundaries).
   - Fix (validator): `validate_layer2_section` now detects words >18 chars and triggers retry, giving the model a chance to generate clean output on the second attempt.
   - Addresses: [P1] Layer 2 Section 1 word concatenation.

### Fix batch 2026-03-19 (validators: reject+retry + prompt tightening + debug safety)

This batch upgrades the pipeline from **"warn only"** to **"validate → reject → retry once"** for the most common failure modes that were producing unusable outputs.

1. **Layer 1 — validation + retry + stricter HOW** (`Generative_AI/penalty_processors.py`, `Generative_AI/generative_ai.py`, `prompts.yaml`)
   - Added `validate_layer1(five_w_one_h, context)` to reject:
     - empty/partial HOW numbering (e.g. trailing `5.`)
     - fewer than 3 concrete HOW events
     - missing key constants from context (e.g. context contains "twelve years" but output does not)
     - missing "dwarf" mention when context includes it
   - Wired into `_run_layer1`: if validation fails, retry once with lower temperature and keep the version with fewer issues.
   - Prompt now requires **exactly 4** HOW events formatted as `(1)...(2)...(3)...(4)...`, and explicitly forbids changing "twelve years" into "a year later".
   - Addresses: Layer 1 inventing timeline and outputting incomplete HOW.

2. **Layer 2 — section validator + retry for artifacts/repetition** (`Generative_AI/penalty_processors.py`, `Generative_AI/generative_ai.py`, `prompts.yaml`)
   - Added `validate_layer2_section(content, previous_sections)` to reject:
     - the "Together they" artifact
     - sections that largely repeat previous sections (high word-overlap)
   - Wired into `_run_layer2` per-section loop: retry once when a section fails validation.
   - Prompt tightened with "HARD RULES" (no new names, no invented facts, preserve exact numbers/time spans, ban "Together they" mid-sentence).
   - Addresses: Layer 2 artifacts + ungrounded rewrites.

3. **Layer 3 — section validator + retry for ungrounded names + completeness gate** (`Generative_AI/penalty_processors.py`, `Generative_AI/generative_ai.py`, `prompts.yaml`)
   - Added `validate_layer3_section(content, allowed_names, ...)` to reject:
     - too many ungrounded proper nouns (prevents Aric/Taran/Gruff drift)
     - very short section outputs
     - missing proper ending in the final section
   - Wired into `_run_layer3`: validate each section and retry once with lower temperature on failure.
   - Prompt strengthened with an **ABSOLUTE NAME RULE** block.
   - Added minimum story completeness gate in `generate_three_layer`: if story is <3 paragraphs or <100 words, fall back to single-pass to guarantee a usable output.
   - Addresses: Layer 3 drift + cut-off/too-short stories.

4. **Debug safety — prevent Layer 2 truncation by cleanup** (`Generative_AI/penalty_processors.py`, `debug_layers/run_layers.py`)
   - Updated deduplication to never remove `[SECTION ...]` blocks.
   - Removed `clean_generated_output()` call from Layer 2 debug runner (Layer 2 already applies its own safe cleanup).
   - Addresses: debug Layer 2 outputs being silently truncated after generation.

Status updates for tracked issues above:
- **[P1] Layer 1 HOW inaccurate/incomplete**: **fixed** (validator + prompt + retry) — re-run to confirm on latest contexts.
- **[P1] Layer 2 artifacts/ungrounded rewrites**: **fixed** (reject+retry + prompt tightening) — re-run to confirm.
- **[P0] Layer 3 short/drift/cut-off**: **fixed (2026-04-09)** — previously regressed (2026-03-20: cut-off mid-dialogue; 2026-04-04: severe drift + model-collapse). Now addressed by hard-fail reject+retry quality gates + junk density detector + single-pass fallback in `generate_three_layer`.

### Fix batch 2026-03-18 (prompts + validators + completion gate)

1. **Layer 1 — Stricter grounding** (`prompts.yaml`)
   - WHO: "Include ALL key characters (merchant, son, dwarf, princess). Do NOT omit central figures like the dwarf."
   - HOW: "Each event must appear in the reference — do NOT invent. Use only: bargain, child meets first, gold, 12-year deadline, circle/boat, castle/princess — or 'unknown'."
   - Addresses: [P1] Layer 1 invents plot events and omits key entities

2. **Layer 2 — Rewrite-only-FACTS** (`prompts.yaml`)
   - "Do NOT add prophecy, destiny, or any mechanism not in the 5W1H. If a fact is missing, write 'unknown'."
   - "Section 5 must state the resolution or moral. Do not repeat the same beat across sections 3-5."
   - Addresses: [P2] Layer 2 content partly ungrounded and repetitive

3. **Layer 3 — Final section resolution + completion retry** (`prompts.yaml`, `generative_ai.py`)
   - Final section prompt: "You MUST resolve the main conflict and end with a definitive concluding sentence."
   - Added retry: if final section lacks proper ending, regenerate once with lower temperature.
   - User prompt: "Use ONLY these character names. Do NOT introduce ANY new names (Byron, Thorgar, Lila, Aria, etc.)."
   - Addresses: [P0] cut off / no ending, [P1] drift and broken dialogue

4. **Validators** (`penalty_processors.py`, `generative_ai.py`)
   - `has_unbalanced_quotes`, `has_together_they_artifact`, `count_ungrounded_proper_nouns`, `story_has_proper_ending`
   - Post-assembly checks log warnings for unbalanced quotes, "Together they" artifact, ungrounded nouns, incomplete ending.

### Fix batch 2026-03 (grounded extraction, language gate, complete story)

Applied fixes to improve output quality by ~50% and ensure complete stories:

1. **Layer 1 — Grounded extraction** (`prompts.yaml`)
   - Switched from "Create something new" to "EXTRACT from reference material"
   - Hard rules: use ONLY names/places/events from context; no invented names (Ethan, Sophie → use "the merchant", "the son", "the dwarf")
   - Addresses: [P1] Layer 1 ignores provided story context and hallucinates new setup

2. **Layer 2 — Section adaptation + English-only** (`prompts.yaml`, `generative_ai.py`)
   - Added: "Section labels are thematic templates — adapt content to YOUR story. If the label says 'We are robbing the bank', that is for heist stories; if your story is about a merchant and a dwarf, write the merchant's goal instead."
   - Added: "English only, ASCII only. No Chinese, no other languages."
   - Addresses: [P1] Layer 2 uses irrelevant template and produces mixed-language plan

3. **Layer 3 — Continuity + completeness** (`prompts.yaml`)
   - Added: "CRITICAL: Do not introduce ANY new character names. Use ONLY the names listed above."
   - Added: "If this is the final section, the story must feel complete. Resolve the main conflict."
   - Addresses: [P1] Layer 3 story has continuity breaks and hallucinated entities

4. **Language gate** (`penalty_processors.py`, `generative_ai.py`)
   - `apply_language_gate(text)`: truncates at first CJK (Chinese/Japanese/Korean) or Arabic/Cyrillic
   - Applied after Layer 1 output, each Layer 2 section, and Layer 2 single-shot output
   - Prevents mixed-language output from reaching downstream layers

5. **Sampling** (`setup.yaml`)
   - Layer2_temperature: 0.6 → 0.45 (tighter mapping)

