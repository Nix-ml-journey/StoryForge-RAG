# Planning what you ingest

[`DATA_PREP.md`](./DATA_PREP.md) tells you **how** to clean a text and run the ingest scripts. This page is about the step before that: **which stories to ingest, and what shape they should have**, so the stories that come out stay grounded.

Everything here is tied to a measurement from this project. Where it is only a guess, it says **hypothesis**. Numbers are from the corpus on 2026-10-12 (72 stories).

---

## 1. What the pipeline does with your corpus

Knowing this makes the rules below obvious.

1. Your query retrieves about **10 chunks**, ranked by relevance. Retrieval picks at least 3 different titles (up to 2 chunks each) and then fills up to 10 chunks in rank order.
2. A local model (Step 2) turns those chunks into numbered facts.
3. The writer (Step 3) may use **only** those facts. It aims for about **1,500 words** (the `long` target).
4. A judge checks the story against the facts.

So the story is only as good as the facts, and the facts are only as good as the chunks retrieved. **If the story you want is short, most of the 10 slots go to other stories.** That is how blended protagonists happen.

---

## 2. What we measured

| Finding | Evidence |
|---|---|
| The corpus is very uneven. | 37 of 72 stories are short wiki-style tales (median 814 words, 3 chunks). The classics are long (median 3,675–17,490 words). One book is 69,417 words. |
| 22 stories have 3 chunks or fewer. | 9 have only 2 chunks (Blaze 371 words, Fini 378, Ina 438, Cirilo 487, Muriel 493, ...). |
| The queries that always failed all match short stories. | Goblin → Cirilo (2 chunks). Huntress → Zelea. Doctor → Kafka's Country Doctor. The queries that pass match one long story that fills all 10 slots (Cthulhu, Reanimator, Whisperer, Jekyll). |
| Retrieval finds the right story. | Cirilo, Zelea and Country Doctor are ranked first today. The data and the chunks are fine. |
| The facts step then ignores the first chunks. | With chunks in retrieval order, the extractor took 0% of its facts from the top story (3 of 3 repeats, goblin and huntress). With the same chunks reversed, it took 31–43%. |

What this does **not** show: that every short story fails (Boris, 591 words, passes), or that fixing the extractor alone will fix accept rate. That test is still pending (see ADR-0011).

---

## 3. Rules of thumb for a story

### Length: give each story enough material
- Aim for **roughly the output length or more**. For a 1,500-word story, that is about 1,500 words and **6 or more chunks** (a chunk is about 1,800 characters).
- A story under about **600 words (2 chunks)** gives the retriever almost nothing to hold on to. The other 8 slots go to other stories.
- *Hypothesis:* the 6-chunk / 1,500-word figure is a sensible target, not a measured threshold. Boris passes at 2 chunks, so it is not a hard line.

If a tale is short and you want to keep it, either:
- merge it with a closely related tale into one longer file (only if it is really one arc), or
- keep it, but expect it to blend with neighbours and write queries that name it specifically, or
- ask for a shorter story length for it (`length: short`).

### One story = one protagonist = one file
- Do not put two tales in one `.txt`. Do not ingest a collected-works file.
- Name the main character the same way throughout. The writer follows the facts, and facts that call the same person three different names turn into three people.

### Make each story easy to tell apart
- Give the lead character a **distinct name** and avoid reusing names across files.
- Be careful with near-duplicates. Layla, Zelea, Luana and Solaine all share tribe / orphan / camp ideas, and they compete for the same queries.
- Put the **concrete nouns** that people will search for in the first chunk: the role (huntress, goblin, doctor), the place, the species, the central object. Retrieval matches what is written, not what you meant.

### Check that the text matches your test query
Before you trust a failing query, search the corpus for its key words:

```powershell
Select-String -Path data/stories/*.txt -Pattern "goblin" -List | ForEach-Object { $_.Filename }
```

If nothing in the corpus is about "a goblin explosives maker", no pipeline can write that story from facts. If a story is, but it still fails, the problem is later in the pipeline (see section 5).

---

## 4. Before you ingest: extra checklist

On top of the JSON checklist in `DATA_PREP.md`:

- [ ] The file is **one story**, with a stable filename stem (that stem is the Title, and chunk ids are built from it).
- [ ] Word count is known, and you accept what it means (section 3).
- [ ] The first chunk names the role, place and central object.
- [ ] No other file in the corpus is a near-duplicate of this one.
- [ ] You wrote **2 or 3 test queries** that this story should answer, using the words a user would use.
- [ ] Added those queries to `tests/fixtures/retrieval_eval_cases.example.json` with the expected title.

Do not touch the chunking code (`ingest_stories.py`, lines 150–240). If a story chunks badly, fix the text (paragraph breaks) rather than the chunker.

---

## 5. After you ingest: three checks

**1. What landed**

```powershell
.\.venv\Scripts\python.exe scripts/validate_chroma_metadata.py
```

**2. Retrieval finds it.** Run your test queries and read the ranks:

```powershell
.\.venv\Scripts\python.exe scripts/retrieval_eval.py --cases tests/fixtures/retrieval_eval_cases.example.json --k 3
```

Floors to hold: top-1 ≥ 0.80, top-3 ≥ 0.833, fact coverage ≥ 0.733.

**3. The facts actually use it.** This is the check that was missing. Retrieval can rank the right story first and the writer can still never see it, because Step 2 ignored its chunks. After a generation run, count where the facts come from:

```powershell
.\.venv\Scripts\python.exe -c "import json,re,collections; r=json.load(open('Evaluation/generation_eval_report_arc.json',encoding='utf-8'))['records'][0]; print(collections.Counter(re.sub(r'_chunk_\d+$','',f['source_chunk_ids'][0]) for f in r['grounded_facts']))"
```

If your target story has **0 facts** but ranks first in retrieval, the loss is in Step 2, not in your data.

---

## 6. Typical symptoms and the likely cause

| Symptom | Likely cause | Where to fix |
|---|---|---|
| The story is about the wrong character | Facts blended from several short stories | Section 3 (length, distinct names) |
| Retrieval ranks the right story first, but its facts are missing | Step 2 favours the last chunks in the list | Extraction (pending, ADR-0011), not ingest |
| Judge says faithfulness 2–4 on a query that "should work" | The corpus has no story for it, or the facts do not contain it | Search the corpus first |
| A query only works with the exact title | The first chunk lacks the role / place words | Rewrite the opening |
| New story never shows up | Filename stem is wrong, or the JSON was stale | `DATA_PREP.md`, "Title contract" |

---

## 7. Target setup: the plan (written 2026-10-12, nothing applied yet)

This is the order we would try the changes in, one at a time, each followed by `measure_generation_length.py --methods arc --repeat 3`. Only step 1 has a measurement behind it.

| # | Change | Type | Re-ingest? | Evidence |
|---|---|---|---|---|
| 1 | Put the most relevant chunk **last** in the facts prompt (or extract per story / per chunk). **Superseded 2026-10-15:** the real cause was Ollama truncating the prompt at 4096 tokens (see the ADR-0011 note); with the context fixed, reversing the order stopped helping, so it was not kept. | 1 line in `extraction.py` | No | Measured on the facts step: 0% of facts from the top story in retrieval order, 31-43% reversed (goblin and huntress, 3 repeats). Effect on accept rate **not measured**. |
| 2 | **Story-first retrieval:** pick the best story, then give Step 2 all of its chunks, instead of 10 chunks from 3 or more titles. | Code in `retrieve_docs` and the loop | No | Hypothesis. Targets the blended-protagonist failure. |
| 3 | **Story cards and chunk headers:** one card per story (title, protagonist, role, setting, summary) indexed beside the chunks; a `Title / character` header on each chunk so facts name their subject. | Ingest code + data | Yes (`reset_and_ingest.py`) | Hypothesis. Standard practice for narrative RAG, untested here. |
| 4 | Lengthen or merge very short stories (under about 600 words / 2 chunks). | Corpus | Yes | Hypothesis. Section 3. |

Not changing: chunk size (about 1,800 characters, 200 overlap) and the chunking code. No `setup.yaml` setting fixes this on its own; lowering `Story_generation_n_results` would only reduce the blending a little.

### Finding the best story length (experiment, written 2026-10-12, not run yet)

The corpus mixes two things: nearly all short stories are wiki-style, and all long ones are classics. So length and style are confounded. The sample therefore holds a length gradient **inside each style**: 20 wiki stories (6 under 600 words, 6 at 600-1000, 6 at 1000-1500, 2 above 1500) and 22 classics (1 under 600, 2 at 600-1000, 1 at 1000-1500, 4 at 1500-2600, 5 at 2600-6000, 9 above 6000). It includes the 8 default queries so it connects to the earlier baselines. Some classic cells have only 1-2 stories, so read them as hints.

```powershell
.\.venv\Scripts\python.exe scripts/measure_generation_length.py --methods arc --queries-file Evaluation/length_sample_queries.json
.\.venv\Scripts\python.exe scripts/analyze_length_buckets.py
```

`Evaluation/length_sample_map.json` holds query, target story, words and style; `analyze_length_buckets.py` prints accept rate, faithfulness, judge average, the share of facts that come from the target story and iterations per style and length bucket. It takes about 80-100 minutes for one pass of 42 queries; `--repeat 2` doubles that and halves the noise.

**It is right to pick a length range if** accept rate and target-story fact share stay high inside it and drop outside it, in both styles. **Length is not the lever if** the buckets look alike once the target-story fact share is near 1.0; then the answer is the pipeline, not the files.

## 8. Known gaps

- The extractor ordering problem above is **measured on the facts step alone** (3 repeats, two queries). The effect on final accept rate has not been measured.
- Known retrieval misses (Frankenstein, Body Snatcher) are unchanged and are not caused by story length.
- The 6-chunk guideline is a hypothesis.
