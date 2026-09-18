# StoryForge RAG — Pattern Audit

> **Status: historical audit snapshot (2026-08-20), largely resolved.** Findings were true at
> audit time; most are no longer current. As of the last review pass:
> - **All 7 High-severity findings (H1–H7) are fixed** — collection-name wiring, the
>   `query_data` embedding mismatch, the `query_type` write/read mismatch, the retrieval
>   evaluator's schema mismatch, `n_results` being inert, `HF_evaluation_temperature` being
>   dead, and `Section_label_model` being ignored on the Ollama path.
> - Medium: M2 (`.env` path) and half of M3 (`Chroma_path` / `Story_input` default drift) are
>   fixed; M1's duplicated `ChatOpenAI` construction is fixed, but streaming still cannot run
>   the attribution gate or length guard mid-stream (architectural, not a bug); M4/M5/M6/M7
>   still stand as described.
> - Low: `Prompts_file`, the missing `vLLM_model` key, `Download_formats` being unreachable,
>   and the dead `download_archive_book_and_save_meta` function are fixed.
>   `Story_generation_rerank_top_n` is **less of a trap** now that `Story_generation_n_results`
>   defaults higher (10), but raising `rerank_top_n` alone above the diversity cap still only
>   reorders — the diversity-before-rerank order of operations is unchanged. Phantom
>   config-key fallbacks and a couple of duplicated helper pairs (JSON fence stripping,
>   `_maybe_tqdm`) are still present. Section-body helpers were consolidated into
>   `length_profile.py`.
> - Story length is now a unified `length` / `Story_length_*` profile (`length_profile.py`);
>   `Min_sentences_per_section` and `Agentic_loop_min_words*` are no longer hand-tuned config.
> - Streaming shares `build_story_prompt` with the non-streaming path (length guidance included);
>   it still cannot run the post-generation length guard or attribution gate mid-stream.
> - Empty-draft recovery: thinking → one fast retry → clear `RuntimeError`; length-guard and
>   agentic loop both catch that error (pre-refine draft / best-so-far) instead of aborting.
> - The Security note below about `.env` not loading is **superseded** — `secrets.py` now
>   resolves the repo root correctly. Still rotate any token that was pasted into chat.
> - The sections below are kept as-written for the historical record of what was found and why —
>   do not treat individual H/M/L items as current bugs without checking the code first.
> - Prefer [`docs/README.md`](docs/README.md) and [`docs/PROJECT_JOURNEY.md`](docs/PROJECT_JOURNEY.md) for current truth.

**Date:** 2026-08-20
**Scope:** all 36 Python files under `src/storyforge/`, `main.py`, `scripts/`, `setup.yaml`, `setup.example.yaml`, `prompts.yaml`

Every finding below was **verified by reading the actual code**, not inferred. File:line references are exact.

## What "the same pattern" means here

Two bugs were fixed earlier today. Both had the same shape:

1. **`ChatOllama` built in two places** (`generation_backend.py` + inline in `orchestration_routes.py`) → copies drifted → wrong kwarg (`think=` vs `reasoning=`) + missing `num_ctx` → **empty generations**.
2. **`Model_max_prompt_tokens` read but never wired** to the client → config looked active, did nothing.

So the pattern is really three related failure modes:

- **A. Duplicated logic that drifts** — same thing written twice, copies diverge.
- **B. Config read but never wired** — the knob looks live, changes nothing.
- **C. Silent failure** — the break is swallowed by a bare `except` or a `.get()` default, so nothing surfaces.

The audit below hunts all three.

---

# HIGH — silently wrong right now

## H1. `Chroma_collection_name` is inert for the pipeline

**Pattern:** B + A

`orchestrator.py:20` reads the key correctly:
```python
self.chroma_collection_name = c.get("Chroma_collection_name") or "StoryForgeRag_v1"
```
But the pipeline steps ignore that value and hardcode the literal:
```python
orchestrator.py:164   res_reset  = self.reset_vector_store(new_collection_name="StoryForgeRag_v1")
orchestrator.py:170   res_ingest = self.ingest_stories(collection_name="StoryForgeRag_v1")
```
Meanwhile `retrieval.py:77` *does* read the config key.

**Consequence:** set `Chroma_collection_name: "MyStories"` and the pipeline resets + ingests into `StoryForgeRag_v1`, while retrieval queries `MyStories` — an empty collection. Zero results, no error, stories generated from no context.

**Why it matters to you now:** you have 1457 chunks in `StoryForgeRag_v1`. The moment you rename the collection, everything silently detaches.

---

## H2. `query_data()` is the 384-vs-768 bug you already hit — but silent

**Pattern:** B + C

This is the *same root cause* as the `ingest_manifest.py` crash from earlier in this session, on a different code path — except this one is swallowed instead of raised.

`chromadb.py:35-38` creates the collection with **no `embedding_function`**:
```python
return _Client.get_or_create_collection(
    name=name,
    metadata={"hnsw:space": "cosine"},
)
```
Ingest writes precomputed **768-dim BGE** vectors. But `chromadb.py:50-57` queries by **text**:
```python
results = Collection.query(
    query_texts=[query],          # <-- forces Chroma's DEFAULT embedder
    ...
)
```
With no embedding function registered, Chroma falls back to `all-MiniLM-L6-v2` (**384-dim**) → dimension mismatch → exception → swallowed at `chromadb.py:59-61`:
```python
except Exception as e:
    logging.error(f"Error querying data: {e}")
    return None
```

**Verified against your live DB:** `collections` table reports `dimension = 768`; `collection_metadata` contains only `hnsw:space=cosine` — no embedding function. Confirmed.

**Blast radius (important):** only `query_vector_result` (`response_parameter.py:192`) uses this path.
- **Story generation is NOT affected** — it uses `retrieve_docs` in `retrieval.py`, which builds its own LangChain `Chroma` with the correct BGE embeddings. That's why your story generated fine.
- **Affected:** the `/orchestration` vector-query endpoint and the retrieval evaluator.

---

## H3. `query_type` mismatch — one insert path writes rows nothing can ever read

**Pattern:** A (magic string duplicated, drifted)

| Site | Value |
|---|---|
| `response_parameter.py:232` (insert) | `"query_type": "Title"` |
| `ingest_stories.py:228` (ingest) | `"query_type": "content"` |
| `records_to_manifest.py:74` (manifest) | `"query_type": "content"` |
| `chromadb.py:50,55` (query) | filters `"content"` |
| `vector_store_check.py:322` (query) | filters `"content"` hardcoded |

**Consequence:** anything inserted via `Orchestrator.vector_store_insert` is written successfully and is then **permanently invisible to every query in the app**. No error at write time; it just never comes back.

---

## H4. The retrieval evaluator always reports 0% — it reads keys nobody writes

**Pattern:** A (two independently-written schemas for the same object)

Producer, `response_parameter.py:200-215`, emits:
```python
{"id": rid, "text": d, "metadata": {series_id, series_name, volume_number,
                                    chapter_id, chapter_number, chapter_name,
                                    section, character}}
```
Consumer, `retrieval_eval.py:88-95`, reads:
```python
title    = str(meta.get("Title",  item.get("Title", "")))       # no such key
author   = str(meta.get("Author", item.get("Author", "")))      # no such key
document = str(item.get("document", item.get("documents", ""))) # producer emits "text"
```

**Consequence:** every `RetrievalHit` comes back `title=""`, `document=""`. `_same_title("", expected)` is always False → `top1_match`/`top3_match` always False → `fact_coverage` always `0.0`.

**The retrieval evaluation harness reports 0% no matter how good retrieval actually is.** It cannot detect a regression, and it has presumably never produced a meaningful number.

Compounding: it also measures the **broken** `query_data` path (H2), not the retriever story generation actually uses.

---

## H5. `n_results` does nothing — and `gen_params` reports it as if it did

**Pattern:** B + "reported ≠ used" (the exact class I fixed for `temperature` and missed here)

`retrieval.py:280-281`:
```python
base_k = int(cfg.get("Story_generation_n_results") or 3)
k = max(base_k, n_stories * max(1, chunks_per_story) * 4)
```
Every caller passes `n_stories=3, chunks_per_story=2` (`langchain_rag.py:62`), so the second term is **always 24**. The API bounds `n_results` at `le=20` (`create_eval_routes.py:34-39`), so `base_k` can never exceed 24 → `max()` always discards it.

Then `_select_diverse_stories(docs, n_stories=3, chunks_per_story=2)` (`retrieval.py:298`) caps the final set at **6 chunks**.

**Consequence:** `n_results: 20` and `n_results: 1` produce **identical** results. And `response_parameter.py:288` still reports `"n_results": int(n_results)` in `gen_params`, claiming a retrieval breadth that was never applied.

I fixed the `temperature`/`top_p` half of this misreporting today. This is the other half of the same field.

---

## H6. `HF_evaluation_temperature: 0.1` is never applied — evaluation always runs at 0.3

**Pattern:** B (dead code path)

```python
evaluation.py:188   def evaluate_model(temperature: float = 0.3, ...):
evaluation.py:70    "temperature": float(temperature if temperature is not None
                                         else cfg.get("HF_evaluation_temperature", 0.1)),
```
All four callers pass **no arguments** (`response_parameter.py:465,481,495`; `agentic_loop.py:371`), so `temperature` is always `0.3` and never `None` — the `cfg.get(...)` branch is unreachable. Line 97's read is likewise shadowed because the `"temperature"` key is already present in the dict.

**Consequence:** rubric scoring runs 3× noisier than configured, which directly perturbs the agentic loop's ACCEPT / REFINE / RE_RETRIEVE decisions (`agentic_loop.py:215-248`).

Note `HF_evaluation_max_new_tokens` on the adjacent config line **is** correctly wired — which is exactly why the dead sibling is easy to miss.

---

## H7. `Section_label_model` is inert on the default Ollama path

**Pattern:** B (argument accepted then dropped)

```python
enrich_records.py:56-61
def _load_section_labeler(cfg, model_id):
    if use_ollama_for_generation(cfg):
        llm = load_ollama_llm(cfg, max_new_tokens=24, temperature=0.0, top_p=0.8)
        return ("ollama", llm, None)      # model_id never used
    gen, tok = _load_local_section_labeler(model_id)
```
`load_ollama_llm` resolves the model itself via `ollama_model_id(cfg)` = `Generative_model`.

**Consequence:** with `Generation_provider: ollama` (your default), `Section_label_model` is silently ignored. Masked today because both are `qwen3.5:9b` — but point it at a small cheap model and you'll still run the 9B for every 24-token label call.

---

# MEDIUM

## M1. The streaming endpoint is still a fork of Step 3 — and skips two guardrails

`/orchestration/generate_stream` duplicates the Step-3 prompt assembly (`orchestration_routes.py:352-364` vs `generation.py:311-339`) but **omits** the final line of `generate_from_facts`:
```python
generation.py:342   return _apply_attribution_gate(story, parsed.facts, cfg)
```
It yields raw tokens straight from `llm.astream`.

**Consequence:** the streaming endpoint bypasses the attribution gate and the `Min_sentences_per_section` guard entirely. `Attribution_gate_truncate`, `Attribution_violation_threshold`, and `Min_sentences_per_section` have no effect on streamed output — two endpoints, same nominal pipeline, different safety behaviour.

Also: I extracted `build_chat_ollama()` today, but the **vLLM `ChatOpenAI` constructor three lines above it is still an inline duplicate** (`orchestration_routes.py:372-380` vs `generation_backend.py:195-203`). Identical today; same trap as before.

## M2. `.env` is loaded from the wrong directory

`secrets.py:7`:
```python
_ROOT = Path(__file__).resolve().parent      # = src/storyforge/config/
```
so `load_dotenv(_ROOT / ".env")` looks for `src/storyforge/config/.env`. Every other module resolves the repo root via `parents[3]`.

**Verified:** no `.env` exists in either location on your machine, so nothing is broken today — but this is *why* putting the HF token in `setup.yaml` was the only thing that worked.

## M3. Same config key, different defaults in different files

| Key | Site A | Site B | Risk |
|---|---|---|---|
| `Chroma_path` | `retrieval.py:76` → `"chroma_db"` | `chromadb.py:24,103` → `"data/chroma_db"` | omit the key → ingest and retrieval use **different directories** |
| `Story_input` | `ingest_stories.py:179` → `"Stories"` | `story_records.py:35` → `"data/stories"` | omit the key → ingest looks in a directory that doesn't exist |
| `HF_evaluation_model` | `extraction.py:93` → `Qwen2.5-**7B**` | `evaluation.py:62` → `Qwen2.5-**1.5B**` | omit the key → extraction and evaluation silently run models 4.7× apart in size |
| `Story_generation_n_results` | `retrieval.py:280` → `3` | `create_eval_routes.py:18` → `1` | API default contradicts pipeline default |

All masked *only* because the keys are currently set in `setup.yaml`.

## M4. `Generation_no_repeat_ngram_size` is Transformers-only

Read at `generation.py:176`, used at `:191` — inside the **Transformers branch only**. `build_chat_ollama` passes only `repeat_penalty` + `num_ctx`; `load_vllm_llm` passes only `frequency_penalty`.

Its sibling `Generation_repetition_penalty` **is** wired to all three backends, so the pair looks symmetric but isn't. The config comment claims *"applies to both fast and thinking modes"* — true, but it doesn't mention it applies to neither of your actual backends.

## M5. `Single_pass_refine_max_tokens` doesn't apply to your active path

Read only at `langchain_rag.py:85` — the single-pass 3-step path. The agentic loop (active, since `Agentic_loop_enabled: true`) computes its own budget at `agentic_loop.py:359-363,501`: `base_max_tokens + refine_token_boost`.

**Consequence:** refine passes use `3200 + 600 = 3800`, not the configured `3600`.

## M6. BGE passage prefix deviates from the trained convention

- Ingest (`ingest_stories.py:49`): `"Represent this passage for retrieval: "` prepended to every passage.
- Query (`retrieval.py:100`): `"Represent this sentence for searching relevant passages: "`.

BGE v1.5's documented recipe puts an instruction on **queries only**; passages go in unprefixed. Two separately-authored literals in two files with no shared constant.

Not catastrophic — the prefix is applied uniformly to all passages, so relative ranking among them is largely preserved — but it is off-convention and the two literals must be changed together. Changing `Vector_store_model` on one side without re-ingesting silently mismatches the index.

## M7. `Model_max_prompt_tokens` — residual after today's fix

The original wiring bug is fixed, but two residuals remain in the helper I added:
1. `num_ctx` is Ollama's **total** window (prompt + output). `ollama_num_ctx()` doesn't subtract `num_predict`, despite its comment saying it leaves room for output. Effective prompt budget is `12288 − 3200 = 9088`.
2. It's applied to the Ollama backend only — vLLM and Transformers still ignore it.

---

# LOW

- **`Prompts_file` is dead config.** Declared in both YAMLs, read nowhere; the path is hardcoded at `config/config.py:17`.
- **Phantom keys** — read in code, absent from both YAMLs, so the hardcoded fallback always wins and the knob is undiscoverable: `Ollama_model`, `GENERATIVE_MODEL`, `Extracted_text_dir`.
- **`setup.example.yaml` drift.** `config.py:26-28` falls back to the example file when `setup.yaml` is missing, so a fresh clone runs on it. It's missing `Generated_story_output` / `Generated_summary_output` (fresh clones write to `Generated_Stories/` and `Summarized_Stories/` instead of `data/outputs/...`), and `Agentic_loop_min_words` differs 4.4× (`1100` vs `250`).
- **`vLLM_model` missing from `setup.yaml`.** Switching `Generation_provider: vllm` falls back to `Generative_model` = `"qwen3.5:9b"` — an Ollama tag, not an HF repo id. Would fail immediately.
- **`Download_formats` unreachable.** Read at `fetch_book.py:184-185` only inside `if formats is None:`, and no caller ever passes `None`.
- **`Story_generation_rerank_top_n: 6` never filters.** Rerank runs *after* `_select_diverse_stories` already capped the set at 6, so `ranked[:6]` on a ≤6 list only reorders. Raising it above 6 is a no-op.
- **`_flow_section_headers(cfg)`** accepts `cfg` and ignores it, implying a config hook that doesn't exist.
- **Duplicated helpers, identical today but drift-prone:** `_split_section_bodies` / `_sentence_count` / `_SECTION_HEADER_RE` in both `generation.py` and `agentic_loop.py`; `_maybe_tqdm` in `fetch_book.py` and `extract_text.py`; JSON-fence stripping in `attribution.py:42-58` (regex, case-insensitive, repairs trailing commas) vs `evaluation.py:223-231` (literal slicing, case-sensitive, no repair).
- **`download_archive_book_and_save_meta` (`fetch_book.py:271-326`) is dead code** — nothing imports it. It's a stale fork of the live `download_book_archive` and has already drifted from it.

---

# Security

**`setup.yaml:24` contains a live Hugging Face token in plaintext.**

I put it there at your request earlier today. It's gitignored, so it won't reach version control — but it has also been pasted into this conversation. **Rotate it** at huggingface.co → Settings → Access Tokens, and prefer the `STORYFORGE_HF_API_KEY` env var going forward (note M2 — repo-root `.env` isn't currently loaded, so set it in the shell).

---

# Suggested order of work

1. **H1 + H3** — one-line-ish fixes, both prevent silent data loss.
2. **H5 + H6 + H7** — three inert config keys; each is a small, contained fix.
3. **H2 + H4** — these two are the same underlying problem (one hit schema, one retriever). Fixing `query_data` to use the same embedding path as `retrieve_docs`, and making producer/consumer share one schema, repairs the vector-query endpoint *and* makes the retrieval evaluator meaningful for the first time.
4. **M1** — route the streaming endpoint through `generate_from_facts` instead of forking it; extract `build_chat_openai()` alongside `build_chat_ollama()`.
5. **M3** — one `config_defaults.py` with a single authoritative default per key removes four verified divergences at once.
