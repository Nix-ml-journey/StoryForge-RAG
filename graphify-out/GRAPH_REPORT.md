# Graph Report - RAG  (2026-09-28)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 898 nodes · 1946 edges · 49 communities (43 shown, 6 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 96 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d7267ed1`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 54 edges
2. `resolve_length_profile()` - 38 edges
3. `Orchestrator` - 27 edges
4. `salvage_grounded_facts_json()` - 25 edges
5. `run_agentic_story_loop()` - 24 edges
6. `decide_action()` - 20 edges
7. `Gen_mode` - 19 edges
8. `ingest_stories_dir()` - 18 edges
9. `_stream_story_sse()` - 18 edges
10. `extract_grounded_facts()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `load_config()`  [INFERRED]
  scripts/peek_vector_store.py → src/storyforge/config/config.py
- `main()` --uses--> `Orchestrator`  [INFERRED]
  scripts/measure_generation_length.py → src/storyforge/orchestrator/orchestrator.py
- `_run_one()` --uses--> `Orchestrator`  [INFERRED]
  scripts/measure_generation_length.py → src/storyforge/orchestrator/orchestrator.py
- `main()` --uses--> `Gen_mode`  [INFERRED]
  scripts/measure_generation_length.py → src/storyforge/rag/generative_ai.py
- `_run_one()` --uses--> `Gen_mode`  [INFERRED]
  scripts/measure_generation_length.py → src/storyforge/rag/generative_ai.py

## Import Cycles
- None detected.

## Communities (49 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (78): _flush(), main(), _delete_dir(), main(), Path, reset_and_ingest.py ------------------- Run this once after upgrading to BGE…, Try to delete a directory. Returns True on success, False if locked., Delete and recreate the collection using the ChromaDB client directly. (+70 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (56): AgenticLoopResult, average_score(), build_feedback(), completeness_report(), CompletenessReport, criterion_score(), decide_action(), Decision (+48 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (55): Chroma, fastapi_responses, langchain_chroma, langchain_huggingface, logging, generate_stream(), GenerateStreamRequest, orchestration_status() (+47 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (46): copy, epub_to_text, fitz, functools, os, main(), extract_text_from_books(), extract_text_from_epub() (+38 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (42): Protocol, Core SSE generator: Steps 1–2 sync in thread, Step 3 streams via Ollama., _stream_story_sse(), build_chat_ollama(), build_chat_openai(), generation_provider(), GenerationLLM, invoke_combined_prompt() (+34 more)

### Community 5 - "Community 5"
Cohesion: 0.10
Nodes (41): asyncio, FastAPI, create_app(), pydantic, create_eval_status(), _default_n_results(), BaseModel, post (+33 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (34): _auto_stub(), fixture, _auto_stub(), _cfg(), _fake_hf_response(), fixture, Tests for Step 2 grounded-facts extraction (extraction.py). Covers: - JSON mode…, By default, Qwen3's hidden chain-of-thought is discouraged two ways:… (+26 more)

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (22): main(), Validate what an ingest actually wrote to Chroma (offline, no HF, no GPU).…, fetch_all(), format_report(), _pct(), problems(), Any, Chroma metadata health report (pure functions, no Chroma import). Used by… (+14 more)

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (23): Resolve length → mode default → Story_length_default → built-in preset., resolve_length_profile(), Tests for the story length target. Pure arithmetic and config resolution — no…, Single_pass_*_max_tokens stays a floor so manual tuning is not lost., A 12-minute script at 140 wpm needs ~1680 words., The whole point: one target moves all three levers in the same direction.…, A near-target draft must pass; a half-length draft must not., test_accept_gate_sits_below_the_target_but_above_half() (+15 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (22): huggingface_hub, ParsedFacts, extract_grounded_facts(), _extract_grounded_facts_local(), _grounded_facts_provider(), _hf_chat_extract_json(), _hf_chat_extract_json_with_retry(), _hf_token() (+14 more)

### Community 10 - "Community 10"
Cohesion: 0.13
Nodes (21): parametrize, Best-effort parse of messy local-model facts output. Returns ``(ParsedFacts,…, salvage_grounded_facts_json(), _strip_think_blocks(), extraction(), fixture, Tests for the LOCAL grounded-facts path (no HF): salvage parser + extraction…, test_local_json_format_config() (+13 more)

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (21): format_facts_for_prompt(), Numbered bullet list for the generation prompt (empty if no facts). Example: 1.…, _get_generation_prompts(), Load prompt templates from prompts.yaml (generation section)., build_refine_prompt(), build_story_prompt(), _flow_section_headers(), generate_from_facts() (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (9): pathlib, CLI wrapper for `storyforge.scripts.check_cuda_compatibility`. Run: py…, main(), Step 1b (actual ingest): - Reads `data/ingest/ingest_manifest.jsonl` - Upserts…, CLI wrapper for `storyforge.evaluation.retrieval_eval`. Run from repo root: py…, storyforge_evaluation_retrieval_eval, storyforge_scripts_check_cuda_compatibility, storyforge_scripts_list_gemini_models (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.23
Nodes (15): _FakeCollection, ingest(), run(), fixture, Path, ingest_stories_dir() must honour reviewed story_json records (P0.1).…, test_crlf_only_difference_is_not_stale(), test_matching_story_json_supplies_chunks_sections_and_metadata() (+7 more)

### Community 15 - "Community 15"
Cohesion: 0.18
Nodes (14): ast, collections, FunctionDef, body_fingerprint(), main(), parse(), py_files(), Path (+6 more)

### Community 16 - "Community 16"
Cohesion: 0.14
Nodes (16): datetime, src_storyforge_data_step1_prepare_and_enrich, src_storyforge_evaluation_evaluation, _eval_to_dict(), evaluate_story_file_result(), evaluate_story_text_result(), evaluate_summary_result(), generate_summary_result() (+8 more)

### Community 17 - "Community 17"
Cohesion: 0.14
Nodes (11): pytest, _cfg(), Tests for empty-draft recovery in generation.py and langchain_rag.py. Covers: -…, If both thinking and fast-mode attempts return empty, RuntimeError is raised., Fast mode returning empty also raises RuntimeError (no retry loop for fast)., If the length guard refine call raises RuntimeError, the original draft is…, When thinking mode returns empty, generate_from_facts retries once with fast…, test_both_attempts_empty_raises_runtime_error() (+3 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (6): storyforge_evaluation, _FakeGemini, test_evaluate_model_falls_back_to_gemini_when_hf_unavailable(), test_invoke_with_retry_routes_local_provider_to_local_invoker(), test_invoke_with_retry_uses_gemini_fallback_after_hf_transient_error(), test_local_evaluation_falls_back_to_api_chain_on_failure()

### Community 19 - "Community 19"
Cohesion: 0.17
Nodes (16): parse_grounded_facts_json(), repair_json(), _strip_json_fences(), A story that invents many new named entities (> threshold) should still be…, Structured facts produce one numbered line per fact., Empty facts list must return empty string so callers fall back to raw JSON., A fact with an empty quote string should not include a quote section., Section headers produced by _flow_section_headers() contain capitalized words… (+8 more)

### Community 20 - "Community 20"
Cohesion: 0.15
Nodes (10): The eval harness must query storyforge.rag.retrieval.retrieve_docs() -- the…, The chunk dicts retrieve_docs()/_docs_to_chunks() produce must be a shape…, tests/fixtures/retrieval_eval_cases.example.json should stay a meaningful…, test_evaluate_retrieval_uses_query_function(), query_fn(), test_example_fixture_has_realistic_diverse_cases(), test_make_retrieve_docs_query_fn_routes_through_real_pipeline(), fake_docs_to_chunks() (+2 more)

### Community 21 - "Community 21"
Cohesion: 0.16
Nodes (15): dataclasses, math, is_thinking_mode(), length_presets(), length_token_cap(), _parse_target(), Any, Resolve a story length target into prompt guidance, tokens, and accept gates.… (+7 more)

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (8): fastapi_testclient, langchain_core_documents, app(), client(), fixture, API contract tests for StoryForge FastAPI endpoints. Uses FastAPI TestClient…, Ensure valid steps are accepted (orchestrator itself is mocked)., test_run_step_accepts_valid_step()

### Community 23 - "Community 23"
Cohesion: 0.20
Nodes (10): importlib, _client_for(), _FakeOrchestrator, _GenMode, _load_create_eval_routes(), Enum, str, test_status_route_is_available() (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.15
Nodes (12): ModuleType, _build_stubs(), fake_docs(), fake_hf_response(), fixture, Register lightweight stubs for all GPU/ML packages. Safe to use in any test…, Build a minimal InferenceClient chat_completion response stub., Return n minimal LangChain Document stubs for use in API / retrieval tests. (+4 more)

### Community 25 - "Community 25"
Cohesion: 0.18
Nodes (6): _FakeDoc, Regression tests for storyforge.rag.retrieval.retrieve_docs()'s pipeline order.…, A title diversity would have dropped (the pool is bigger than the target chunk…, test_retrieve_docs_reranks_before_diversity_selection(), fake_rerank(), test_retrieve_docs_skips_diversity_narrowing_before_rerank_sees_it()

### Community 26 - "Community 26"
Cohesion: 0.23
Nodes (12): re, build_debug_attribution_stub(), GroundedFact, _lenient_fact(), _norm_chunk_id(), Any, Grounded facts parsing and light hallucination checks. Step 2 returns JSON…, Map a model-cited id onto a real retrieved chunk id (or None). Accepts exact /… (+4 more)

### Community 27 - "Community 27"
Cohesion: 0.24
Nodes (11): download_archive_book(), extract_books_info(), _find_file_for_formats(), Any, receive_book(), search_in_archive(), download_book_archive(), _identifier_already_downloaded() (+3 more)

### Community 28 - "Community 28"
Cohesion: 0.38
Nodes (6): Gen_mode, parse_gen_mode(), parse_story_type(), Enum, str, StoryType

### Community 29 - "Community 29"
Cohesion: 0.20
Nodes (11): generate_story_agentic_result(), generate_story_result(), Any, Agentic generation until accept or max iterations., clean_story_output(), _close_quote_for(), Path, Return the matching close-double-quote for the opening in para. (+3 more)

### Community 30 - "Community 30"
Cohesion: 0.36
Nodes (10): storyforge_data_story_records, Path, New workflow test: - `.txt` stories -> `data/story_json/<title>.json` (manual…, _read_json(), _set_tmp_config(), test_create_story_records_writes_editable_json(), test_records_to_ingest_manifest_omits_series_fields_when_not_series(), test_records_to_ingest_manifest_writes_jsonl() (+2 more)

### Community 31 - "Community 31"
Cohesion: 0.22
Nodes (7): argparse, json, main(), main(), push_section_metadata.py ------------------------- After…, refresh_chunk_embeddings.py ---------------------------- Re-embed and upsert…, storyforge_data_records_to_manifest

### Community 32 - "Community 32"
Cohesion: 0.31
Nodes (10): _fake_llm(), _local_cfg(), test_hf_failure_falls_back_to_hardened_local_path(), test_load_facts_llm_passes_schema_to_ollama(), test_local_failure_logs_error_and_returns_empty(), test_local_json_format_defaults_to_schema_and_falls_back_when_rejected(), _loader(), test_local_provider_never_calls_hf() (+2 more)

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (8): requests, check_server(), main(), print_divider(), test_generation.py ------------------ Manual smoke-tests for the story…, run_test(), textwrap, time

### Community 34 - "Community 34"
Cohesion: 0.22
Nodes (6): _build_profile(), LengthProfile, Every length-related number for one generation request., Narration time at ``words_per_minute``, for video-script targets., The ``{length_guidance}`` block injected into the story prompts., Compact summary for API ``gen_params`` responses.

### Community 35 - "Community 35"
Cohesion: 0.32
Nodes (7): check_nvidia_smi(), check_pytorch_cuda(), Advanced CUDA check for RTX 50-series and Blackwell architecture. CLI wrapper…, Check CUDA via nvidia-smi (driver + GPU info)., Detailed check for PyTorch and GPU compatibility., run_cuda_compatibility_check(), subprocess

### Community 36 - "Community 36"
Cohesion: 0.38
Nodes (7): attribution_violations(), extract_named_entities_heuristic(), Rough set of capitalized words treated as proper nouns (for attribution check)., _apply_attribution_gate(), Log (or optionally truncate) names not present in grounded facts., Mirror of the threshold logic in langchain_rag.generate_story_3step_langchain.…, _should_truncate()

### Community 37 - "Community 37"
Cohesion: 0.29
Nodes (6): _fake_docs(), _fake_parsed_facts(), _mock_rag_for_stream(), Document, Patch retrieve_docs, extract_grounded_facts, and the streaming LLM. The LLM…, Minimal ParsedFacts stub.

### Community 38 - "Community 38"
Cohesion: 0.40
Nodes (3): main(), Probe Step-2 Hugging Face grounded-facts extraction (extraction.py). Calls the…, _sample_chunks()

### Community 39 - "Community 39"
Cohesion: 0.53
Nodes (5): main(), Any, measure_generation_length.py ----------------------------- Phase 2 measurement…, _run_one(), summarize()

### Community 40 - "Community 40"
Cohesion: 0.40
Nodes (4): BaseException, is_retryable_api_error(), Shared classification of transient remote-API errors. Used by grounded-facts…, True when ``exc`` looks like a transient remote failure worth retrying.

### Community 41 - "Community 41"
Cohesion: 0.40
Nodes (4): root(), main(), _run(), One-command Step 1: - Create missing `data/story_json/*.json` from…

### Community 42 - "Community 42"
Cohesion: 0.40
Nodes (5): _sections_below_min_sentences(), Map SECTION number -> its body text, using SECTION_HEADER_RE as delimiters., Rough sentence count via terminal-punctuation runs., sentence_count(), split_section_bodies()

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (3): is_marker(), merge_file(), Merge story file: split by --- lines, merge each block to one line (Statement…

## Knowledge Gaps
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 38`, `Community 39`, `Community 7`, `Community 12`, `Community 13`, `Community 16`, `Community 27`, `Community 28`, `Community 29`, `Community 31`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `resolve_length_profile()` connect `Community 8` to `Community 1`, `Community 2`, `Community 34`, `Community 4`, `Community 11`, `Community 16`, `Community 21`, `Community 29`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `run_agentic_story_loop()` connect `Community 1` to `Community 2`, `Community 3`, `Community 8`, `Community 9`, `Community 11`, `Community 16`, `Community 21`, `Community 28`, `Community 29`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `load_config()` (e.g. with `main()` and `main()`) actually correct?**
  _`load_config()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Orchestrator` (e.g. with `main()` and `_run_one()`) actually correct?**
  _`Orchestrator` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05201292976785189 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.07595628415300547 - nodes in this community are weakly interconnected._