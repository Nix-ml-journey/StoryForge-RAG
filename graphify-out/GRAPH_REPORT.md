# Graph Report - RAG  (2026-09-28)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 906 nodes · 1963 edges · 48 communities (40 shown, 8 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 96 edges (avg confidence: 0.88)
- Token cost: 3,599 input · 550 output

## Graph Freshness
- Built from commit: `367b2b3f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- SSE Stream Generation
- Agentic Story Loop
- Draft Recovery Tests
- Length Profile Resolution
- Vector Store Validation
- API Contract Testing
- RAG Orchestration
- Data Ingestion Pipeline
- Local Fact Parsing
- Remote API Error Handling
- Book Fetching Logic
- Chroma Metadata Validation
- Configuration Management
- Attribution Gate Logic
- Evaluation Route Definitions
- Generation Performance Tracking
- FastAPI Orchestration Routes
- Vector Store Operations
- Ingestion Metadata Tests
- Codebase Cleanup Tools
- Evaluation Model Logic
- Book Processing API
- Core Orchestrator Logic
- Retrieval Evaluation Tests
- Retrieval Logic Tests
- Configuration Loading
- Document Text Extraction
- Mocking and Stubs
- Fact Parsing Logic
- Story Record Processing
- Agentic Result Cleanup
- Story Record Workflows
- Vector Store Management
- FastAPI Application Entry
- System Compatibility Scripts
- Manual Smoke Tests
- CUDA Compatibility Checks
- Hugging Face Debugging
- Knowledge Graph Updates
- Vector Store Reset
- Gemini Model Listing
- Configuration Testing
- Book Search Module
- Configuration Module
- Vector Store Module

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 54 edges
2. `resolve_length_profile()` - 38 edges
3. `Orchestrator` - 27 edges
4. `salvage_grounded_facts_json()` - 25 edges
5. `run_agentic_story_loop()` - 24 edges
6. `decide_action()` - 20 edges
7. `Gen_mode` - 19 edges
8. `_stream_story_sse()` - 18 edges
9. `ingest_stories_dir()` - 18 edges
10. `extract_grounded_facts()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `load_config()`  [INFERRED]
  scripts/peek_vector_store.py → src/storyforge/config/config.py
- `test_is_thinking_mode_accepts_enum_values_and_aliases()` --uses--> `Gen_mode`  [INFERRED]
  tests/test_length_profile.py → src/storyforge/rag/generative_ai.py
- `main()` --uses--> `Orchestrator`  [INFERRED]
  scripts/measure_generation_length.py → src/storyforge/orchestrator/orchestrator.py
- `_run_one()` --uses--> `Orchestrator`  [INFERRED]
  scripts/measure_generation_length.py → src/storyforge/orchestrator/orchestrator.py
- `_fake_parsed_facts()` --uses--> `ParsedFacts`  [INFERRED]
  tests/test_api_contracts.py → src/storyforge/rag/attribution.py

## Import Cycles
- None detected.

## Communities (48 total, 8 thin omitted)

### Community 0 - "SSE Stream Generation"
Cohesion: 0.05
Nodes (69): Protocol, Core SSE generator: Steps 1–2 sync in thread, Step 3 streams via Ollama., _stream_story_sse(), format_facts_for_prompt(), Numbered bullet list for the generation prompt (empty if no facts). Example: 1.…, _get_generation_prompts(), Load prompt templates from prompts.yaml (generation section)., _apply_attribution_gate() (+61 more)

### Community 1 - "Agentic Story Loop"
Cohesion: 0.07
Nodes (60): AgenticLoopResult, average_score(), build_feedback(), completeness_report(), CompletenessReport, criterion_score(), decide_action(), Decision (+52 more)

### Community 2 - "Draft Recovery Tests"
Cohesion: 0.06
Nodes (45): _auto_stub(), _cfg(), fixture, Tests for empty-draft recovery in generation.py and langchain_rag.py. Covers: -…, If both thinking and fast-mode attempts return empty, RuntimeError is raised., Fast mode returning empty also raises RuntimeError (no retry loop for fast)., If the length guard refine call raises RuntimeError, the original draft is…, When thinking mode returns empty, generate_from_facts retries once with fast… (+37 more)

### Community 3 - "Length Profile Resolution"
Cohesion: 0.08
Nodes (40): dataclasses, math, _build_profile(), is_thinking_mode(), length_presets(), length_token_cap(), _parse_target(), Any (+32 more)

### Community 4 - "Vector Store Validation"
Cohesion: 0.13
Nodes (38): _configured_collection(), get_vector_store_count(), get_vector_store_ids(), get_vector_store_item(), list_vector_store(), _normalize_docs(), _normalize_ids(), _normalize_metas() (+30 more)

### Community 5 - "API Contract Testing"
Cohesion: 0.07
Nodes (23): fastapi_testclient, importlib, app(), client(), _fake_docs(), _fake_parsed_facts(), _mock_rag_for_stream(), Document (+15 more)

### Community 6 - "RAG Orchestration"
Cohesion: 0.10
Nodes (33): Chroma, langchain_chroma, langchain_core_documents, langchain_huggingface, build_debug_attribution_stub(), Debug payload: each sentence tagged with all source chunk ids from facts., _sections_below_min_sentences(), generate_story_3step_langchain() (+25 more)

### Community 7 - "Data Ingestion Pipeline"
Cohesion: 0.09
Nodes (31): main(), _flush(), Step 1b (actual ingest): - Reads `data/ingest/ingest_manifest.jsonl` - Upserts…, main(), _delete_dir(), main(), Path, reset_and_ingest.py ------------------- Run this once after upgrading to BGE… (+23 more)

### Community 8 - "Local Fact Parsing"
Cohesion: 0.12
Nodes (28): Best-effort parse of messy local-model facts output. Returns ``(ParsedFacts,…, salvage_grounded_facts_json(), extraction(), _fake_llm(), _local_cfg(), fixture, Tests for the LOCAL grounded-facts path (no HF): salvage parser + extraction…, test_hf_failure_falls_back_to_hardened_local_path() (+20 more)

### Community 9 - "Remote API Error Handling"
Cohesion: 0.13
Nodes (26): BaseException, huggingface_hub, is_retryable_api_error(), Shared classification of transient remote-API errors. Used by grounded-facts…, True when ``exc`` looks like a transient remote failure worth retrying., ParsedFacts, extract_grounded_facts(), _extract_grounded_facts_local() (+18 more)

### Community 10 - "Book Fetching Logic"
Cohesion: 0.11
Nodes (25): download_archive_book(), extract_books_info(), _find_file_for_formats(), Any, receive_book(), search_in_archive(), src_storyforge_data_step1_prepare_and_enrich, src_storyforge_evaluation_evaluation (+17 more)

### Community 11 - "Chroma Metadata Validation"
Cohesion: 0.17
Nodes (22): main(), Validate what an ingest actually wrote to Chroma (offline, no HF, no GPU).…, fetch_all(), format_report(), _pct(), problems(), Any, Chroma metadata health report (pure functions, no Chroma import). Used by… (+14 more)

### Community 12 - "Configuration Management"
Cohesion: 0.13
Nodes (22): _default_n_results(), Default n_results from setup.yaml (read at request time, not import time). The…, load_config(), Path, resolve_config_path(), _generation_prompts(), Tests for config loading and prompt YAML contracts., Word/sentence minimums are derived from the length target. Leaving them in… (+14 more)

### Community 13 - "Attribution Gate Logic"
Cohesion: 0.14
Nodes (21): re, attribution_violations(), extract_named_entities_heuristic(), GroundedFact, parse_grounded_facts_json(), Rough set of capitalized words treated as proper nouns (for attribution check)., A story that invents many new named entities (> threshold) should still be…, Structured facts produce one numbered line per fact. (+13 more)

### Community 14 - "Evaluation Route Definitions"
Cohesion: 0.23
Nodes (20): create_eval_status(), BaseModel, post, StatusResponse, story_evaluate(), story_evaluate_file(), story_generate(), StoryEvaluateFileRequest (+12 more)

### Community 15 - "Generation Performance Tracking"
Cohesion: 0.20
Nodes (12): datetime, main(), Any, measure_generation_length.py ----------------------------- Phase 2 measurement…, _run_one(), summarize(), Gen_mode, parse_story_type() (+4 more)

### Community 16 - "FastAPI Orchestration Routes"
Cohesion: 0.19
Nodes (18): fastapi_responses, generate_stream(), GenerateStreamRequest, orchestration_status(), BaseModel, post, Run one or more orchestration pipeline steps and return their combined result., Run only one pipeline step while keeping the endpoint surface small. (+10 more)

### Community 17 - "Vector Store Operations"
Cohesion: 0.14
Nodes (16): logging, query_vector_result(), Query the vector store; return hits with full stored metadata., vector_delete_result(), delete_data(), query_data(), Query with the same embed model as ingest (never Chroma's default 384-dim)., embed_query() (+8 more)

### Community 18 - "Ingestion Metadata Tests"
Cohesion: 0.23
Nodes (15): _FakeCollection, ingest(), run(), fixture, Path, ingest_stories_dir() must honour reviewed story_json records (P0.1).…, test_crlf_only_difference_is_not_stale(), test_matching_story_json_supplies_chunks_sections_and_metadata() (+7 more)

### Community 19 - "Codebase Cleanup Tools"
Cohesion: 0.18
Nodes (14): ast, collections, FunctionDef, body_fingerprint(), main(), parse(), py_files(), Path (+6 more)

### Community 20 - "Evaluation Model Logic"
Cohesion: 0.12
Nodes (6): storyforge_evaluation, _FakeGemini, test_evaluate_model_falls_back_to_gemini_when_hf_unavailable(), test_invoke_with_retry_routes_local_provider_to_local_invoker(), test_invoke_with_retry_uses_gemini_fallback_after_hf_transient_error(), test_local_evaluation_falls_back_to_api_chain_on_failure()

### Community 21 - "Book Processing API"
Cohesion: 0.23
Nodes (16): asyncio, pydantic, book_status(), download_book(), DownloadBookRequest, DownloadBookResponse, extract_text(), ExtractTextRequest (+8 more)

### Community 23 - "Retrieval Evaluation Tests"
Cohesion: 0.15
Nodes (10): The eval harness must query storyforge.rag.retrieval.retrieve_docs() -- the…, The chunk dicts retrieve_docs()/_docs_to_chunks() produce must be a shape…, tests/fixtures/retrieval_eval_cases.example.json should stay a meaningful…, test_evaluate_retrieval_uses_query_function(), query_fn(), test_example_fixture_has_realistic_diverse_cases(), test_make_retrieve_docs_query_fn_routes_through_real_pipeline(), fake_docs_to_chunks() (+2 more)

### Community 24 - "Retrieval Logic Tests"
Cohesion: 0.18
Nodes (6): _FakeDoc, Regression tests for storyforge.rag.retrieval.retrieve_docs()'s pipeline order.…, A title diversity would have dropped (the pool is bigger than the target chunk…, test_retrieve_docs_reranks_before_diversity_selection(), fake_rerank(), test_retrieve_docs_skips_diversity_narrowing_before_rerank_sees_it()

### Community 25 - "Configuration Loading"
Cohesion: 0.27
Nodes (11): copy, functools, os, _load_config_cached(), load_prompts(), _load_prompts_cached(), _normalize_base_path(), Any (+3 more)

### Community 26 - "Document Text Extraction"
Cohesion: 0.24
Nodes (11): epub_to_text, fitz, extract_text_from_books(), extract_text_from_epub(), extract_text_from_pdf(), format_extracted_text(), _iter_files(), Any (+3 more)

### Community 27 - "Mocking and Stubs"
Cohesion: 0.17
Nodes (11): ModuleType, pytest, _build_stubs(), fake_docs(), fake_hf_response(), fixture, Register lightweight stubs for all GPU/ML packages. Safe to use in any test…, Build a minimal InferenceClient chat_completion response stub. (+3 more)

### Community 28 - "Fact Parsing Logic"
Cohesion: 0.23
Nodes (11): _lenient_fact(), _norm_chunk_id(), Any, Grounded facts parsing and light hallucination checks. Step 2 returns JSON…, Map a model-cited id onto a real retrieved chunk id (or None). Accepts exact /…, Decode every complete {...} object after the "facts" key (tolerates truncation)., repair_json(), _resolve_chunk_id() (+3 more)

### Community 29 - "Story Record Processing"
Cohesion: 0.20
Nodes (4): is_marker(), merge_file(), Merge story file: split by --- lines, merge each block to one line (Statement…, sys

### Community 30 - "Agentic Result Cleanup"
Cohesion: 0.20
Nodes (11): generate_story_agentic_result(), generate_story_result(), Any, Agentic generation until accept or max iterations., clean_story_output(), _close_quote_for(), Path, Return the matching close-double-quote for the opening in para. (+3 more)

### Community 31 - "Story Record Workflows"
Cohesion: 0.36
Nodes (10): storyforge_data_story_records, Path, New workflow test: - `.txt` stories -> `data/story_json/<title>.json` (manual…, _read_json(), _set_tmp_config(), test_create_story_records_writes_editable_json(), test_records_to_ingest_manifest_omits_series_fields_when_not_series(), test_records_to_ingest_manifest_writes_jsonl() (+2 more)

### Community 32 - "Vector Store Management"
Cohesion: 0.22
Nodes (7): argparse, json, main(), main(), push_section_metadata.py ------------------------- After…, refresh_chunk_embeddings.py ---------------------------- Re-embed and upsert…, storyforge_data_records_to_manifest

### Community 33 - "FastAPI Application Entry"
Cohesion: 0.25
Nodes (7): FastAPI, create_app(), root(), main(), _run(), One-command Step 1: - Create missing `data/story_json/*.json` from…, uvicorn

### Community 34 - "System Compatibility Scripts"
Cohesion: 0.22
Nodes (6): pathlib, CLI wrapper for `storyforge.scripts.check_cuda_compatibility`. Run: py…, CLI wrapper for `storyforge.evaluation.retrieval_eval`. Run from repo root: py…, storyforge_evaluation_retrieval_eval, storyforge_scripts_check_cuda_compatibility, storyforge_scripts_list_gemini_models

### Community 35 - "Manual Smoke Tests"
Cohesion: 0.33
Nodes (8): requests, check_server(), main(), print_divider(), test_generation.py ------------------ Manual smoke-tests for the story…, run_test(), textwrap, time

### Community 36 - "CUDA Compatibility Checks"
Cohesion: 0.32
Nodes (7): check_nvidia_smi(), check_pytorch_cuda(), Advanced CUDA check for RTX 50-series and Blackwell architecture. CLI wrapper…, Check CUDA via nvidia-smi (driver + GPU info)., Detailed check for PyTorch and GPU compatibility., run_cuda_compatibility_check(), subprocess

### Community 37 - "Hugging Face Debugging"
Cohesion: 0.33
Nodes (4): main(), Probe Step-2 Hugging Face grounded-facts extraction (extraction.py). Calls the…, _sample_chunks(), unittest_mock

### Community 38 - "Knowledge Graph Updates"
Cohesion: 0.48
Nodes (6): _ensure_model(), main(), _ollama_models(), Refresh the graphify knowledge graph and name its communities with local Gemma…, _run(), urllib_request

### Community 39 - "Vector Store Reset"
Cohesion: 0.50
Nodes (5): reset_vector_store_result(), get_chroma_storage_path(), Path, Aggressive reset: deletes the entire Chroma directory on disk and recreates it.…, reset_vector_store_dir()

### Community 40 - "Gemini Model Listing"
Cohesion: 0.50
Nodes (3): main(), list_gemini_models(), List Gemini models for the configured API key. CLI wrapper exists at…

## Knowledge Gaps
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `Configuration Management` to `SSE Stream Generation`, `Agentic Story Loop`, `Vector Store Validation`, `RAG Orchestration`, `Data Ingestion Pipeline`, `Book Fetching Logic`, `Chroma Metadata Validation`, `Evaluation Route Definitions`, `Generation Performance Tracking`, `FastAPI Orchestration Routes`, `Vector Store Operations`, `Core Orchestrator Logic`, `Configuration Loading`, `Document Text Extraction`, `Agentic Result Cleanup`, `Vector Store Management`, `FastAPI Application Entry`, `Hugging Face Debugging`, `Vector Store Reset`, `Gemini Model Listing`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Why does `resolve_length_profile()` connect `Length Profile Resolution` to `SSE Stream Generation`, `Agentic Story Loop`, `RAG Orchestration`, `Book Fetching Logic`, `FastAPI Orchestration Routes`, `Agentic Result Cleanup`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `run_agentic_story_loop()` connect `Agentic Story Loop` to `SSE Stream Generation`, `Length Profile Resolution`, `RAG Orchestration`, `Remote API Error Handling`, `Book Fetching Logic`, `Configuration Management`, `Generation Performance Tracking`, `Agentic Result Cleanup`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `load_config()` (e.g. with `main()` and `main()`) actually correct?**
  _`load_config()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Orchestrator` (e.g. with `main()` and `_run_one()`) actually correct?**
  _`Orchestrator` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `SSE Stream Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.05228105228105228 - nodes in this community are weakly interconnected._
- **Should `Agentic Story Loop` be split into smaller, more focused modules?**
  _Cohesion score 0.06971153846153846 - nodes in this community are weakly interconnected._