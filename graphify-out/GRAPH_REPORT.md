# Graph Report - RAG  (2026-10-01)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 906 nodes · 1956 edges · 44 communities (38 shown, 6 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 96 edges (avg confidence: 0.88)
- Token cost: 3,544 input · 507 output

## Graph Freshness
- Built from commit: `0abebebb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Streaming Generation Logic
- Agentic Story Loop
- Hugging Face Extraction
- Length Profile Resolution
- Vector Store Management
- API Contract Testing
- RAG Story Generation
- Database Reset Tools
- Local Fact Parsing
- Fact Extraction Logic
- Data Preparation Pipeline
- Codebase Cleanup Tools
- Configuration Management
- Attribution and Parsing
- Orchestration and Evaluation
- Draft Recovery Tests
- FastAPI Web Server
- Book Fetching Logic
- Metadata Ingestion Tests
- Evaluation Route Tests
- Evaluation System Tests
- LLM Provider Testing
- Streaming Mock Stubs
- Retrieval Evaluation
- Retrieval Regression Tests
- Configuration and Secrets
- Document Text Extraction
- Mocking and Stubs
- Fact Parsing Logic
- Story Record Processing
- Mocked Orchestrator Logic
- Story Workflow Tests
- Metadata and Embeddings
- Data Enrichment Script
- Ingestion and CLI
- Manual Smoke Tests
- CUDA Compatibility Checks
- Hugging Face Debugging
- Knowledge Graph Updates
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
- `_fake_parsed_facts()` --uses--> `ParsedFacts`  [INFERRED]
  tests/test_api_contracts.py → src/storyforge/rag/attribution.py
- `test_both_attempts_empty_raises_runtime_error()` --uses--> `ParsedFacts`  [INFERRED]
  tests/test_empty_draft_recovery.py → src/storyforge/rag/attribution.py
- `test_fast_mode_empty_raises_runtime_error()` --uses--> `ParsedFacts`  [INFERRED]
  tests/test_empty_draft_recovery.py → src/storyforge/rag/attribution.py

## Import Cycles
- None detected.

## Communities (44 total, 6 thin omitted)

### Community 0 - "Streaming Generation Logic"
Cohesion: 0.05
Nodes (69): Protocol, Core SSE generator: Steps 1–2 sync in thread, Step 3 streams via Ollama., _stream_story_sse(), format_facts_for_prompt(), Numbered bullet list for the generation prompt (empty if no facts). Example: 1.…, _get_generation_prompts(), Load prompt templates from prompts.yaml (generation section)., _apply_attribution_gate() (+61 more)

### Community 1 - "Agentic Story Loop"
Cohesion: 0.07
Nodes (60): AgenticLoopResult, average_score(), build_feedback(), completeness_report(), CompletenessReport, criterion_score(), decide_action(), Decision (+52 more)

### Community 2 - "Hugging Face Extraction"
Cohesion: 0.10
Nodes (34): _hf_chat_extract_json(), Step 2 via Hugging Face Inference API (chat_completion router endpoint). Uses…, _auto_stub(), _cfg(), _fake_hf_response(), fixture, Tests for Step 2 grounded-facts extraction (extraction.py). Covers: - JSON mode…, By default, Qwen3's hidden chain-of-thought is discouraged two ways:… (+26 more)

### Community 3 - "Length Profile Resolution"
Cohesion: 0.08
Nodes (40): dataclasses, math, _build_profile(), is_thinking_mode(), length_presets(), length_token_cap(), _parse_target(), Any (+32 more)

### Community 4 - "Vector Store Management"
Cohesion: 0.05
Nodes (76): logging, _flush(), main(), _configured_collection(), get_vector_store_count(), get_vector_store_ids(), get_vector_store_item(), list_vector_store() (+68 more)

### Community 5 - "API Contract Testing"
Cohesion: 0.14
Nodes (7): fastapi_testclient, app(), client(), fixture, API contract tests for StoryForge FastAPI endpoints. Uses FastAPI TestClient…, Ensure valid steps are accepted (orchestrator itself is mocked)., test_run_step_accepts_valid_step()

### Community 6 - "RAG Story Generation"
Cohesion: 0.10
Nodes (33): Chroma, langchain_chroma, langchain_core_documents, langchain_huggingface, build_debug_attribution_stub(), Debug payload: each sentence tagged with all source chunk ids from facts., _sections_below_min_sentences(), generate_story_3step_langchain() (+25 more)

### Community 7 - "Database Reset Tools"
Cohesion: 0.31
Nodes (8): _delete_dir(), main(), Path, reset_and_ingest.py ------------------- Run this once after upgrading to BGE…, Try to delete a directory. Returns True on success, False if locked., Delete and recreate the collection using the ChromaDB client directly., _reset_via_chromadb(), shutil

### Community 8 - "Local Fact Parsing"
Cohesion: 0.14
Nodes (20): parametrize, Best-effort parse of messy local-model facts output. Returns ``(ParsedFacts,…, salvage_grounded_facts_json(), extraction(), fixture, Tests for the LOCAL grounded-facts path (no HF): salvage parser + extraction…, test_local_json_format_config(), test_salvage_accepts_singular_source_key() (+12 more)

### Community 9 - "Fact Extraction Logic"
Cohesion: 0.14
Nodes (24): BaseException, huggingface_hub, is_retryable_api_error(), Shared classification of transient remote-API errors. Used by grounded-facts…, True when ``exc`` looks like a transient remote failure worth retrying., ParsedFacts, extract_grounded_facts(), _extract_grounded_facts_local() (+16 more)

### Community 10 - "Data Preparation Pipeline"
Cohesion: 0.13
Nodes (17): src_storyforge_data_step1_prepare_and_enrich, src_storyforge_evaluation_evaluation, _eval_to_dict(), evaluate_story_file_result(), evaluate_story_text_result(), evaluate_summary_result(), extract_text_result(), query_vector_result() (+9 more)

### Community 11 - "Codebase Cleanup Tools"
Cohesion: 0.09
Nodes (36): ast, collections, FunctionDef, body_fingerprint(), main(), parse(), py_files(), Path (+28 more)

### Community 12 - "Configuration Management"
Cohesion: 0.12
Nodes (23): _default_n_results(), Default n_results from setup.yaml (read at request time, not import time). The…, load_config(), Path, resolve_config_path(), generate_summary_result(), _generation_prompts(), Tests for config loading and prompt YAML contracts. (+15 more)

### Community 13 - "Attribution and Parsing"
Cohesion: 0.14
Nodes (21): re, attribution_violations(), extract_named_entities_heuristic(), GroundedFact, parse_grounded_facts_json(), Rough set of capitalized words treated as proper nouns (for attribution check)., Structured facts produce one numbered line per fact., Empty facts list must return empty string so callers fall back to raw JSON. (+13 more)

### Community 14 - "Orchestration and Evaluation"
Cohesion: 0.06
Nodes (45): datetime, main(), Any, measure_generation_length.py ----------------------------- Phase 2 measurement…, _run_one(), summarize(), create_eval_status(), BaseModel (+37 more)

### Community 15 - "Draft Recovery Tests"
Cohesion: 0.12
Nodes (13): _auto_stub(), _cfg(), fixture, Tests for empty-draft recovery in generation.py and langchain_rag.py. Covers: -…, If both thinking and fast-mode attempts return empty, RuntimeError is raised., Fast mode returning empty also raises RuntimeError (no retry loop for fast)., If the length guard refine call raises RuntimeError, the original draft is…, Ensure langchain / HF / torch stubs are active for every test here. (+5 more)

### Community 16 - "FastAPI Web Server"
Cohesion: 0.10
Nodes (36): asyncio, FastAPI, fastapi_responses, create_app(), pydantic, book_status(), download_book(), DownloadBookRequest (+28 more)

### Community 17 - "Book Fetching Logic"
Cohesion: 0.24
Nodes (11): download_archive_book(), extract_books_info(), _find_file_for_formats(), Any, receive_book(), search_in_archive(), download_book_archive(), _identifier_already_downloaded() (+3 more)

### Community 18 - "Metadata Ingestion Tests"
Cohesion: 0.23
Nodes (15): _FakeCollection, ingest(), run(), fixture, Path, ingest_stories_dir() must honour reviewed story_json records (P0.1).…, test_crlf_only_difference_is_not_stale(), test_matching_story_json_supplies_chunks_sections_and_metadata() (+7 more)

### Community 19 - "Evaluation Route Tests"
Cohesion: 0.36
Nodes (9): importlib, _client_for(), _GenMode, _load_create_eval_routes(), Enum, str, test_status_route_is_available(), test_story_evaluate_route_returns_scores() (+1 more)

### Community 20 - "Evaluation System Tests"
Cohesion: 0.12
Nodes (6): storyforge_evaluation, _FakeGemini, test_evaluate_model_falls_back_to_gemini_when_hf_unavailable(), test_invoke_with_retry_routes_local_provider_to_local_invoker(), test_invoke_with_retry_uses_gemini_fallback_after_hf_transient_error(), test_local_evaluation_falls_back_to_api_chain_on_failure()

### Community 21 - "LLM Provider Testing"
Cohesion: 0.31
Nodes (10): _fake_llm(), _local_cfg(), test_hf_failure_falls_back_to_hardened_local_path(), test_load_facts_llm_passes_schema_to_ollama(), test_local_failure_logs_error_and_returns_empty(), test_local_json_format_defaults_to_schema_and_falls_back_when_rejected(), _loader(), test_local_provider_never_calls_hf() (+2 more)

### Community 22 - "Streaming Mock Stubs"
Cohesion: 0.29
Nodes (6): _fake_docs(), _fake_parsed_facts(), _mock_rag_for_stream(), Document, Patch retrieve_docs, extract_grounded_facts, and the streaming LLM. The LLM…, Minimal ParsedFacts stub.

### Community 23 - "Retrieval Evaluation"
Cohesion: 0.15
Nodes (10): The eval harness must query storyforge.rag.retrieval.retrieve_docs() -- the…, The chunk dicts retrieve_docs()/_docs_to_chunks() produce must be a shape…, tests/fixtures/retrieval_eval_cases.example.json should stay a meaningful…, test_evaluate_retrieval_uses_query_function(), query_fn(), test_example_fixture_has_realistic_diverse_cases(), test_make_retrieve_docs_query_fn_routes_through_real_pipeline(), fake_docs_to_chunks() (+2 more)

### Community 24 - "Retrieval Regression Tests"
Cohesion: 0.18
Nodes (6): _FakeDoc, Regression tests for storyforge.rag.retrieval.retrieve_docs()'s pipeline order.…, A title diversity would have dropped (the pool is bigger than the target chunk…, test_retrieve_docs_reranks_before_diversity_selection(), fake_rerank(), test_retrieve_docs_skips_diversity_narrowing_before_rerank_sees_it()

### Community 25 - "Configuration and Secrets"
Cohesion: 0.27
Nodes (11): copy, functools, os, _load_config_cached(), load_prompts(), _load_prompts_cached(), _normalize_base_path(), Any (+3 more)

### Community 26 - "Document Text Extraction"
Cohesion: 0.24
Nodes (11): epub_to_text, fitz, extract_text_from_books(), extract_text_from_epub(), extract_text_from_pdf(), format_extracted_text(), _iter_files(), Any (+3 more)

### Community 27 - "Mocking and Stubs"
Cohesion: 0.15
Nodes (12): ModuleType, pytest, _build_stubs(), fake_docs(), fake_hf_response(), fixture, Register lightweight stubs for all GPU/ML packages. Safe to use in any test…, Build a minimal InferenceClient chat_completion response stub. (+4 more)

### Community 28 - "Fact Parsing Logic"
Cohesion: 0.23
Nodes (11): _lenient_fact(), _norm_chunk_id(), Any, Grounded facts parsing and light hallucination checks. Step 2 returns JSON…, Map a model-cited id onto a real retrieved chunk id (or None). Accepts exact /…, Decode every complete {...} object after the "facts" key (tolerates truncation)., repair_json(), _resolve_chunk_id() (+3 more)

### Community 29 - "Story Record Processing"
Cohesion: 0.20
Nodes (4): is_marker(), merge_file(), Merge story file: split by --- lines, merge each block to one line (Statement…, sys

### Community 31 - "Story Workflow Tests"
Cohesion: 0.36
Nodes (10): storyforge_data_story_records, Path, New workflow test: - `.txt` stories -> `data/story_json/<title>.json` (manual…, _read_json(), _set_tmp_config(), test_create_story_records_writes_editable_json(), test_records_to_ingest_manifest_omits_series_fields_when_not_series(), test_records_to_ingest_manifest_writes_jsonl() (+2 more)

### Community 32 - "Metadata and Embeddings"
Cohesion: 0.22
Nodes (7): argparse, json, main(), main(), push_section_metadata.py ------------------------- After…, refresh_chunk_embeddings.py ---------------------------- Re-embed and upsert…, storyforge_data_records_to_manifest

### Community 33 - "Data Enrichment Script"
Cohesion: 0.40
Nodes (4): root(), main(), _run(), One-command Step 1: - Create missing `data/story_json/*.json` from…

### Community 34 - "Ingestion and CLI"
Cohesion: 0.12
Nodes (11): pathlib, CLI wrapper for `storyforge.scripts.check_cuda_compatibility`. Run: py…, main(), Step 1b (actual ingest): - Reads `data/ingest/ingest_manifest.jsonl` - Upserts…, main(), CLI wrapper for `storyforge.evaluation.retrieval_eval`. Run from repo root: py…, list_gemini_models(), List Gemini models for the configured API key. CLI wrapper exists at… (+3 more)

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

## Knowledge Gaps
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `Configuration Management` to `Metadata and Embeddings`, `Streaming Generation Logic`, `Ingestion and CLI`, `Agentic Story Loop`, `Vector Store Management`, `Hugging Face Debugging`, `RAG Story Generation`, `Database Reset Tools`, `Data Preparation Pipeline`, `Codebase Cleanup Tools`, `Orchestration and Evaluation`, `FastAPI Web Server`, `Book Fetching Logic`, `Configuration and Secrets`, `Document Text Extraction`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `resolve_length_profile()` connect `Length Profile Resolution` to `Streaming Generation Logic`, `Agentic Story Loop`, `RAG Story Generation`, `Data Preparation Pipeline`, `Orchestration and Evaluation`, `FastAPI Web Server`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `run_agentic_story_loop()` connect `Agentic Story Loop` to `Streaming Generation Logic`, `Length Profile Resolution`, `RAG Story Generation`, `Fact Extraction Logic`, `Data Preparation Pipeline`, `Configuration Management`, `Orchestration and Evaluation`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `load_config()` (e.g. with `main()` and `main()`) actually correct?**
  _`load_config()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Orchestrator` (e.g. with `main()` and `_run_one()`) actually correct?**
  _`Orchestrator` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `Streaming Generation Logic` be split into smaller, more focused modules?**
  _Cohesion score 0.05228105228105228 - nodes in this community are weakly interconnected._
- **Should `Agentic Story Loop` be split into smaller, more focused modules?**
  _Cohesion score 0.06971153846153846 - nodes in this community are weakly interconnected._