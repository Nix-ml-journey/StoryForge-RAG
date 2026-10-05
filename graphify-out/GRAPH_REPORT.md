# Graph Report - RAG  (2026-10-05)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 929 nodes · 1992 edges · 55 communities (35 shown, 20 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 100 edges (avg confidence: 0.88)
- Token cost: 3,952 input · 619 output

## Graph Freshness
- Built from commit: `366ad167`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Streaming API Routes
- Agentic Loop Logic
- Fact Extraction Tests
- Length Profile Logic
- Vector Store Utilities
- API Contract Tests
- Vector Retrieval Logic
- Data Reset Utilities
- Fact Attribution Parsing
- Fact Extraction Logic
- Data Preparation Tools
- Chroma Metadata Reporting
- Configuration Management
- Fact Formatting Logic
- Orchestration Management
- Generation Logic Tests
- Core Application Framework
- Book Fetching Utilities
- Ingestion Metadata Tests
- Mocking and Stubs
- Prompt Management
- Length Calculation Logic
- Document Retrieval Logic
- Retrieval Evaluation
- Retrieval Regression Tests
- Environment and Config
- Document Text Extraction
- Integration Pipeline Tests
- RAG Pipeline Logic
- Story Record Processing
- Attribution Violation Checks
- Story Workflow Tests
- CLI and Evaluation
- Length Profile Generation
- Data Ingestion Pipeline
- CUDA Compatibility Checks
- Fact Extraction Debugging
- Knowledge Graph Updates
- HTTP Request Handling
- Embedding Utilities
- API Error Handling
- Gemini Model Listing
- Provider Error Handling
- BM25 Retrieval Tests
- Mocking Infrastructure
- HF Token Validation

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 54 edges
2. `resolve_length_profile()` - 38 edges
3. `run_agentic_story_loop()` - 27 edges
4. `Orchestrator` - 25 edges
5. `salvage_grounded_facts_json()` - 25 edges
6. `decide_action()` - 24 edges
7. `retrieve_docs()` - 21 edges
8. `Gen_mode` - 19 edges
9. `_stream_story_sse()` - 19 edges
10. `extract_grounded_facts()` - 19 edges

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

## Communities (55 total, 20 thin omitted)

### Community 0 - "Streaming API Routes"
Cohesion: 0.06
Nodes (35): generate_stream(), GenerateStreamRequest, orchestration_status(), run_pipeline(), run_step(), RunPipelineRequest, RunPipelineResponse, RunStepRequest (+27 more)

### Community 1 - "Agentic Loop Logic"
Cohesion: 0.06
Nodes (54): AgenticLoopResult, average_score(), build_feedback(), completeness_report(), CompletenessReport, criterion_score(), decide_action(), Decision (+46 more)

### Community 2 - "Fact Extraction Tests"
Cohesion: 0.12
Nodes (17): _cfg(), _fake_hf_response(), test_disable_thinking_can_be_turned_off_by_config(), _fake_chat_completion(), test_disables_thinking_by_default(), _fake_chat_completion(), test_drops_both_optional_kwargs_on_type_error(), _fake_chat_completion() (+9 more)

### Community 3 - "Length Profile Logic"
Cohesion: 0.14
Nodes (17): resolve_length_profile(), test_accept_gate_sits_below_the_target_but_above_half(), test_as_dict_reports_the_target_for_api_responses(), test_config_presets_override_builtins(), test_duration_target_converts_narration_minutes_to_words(), test_empty_config_still_resolves_a_usable_target(), test_explicit_length_wins_over_mode_default(), test_guidance_text_states_the_resolved_numbers() (+9 more)

### Community 4 - "Vector Store Utilities"
Cohesion: 0.08
Nodes (36): _configured_collection(), get_vector_store_count(), get_vector_store_ids(), get_vector_store_item(), list_vector_store(), _normalize_docs(), _normalize_ids(), _normalize_metas() (+28 more)

### Community 5 - "API Contract Tests"
Cohesion: 0.10
Nodes (6): app(), client(), _fake_docs(), _fake_parsed_facts(), _mock_rag_for_stream(), test_run_step_accepts_valid_step()

### Community 6 - "Vector Retrieval Logic"
Cohesion: 0.15
Nodes (9): _build_vectorstore(), embed_query(), _get_paths_and_names(), _get_reranker(), _merge_filters(), _rerank_docs(), _resolve_device(), story_type_filter() (+1 more)

### Community 7 - "Data Reset Utilities"
Cohesion: 0.31
Nodes (3): _delete_dir(), main(), _reset_via_chromadb()

### Community 8 - "Fact Attribution Parsing"
Cohesion: 0.08
Nodes (36): build_debug_attribution_stub(), GroundedFact, _lenient_fact(), _norm_chunk_id(), repair_json(), _resolve_chunk_id(), salvage_grounded_facts_json(), _scan_fact_objects() (+28 more)

### Community 9 - "Fact Extraction Logic"
Cohesion: 0.16
Nodes (11): ParsedFacts, extract_grounded_facts(), _extract_grounded_facts_local(), _grounded_facts_provider(), _hf_chat_extract_json(), _hf_chat_extract_json_with_retry(), _hf_token(), _invoke_local_facts() (+3 more)

### Community 10 - "Data Preparation Tools"
Cohesion: 0.17
Nodes (10): _eval_to_dict(), evaluate_story_file_result(), evaluate_story_text_result(), evaluate_summary_result(), generate_summary_result(), ingest_stories_result(), query_vector_result(), step1_prepare_and_enrich_result() (+2 more)

### Community 11 - "Chroma Metadata Reporting"
Cohesion: 0.16
Nodes (15): main(), fetch_all(), format_report(), _pct(), problems(), _s(), summarize_metadata(), _md() (+7 more)

### Community 12 - "Configuration Management"
Cohesion: 0.13
Nodes (17): _default_n_results(), load_config(), resolve_config_path(), _generation_prompts(), test_example_config_exposes_generation_precision_flag(), test_example_config_exposes_hf_grounded_facts_json_mode_flag(), test_example_config_exposes_ollama_generation_settings(), test_example_config_exposes_story_length_target_settings() (+9 more)

### Community 13 - "Fact Formatting Logic"
Cohesion: 0.21
Nodes (8): parse_grounded_facts_json(), test_attribution_violations_flags_new_named_entities(), test_format_facts_for_prompt_omits_quote_when_blank(), test_format_facts_for_prompt_produces_numbered_bullet_list(), test_format_facts_for_prompt_returns_empty_string_for_no_facts(), test_many_hallucinated_entities_trigger_truncation(), test_parse_grounded_facts_drops_missing_sources(), test_section_header_words_do_not_trigger_truncation()

### Community 14 - "Orchestration Management"
Cohesion: 0.06
Nodes (32): main(), _run_one(), summarize(), create_eval_status(), StatusResponse, story_evaluate(), story_evaluate_file(), story_generate() (+24 more)

### Community 15 - "Generation Logic Tests"
Cohesion: 0.06
Nodes (11): _auto_stub(), _cfg(), test_both_attempts_empty_raises_runtime_error(), test_fast_mode_empty_raises_runtime_error(), test_length_guard_falls_back_to_original_when_refine_raises(), test_thinking_mode_empty_retries_with_fast(), _FakeGemini, test_evaluate_model_falls_back_to_gemini_when_hf_unavailable() (+3 more)

### Community 16 - "Core Application Framework"
Cohesion: 0.15
Nodes (12): create_app(), book_status(), download_book(), DownloadBookRequest, DownloadBookResponse, extract_text(), ExtractTextRequest, ExtractTextResponse (+4 more)

### Community 17 - "Book Fetching Utilities"
Cohesion: 0.22
Nodes (8): download_archive_book(), extract_books_info(), _find_file_for_formats(), receive_book(), search_in_archive(), download_book_archive(), _identifier_already_downloaded(), search_books()

### Community 18 - "Ingestion Metadata Tests"
Cohesion: 0.23
Nodes (12): _FakeCollection, ingest(), run(), test_crlf_only_difference_is_not_stale(), test_matching_story_json_supplies_chunks_sections_and_metadata(), test_missing_story_json_keeps_old_behaviour(), test_record_with_only_empty_chunks_falls_back_to_txt(), test_stale_story_json_uses_txt_chunks_but_keeps_story_metadata() (+4 more)

### Community 19 - "Mocking and Stubs"
Cohesion: 0.09
Nodes (11): _build_stubs(), fake_docs(), fake_hf_response(), stub_heavy_deps(), _client_for(), _FakeOrchestrator, _GenMode, _load_create_eval_routes() (+3 more)

### Community 20 - "Prompt Management"
Cohesion: 0.15
Nodes (12): _get_generation_prompts(), load_vllm_llm(), build_refine_prompt(), build_story_prompt(), _flow_section_headers(), generate_from_facts(), _load_generation_llm(), _load_or_get_cached_local_model() (+4 more)

### Community 21 - "Length Calculation Logic"
Cohesion: 0.14
Nodes (6): length_presets(), length_token_cap(), _parse_target(), split_section_bodies(), _words_per_minute(), test_length_token_cap_ignores_missing_and_invalid_values()

### Community 22 - "Document Retrieval Logic"
Cohesion: 0.17
Nodes (9): _bm25_global_docs(), _bm25_rank_docs(), _bm25_tokens(), _docs_to_context(), retrieve_docs(), _rrf_fuse(), _select_diverse_stories(), test_retrieval_returns_relevant_chunk_from_real_chroma() (+1 more)

### Community 23 - "Retrieval Evaluation"
Cohesion: 0.14
Nodes (7): test_evaluate_retrieval_uses_query_function(), query_fn(), test_example_fixture_has_realistic_diverse_cases(), test_make_retrieve_docs_query_fn_routes_through_real_pipeline(), fake_docs_to_chunks(), fake_retrieve_docs(), test_retrieve_docs_query_fn_output_is_scorable_end_to_end()

### Community 24 - "Retrieval Regression Tests"
Cohesion: 0.18
Nodes (4): _FakeDoc, test_retrieve_docs_reranks_before_diversity_selection(), fake_rerank(), test_retrieve_docs_skips_diversity_narrowing_before_rerank_sees_it()

### Community 25 - "Environment and Config"
Cohesion: 0.27
Nodes (6): _load_config_cached(), load_prompts(), _load_prompts_cached(), _normalize_base_path(), overlay_api_keys_from_env(), try_load_dotenv()

### Community 26 - "Document Text Extraction"
Cohesion: 0.20
Nodes (6): extract_text_from_books(), extract_text_from_epub(), extract_text_from_pdf(), format_extracted_text(), _iter_files(), extract_text_result()

### Community 27 - "Integration Pipeline Tests"
Cohesion: 0.16
Nodes (5): cfg(), ollama_url(), store(), _story_text(), test_ollama_evaluator_scores_story_via_fake_server()

### Community 28 - "RAG Pipeline Logic"
Cohesion: 0.31
Nodes (6): _sections_below_min_sentences(), generate_story_3step_langchain(), RAG3StepResult, _docs_to_chunks(), test_local_facts_extraction_cites_only_retrieved_chunks(), test_three_step_pipeline_end_to_end()

### Community 30 - "Attribution Violation Checks"
Cohesion: 0.38
Nodes (4): attribution_violations(), extract_named_entities_heuristic(), _apply_attribution_gate(), _should_truncate()

### Community 31 - "Story Workflow Tests"
Cohesion: 0.36
Nodes (7): _read_json(), _set_tmp_config(), test_create_story_records_writes_editable_json(), test_records_to_ingest_manifest_omits_series_fields_when_not_series(), test_records_to_ingest_manifest_writes_jsonl(), test_story_json_to_manifest_workflow(), _write_series_template()

### Community 32 - "CLI and Evaluation"
Cohesion: 0.14
Nodes (3): main(), main(), main()

### Community 34 - "Data Ingestion Pipeline"
Cohesion: 0.13
Nodes (14): main(), _flush(), main(), get_or_create_collection(), _chunk_text(), _embed_chunks(), _get_embed_model(), ingest_stories_dir() (+6 more)

### Community 36 - "CUDA Compatibility Checks"
Cohesion: 0.32
Nodes (3): check_nvidia_smi(), check_pytorch_cuda(), run_cuda_compatibility_check()

### Community 38 - "Knowledge Graph Updates"
Cohesion: 0.48
Nodes (4): _ensure_model(), main(), _ollama_models(), _run()

## Knowledge Gaps
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `Configuration Management` to `CLI and Evaluation`, `Streaming API Routes`, `Data Ingestion Pipeline`, `Agentic Loop Logic`, `Vector Store Utilities`, `Fact Extraction Debugging`, `Data Reset Utilities`, `Data Preparation Tools`, `Chroma Metadata Reporting`, `Gemini Model Listing`, `Orchestration Management`, `Core Application Framework`, `Book Fetching Utilities`, `Environment and Config`, `Document Text Extraction`, `RAG Pipeline Logic`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Why does `resolve_length_profile()` connect `Length Profile Logic` to `Streaming API Routes`, `Agentic Loop Logic`, `Length Profile Generation`, `Data Preparation Tools`, `Orchestration Management`, `Prompt Management`, `Length Calculation Logic`, `RAG Pipeline Logic`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `run_agentic_story_loop()` connect `Agentic Loop Logic` to `Streaming API Routes`, `Length Profile Logic`, `Fact Extraction Logic`, `Data Preparation Tools`, `Configuration Management`, `Orchestration Management`, `Prompt Management`, `Document Retrieval Logic`, `RAG Pipeline Logic`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `load_config()` (e.g. with `main()` and `main()`) actually correct?**
  _`load_config()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `run_agentic_story_loop()` (e.g. with `Gen_mode` and `StoryType`) actually correct?**
  _`run_agentic_story_loop()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Orchestrator` (e.g. with `main()` and `_run_one()`) actually correct?**
  _`Orchestrator` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `Streaming API Routes` be split into smaller, more focused modules?**
  _Cohesion score 0.0574400723654455 - nodes in this community are weakly interconnected._