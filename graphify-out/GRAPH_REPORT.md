# Graph Report - RAG  (2026-10-07)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 989 nodes · 2155 edges · 62 communities (41 shown, 21 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 113 edges (avg confidence: 0.88)
- Token cost: 4,725 input · 709 output

## Graph Freshness
- Built from commit: `36854774`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- LLM Provider Integration
- Agentic Loop Logic
- Fact Extraction Tests
- Story Length Calculation
- Vector Store Management
- API Contract Tests
- Hybrid Retrieval System
- Data Reset Utilities
- Local Fact Parsing
- HuggingFace Fact Extraction
- Evaluation System
- Chroma Metadata Reporting
- Configuration Management
- Fact Parsing Logic
- Orchestrator and Evaluation
- Draft Recovery Tests
- FastAPI Web Server
- Book Data Processing
- Story Metadata Ingestion
- Mocking and Stubs
- Streaming Response Logic
- Length and Token Limits
- Hybrid Retrieval Logic
- Retrieval Evaluation
- Retrieval Regression Tests
- Configuration and Environment
- Document Text Extraction
- Integration Pipeline Tests
- Three-Step RAG Pipeline
- Data Ingestion Pipeline
- Attribution Gate Logic
- Story Workflow Tests
- Vector Store Utilities
- Length Profile Data
- Sectioned Generation Tests
- CUDA Compatibility Checks
- HuggingFace Debugging
- Knowledge Graph Updates
- HTTP Request Handling
- Embedding Utilities
- API Error Handling
- Model Listing Utilities
- Provider Error Types
- Agentic Loop Core
- Attribution and Fact Parsing
- Story Arc Generation
- Evaluation Route Tests
- Prompt Management
- Mocked RAG Components
- Mocked Orchestrator
- File Merging Utilities
- Extraction Utilities
- Test Parameterization

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 54 edges
2. `resolve_length_profile()` - 40 edges
3. `run_agentic_story_loop()` - 32 edges
4. `decide_action()` - 26 edges
5. `salvage_grounded_facts_json()` - 25 edges
6. `Orchestrator` - 22 edges
7. `retrieve_docs()` - 22 edges
8. `Gen_mode` - 19 edges
9. `_stream_story_sse()` - 19 edges
10. `generate_from_facts()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `test_default_mode_is_single_pass()` --uses--> `ParsedFacts`  [INFERRED]
  tests/test_sectioned_generation.py → src/storyforge/rag/attribution.py
- `main()` --calls--> `load_config()`  [INFERRED]
  scripts/peek_vector_store.py → src/storyforge/config/config.py
- `_flush()` --calls--> `_embed_chunks()`  [INFERRED]
  scripts/ingest_manifest.py → src/storyforge/vector_store/ingest_stories.py
- `test_arc_prompt_set_matches_the_section_placeholders()` --calls--> `load_prompts()`  [INFERRED]
  tests/test_sectioned_generation.py → src/storyforge/config/config.py
- `test_is_thinking_mode_accepts_enum_values_and_aliases()` --uses--> `Gen_mode`  [INFERRED]
  tests/test_length_profile.py → src/storyforge/rag/generative_ai.py

## Import Cycles
- None detected.

## Communities (62 total, 21 thin omitted)

### Community 0 - "LLM Provider Integration"
Cohesion: 0.09
Nodes (23): build_chat_ollama(), build_chat_openai(), generation_provider(), GenerationLLM, invoke_combined_prompt(), invoke_prompt(), load_ollama_llm(), ollama_base_url() (+15 more)

### Community 1 - "Agentic Loop Logic"
Cohesion: 0.06
Nodes (60): AgenticLoopResult, average_score(), build_feedback(), completeness_report(), CompletenessReport, criterion_score(), decide_action(), Decision (+52 more)

### Community 2 - "Fact Extraction Tests"
Cohesion: 0.10
Nodes (19): _auto_stub(), _cfg(), _fake_hf_response(), test_disable_thinking_can_be_turned_off_by_config(), _fake_chat_completion(), test_disables_thinking_by_default(), _fake_chat_completion(), test_drops_both_optional_kwargs_on_type_error() (+11 more)

### Community 3 - "Story Length Calculation"
Cohesion: 0.14
Nodes (17): resolve_length_profile(), test_accept_gate_sits_below_the_target_but_above_half(), test_as_dict_reports_the_target_for_api_responses(), test_config_presets_override_builtins(), test_duration_target_converts_narration_minutes_to_words(), test_empty_config_still_resolves_a_usable_target(), test_explicit_length_wins_over_mode_default(), test_guidance_text_states_the_resolved_numbers() (+9 more)

### Community 4 - "Vector Store Management"
Cohesion: 0.06
Nodes (48): main(), _configured_collection(), get_vector_store_count(), get_vector_store_ids(), get_vector_store_item(), list_vector_store(), _normalize_docs(), _normalize_ids() (+40 more)

### Community 5 - "API Contract Tests"
Cohesion: 0.13
Nodes (3): app(), client(), test_run_step_accepts_valid_step()

### Community 6 - "Hybrid Retrieval System"
Cohesion: 0.15
Nodes (9): _build_vectorstore(), embed_query(), _get_paths_and_names(), _get_reranker(), _merge_filters(), _rerank_docs(), _resolve_device(), story_type_filter() (+1 more)

### Community 7 - "Data Reset Utilities"
Cohesion: 0.31
Nodes (3): _delete_dir(), main(), _reset_via_chromadb()

### Community 8 - "Local Fact Parsing"
Cohesion: 0.13
Nodes (28): salvage_grounded_facts_json(), _fake_llm(), _local_cfg(), test_hf_failure_falls_back_to_hardened_local_path(), test_load_facts_llm_passes_schema_to_ollama(), test_local_failure_logs_error_and_returns_empty(), test_local_json_format_defaults_to_schema_and_falls_back_when_rejected(), _loader() (+20 more)

### Community 9 - "HuggingFace Fact Extraction"
Cohesion: 0.14
Nodes (14): merge_facts(), ParsedFacts, extract_grounded_facts(), _extract_grounded_facts_llm(), _extract_grounded_facts_local(), _grounded_facts_provider(), _hf_chat_extract_json(), _hf_chat_extract_json_with_retry() (+6 more)

### Community 10 - "Evaluation System"
Cohesion: 0.09
Nodes (6): _FakeGemini, test_evaluate_model_falls_back_to_gemini_when_hf_unavailable(), test_evaluate_model_returns_ollama_evaluator(), test_evaluate_story_text_retries_once_on_unparseable_judge_output(), test_invoke_with_retry_uses_gemini_fallback_after_hf_transient_error(), test_ollama_unreachable_fails_explicitly_without_api_fallback()

### Community 11 - "Chroma Metadata Reporting"
Cohesion: 0.16
Nodes (15): main(), fetch_all(), format_report(), _pct(), problems(), _s(), summarize_metadata(), _md() (+7 more)

### Community 12 - "Configuration Management"
Cohesion: 0.13
Nodes (17): _default_n_results(), load_config(), resolve_config_path(), _generation_prompts(), test_example_config_exposes_generation_precision_flag(), test_example_config_exposes_hf_grounded_facts_json_mode_flag(), test_example_config_exposes_ollama_generation_settings(), test_example_config_exposes_story_length_target_settings() (+9 more)

### Community 13 - "Fact Parsing Logic"
Cohesion: 0.17
Nodes (11): parse_grounded_facts_json(), repair_json(), _strip_json_fences(), test_attribution_violations_flags_new_named_entities(), test_format_facts_for_prompt_omits_quote_when_blank(), test_format_facts_for_prompt_produces_numbered_bullet_list(), test_format_facts_for_prompt_returns_empty_string_for_no_facts(), test_many_hallucinated_entities_trigger_truncation() (+3 more)

### Community 14 - "Orchestrator and Evaluation"
Cohesion: 0.06
Nodes (26): format_comparison(), _install_overrides(), main(), _run_one(), summarize(), generate_stream(), GenerateStreamRequest, orchestration_status() (+18 more)

### Community 15 - "Draft Recovery Tests"
Cohesion: 0.12
Nodes (6): _auto_stub(), _cfg(), test_both_attempts_empty_raises_runtime_error(), test_fast_mode_empty_raises_runtime_error(), test_length_guard_falls_back_to_original_when_refine_raises(), test_thinking_mode_empty_retries_with_fast()

### Community 16 - "FastAPI Web Server"
Cohesion: 0.09
Nodes (30): create_app(), create_eval_status(), StatusResponse, story_evaluate(), story_evaluate_file(), story_generate(), StoryEvaluateFileRequest, StoryEvaluateRequest (+22 more)

### Community 17 - "Book Data Processing"
Cohesion: 0.12
Nodes (16): download_archive_book(), extract_books_info(), _find_file_for_formats(), receive_book(), search_in_archive(), download_book_archive(), _eval_to_dict(), evaluate_story_file_result() (+8 more)

### Community 18 - "Story Metadata Ingestion"
Cohesion: 0.23
Nodes (12): _FakeCollection, ingest(), run(), test_crlf_only_difference_is_not_stale(), test_matching_story_json_supplies_chunks_sections_and_metadata(), test_missing_story_json_keeps_old_behaviour(), test_record_with_only_empty_chunks_falls_back_to_txt(), test_stale_story_json_uses_txt_chunks_but_keeps_story_metadata() (+4 more)

### Community 19 - "Mocking and Stubs"
Cohesion: 0.15
Nodes (4): _build_stubs(), fake_docs(), fake_hf_response(), stub_heavy_deps()

### Community 20 - "Streaming Response Logic"
Cohesion: 0.16
Nodes (9): _stream_story_sse(), load_vllm_llm(), _invoke_nonempty(), _load_generation_llm(), _load_or_get_cached_local_model(), _mode_generation_params(), _resolve_generation_dtype(), is_thinking_mode() (+1 more)

### Community 21 - "Length and Token Limits"
Cohesion: 0.22
Nodes (5): length_presets(), length_token_cap(), _parse_target(), _words_per_minute(), test_length_token_cap_ignores_missing_and_invalid_values()

### Community 22 - "Hybrid Retrieval Logic"
Cohesion: 0.18
Nodes (8): _bm25_global_docs(), _bm25_rank_docs(), _bm25_tokens(), retrieve_docs(), _rrf_fuse(), _select_diverse_stories(), test_retrieval_returns_relevant_chunk_from_real_chroma(), test_story_type_filters_on_is_series()

### Community 23 - "Retrieval Evaluation"
Cohesion: 0.14
Nodes (7): test_evaluate_retrieval_uses_query_function(), query_fn(), test_example_fixture_has_realistic_diverse_cases(), test_make_retrieve_docs_query_fn_routes_through_real_pipeline(), fake_docs_to_chunks(), fake_retrieve_docs(), test_retrieve_docs_query_fn_output_is_scorable_end_to_end()

### Community 24 - "Retrieval Regression Tests"
Cohesion: 0.18
Nodes (4): _FakeDoc, test_retrieve_docs_reranks_before_diversity_selection(), fake_rerank(), test_retrieve_docs_skips_diversity_narrowing_before_rerank_sees_it()

### Community 25 - "Configuration and Environment"
Cohesion: 0.27
Nodes (5): _load_config_cached(), _load_prompts_cached(), _normalize_base_path(), overlay_api_keys_from_env(), try_load_dotenv()

### Community 26 - "Document Text Extraction"
Cohesion: 0.24
Nodes (5): extract_text_from_books(), extract_text_from_epub(), extract_text_from_pdf(), format_extracted_text(), _iter_files()

### Community 27 - "Integration Pipeline Tests"
Cohesion: 0.12
Nodes (6): cfg(), ollama_url(), store(), _story_text(), test_global_bm25_recovers_story_dense_retrieval_missed(), test_ollama_evaluator_scores_story_via_fake_server()

### Community 28 - "Three-Step RAG Pipeline"
Cohesion: 0.24
Nodes (8): retrieve_and_extract(), build_debug_attribution_stub(), generate_story_3step_langchain(), RAG3StepResult, _docs_to_chunks(), _docs_to_context(), test_local_facts_extraction_cites_only_retrieved_chunks(), test_three_step_pipeline_end_to_end()

### Community 30 - "Attribution Gate Logic"
Cohesion: 0.38
Nodes (4): attribution_violations(), extract_named_entities_heuristic(), _apply_attribution_gate(), _should_truncate()

### Community 31 - "Story Workflow Tests"
Cohesion: 0.36
Nodes (7): _read_json(), _set_tmp_config(), test_create_story_records_writes_editable_json(), test_records_to_ingest_manifest_omits_series_fields_when_not_series(), test_records_to_ingest_manifest_writes_jsonl(), test_story_json_to_manifest_workflow(), _write_series_template()

### Community 34 - "Sectioned Generation Tests"
Cohesion: 0.16
Nodes (14): _auto_stub(), _body(), _run(), _run_arc(), _Scripted, test_5w1h_stays_the_default_method(), test_arc_method_uses_its_own_outline_prompts_and_section_roles(), test_arc_prompt_set_matches_the_section_placeholders() (+6 more)

### Community 36 - "CUDA Compatibility Checks"
Cohesion: 0.32
Nodes (3): check_nvidia_smi(), check_pytorch_cuda(), run_cuda_compatibility_check()

### Community 38 - "Knowledge Graph Updates"
Cohesion: 0.48
Nodes (4): _ensure_model(), main(), _ollama_models(), _run()

### Community 45 - "Model Listing Utilities"
Cohesion: 0.22
Nodes (3): list_gemini_models(), main(), main()

### Community 49 - "Agentic Loop Core"
Cohesion: 0.20
Nodes (5): format_facts_for_prompt(), _sections_below_min_sentences(), _sections_to_write(), sentence_count(), split_section_bodies()

### Community 50 - "Attribution and Fact Parsing"
Cohesion: 0.21
Nodes (8): extractive_facts(), GroundedFact, _lenient_fact(), _norm_chunk_id(), _resolve_chunk_id(), _scan_fact_objects(), _strip_think_blocks(), test_extractive_facts_are_deterministic_verbatim_and_tagged_with_chunk_ids()

### Community 51 - "Story Arc Generation"
Cohesion: 0.24
Nodes (5): load_prompts(), _arc_prompts(), generate_arc(), _generate_sectioned(), _trim_to_sentence()

### Community 55 - "Evaluation Route Tests"
Cohesion: 0.36
Nodes (6): _client_for(), _GenMode, _load_create_eval_routes(), test_status_route_is_available(), test_story_evaluate_route_returns_scores(), test_story_generate_route_returns_generation_payload()

### Community 56 - "Prompt Management"
Cohesion: 0.24
Nodes (5): _get_generation_prompts(), build_refine_prompt(), build_story_prompt(), _flow_section_headers(), generate_from_facts()

### Community 57 - "Mocked RAG Components"
Cohesion: 0.29
Nodes (3): _fake_docs(), _fake_parsed_facts(), _mock_rag_for_stream()

## Knowledge Gaps
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `Configuration Management` to `Vector Store Utilities`, `Agentic Loop Logic`, `Vector Store Management`, `HuggingFace Debugging`, `Data Reset Utilities`, `Chroma Metadata Reporting`, `Model Listing Utilities`, `Orchestrator and Evaluation`, `FastAPI Web Server`, `Book Data Processing`, `Agentic Loop Core`, `Streaming Response Logic`, `Configuration and Environment`, `Document Text Extraction`, `Three-Step RAG Pipeline`, `Data Ingestion Pipeline`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `resolve_length_profile()` connect `Story Length Calculation` to `Agentic Loop Logic`, `Length Profile Data`, `Sectioned Generation Tests`, `Orchestrator and Evaluation`, `Book Data Processing`, `Agentic Loop Core`, `Streaming Response Logic`, `Length and Token Limits`, `Prompt Management`, `Three-Step RAG Pipeline`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `run_agentic_story_loop()` connect `Agentic Loop Logic` to `Story Length Calculation`, `HuggingFace Fact Extraction`, `Configuration Management`, `Orchestrator and Evaluation`, `Book Data Processing`, `Agentic Loop Core`, `Streaming Response Logic`, `Hybrid Retrieval Logic`, `Prompt Management`, `Three-Step RAG Pipeline`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `load_config()` (e.g. with `main()` and `main()`) actually correct?**
  _`load_config()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `run_agentic_story_loop()` (e.g. with `Gen_mode` and `StoryType`) actually correct?**
  _`run_agentic_story_loop()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `LLM Provider Integration` be split into smaller, more focused modules?**
  _Cohesion score 0.08859357696567 - nodes in this community are weakly interconnected._
- **Should `Agentic Loop Logic` be split into smaller, more focused modules?**
  _Cohesion score 0.06049213943950786 - nodes in this community are weakly interconnected._