# Graph Report - RAG  (2026-10-08)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1033 nodes · 2292 edges · 49 communities (38 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 121 edges (avg confidence: 0.88)
- Token cost: 3,850 input · 540 output

## Graph Freshness
- Built from commit: `113da1ad`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Streaming Generation
- Agentic Loop Logic
- Mocking and Stubs
- Length Profile Logic
- Vector Store Management
- API Testing Tools
- Reranking and Retrieval
- Data Reset Tools
- Fact Extraction Repair
- API Error Handling
- Evaluation Framework
- Metadata Reporting
- Configuration Management
- Rule Validation
- Orchestrator and Metrics
- Draft Recovery Tests
- Web Framework Core
- Content Acquisition
- Ingestion Metadata Tests
- Embedding Utilities
- Prompt Engineering
- Length Preset Mapping
- Hybrid Retrieval
- Retrieval Evaluation
- Retrieval Regression Tests
- Path and Config
- Document Parsing
- Integration Testing
- Chroma Database Helpers
- Data Ingestion Pipeline
- Attribution Filtering
- Manifest Generation
- CLI and Discovery
- Sectioned Generation Tests
- CUDA Compatibility Checks
- HTTP Server Handling
- Hashing Embeddings
- Length Profile Logic
- Fact Attribution
- Story Arc Generation

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 54 edges
2. `run_agentic_story_loop()` - 40 edges
3. `resolve_length_profile()` - 40 edges
4. `decide_action()` - 27 edges
5. `salvage_grounded_facts_json()` - 25 edges
6. `Orchestrator` - 22 edges
7. `retrieve_docs()` - 22 edges
8. `LengthProfile` - 21 edges
9. `generate_from_facts()` - 21 edges
10. `parse_grounded_facts_json()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `test_default_mode_is_single_pass()` --uses--> `ParsedFacts`  [INFERRED]
  tests/test_sectioned_generation.py → src/storyforge/rag/attribution.py
- `test_rule_driven_flag_off_or_wrong_method_uses_old_loop()` --calls--> `_rule_driven_enabled()`  [INFERRED]
  tests/test_rule_loop.py → src/storyforge/rag/agentic_loop.py
- `test_agentic_loop_end_to_end_uses_ollama_judge()` --calls--> `run_agentic_story_loop()`  [INFERRED]
  tests/test_integration_pipeline.py → src/storyforge/rag/agentic_loop.py
- `main()` --calls--> `load_config()`  [INFERRED]
  scripts/peek_vector_store.py → src/storyforge/config/config.py
- `test_arc_prompt_set_matches_the_section_placeholders()` --calls--> `load_prompts()`  [INFERRED]
  tests/test_sectioned_generation.py → src/storyforge/config/config.py

## Import Cycles
- None detected.

## Communities (49 total, 11 thin omitted)

### Community 0 - "Streaming Generation"
Cohesion: 0.08
Nodes (24): _stream_story_sse(), build_chat_ollama(), build_chat_openai(), generation_provider(), GenerationLLM, invoke_combined_prompt(), invoke_prompt(), load_ollama_llm() (+16 more)

### Community 1 - "Agentic Loop Logic"
Cohesion: 0.05
Nodes (72): AgenticLoopResult, average_score(), build_feedback(), completeness_report(), CompletenessReport, criterion_score(), decide_action(), Decision (+64 more)

### Community 2 - "Mocking and Stubs"
Cohesion: 0.06
Nodes (23): _build_stubs(), fake_docs(), fake_hf_response(), stub_heavy_deps(), _auto_stub(), _cfg(), _fake_hf_response(), test_disable_thinking_can_be_turned_off_by_config() (+15 more)

### Community 3 - "Length Profile Logic"
Cohesion: 0.14
Nodes (17): resolve_length_profile(), test_accept_gate_sits_below_the_target_but_above_half(), test_as_dict_reports_the_target_for_api_responses(), test_config_presets_override_builtins(), test_duration_target_converts_narration_minutes_to_words(), test_empty_config_still_resolves_a_usable_target(), test_explicit_length_wins_over_mode_default(), test_guidance_text_states_the_resolved_numbers() (+9 more)

### Community 4 - "Vector Store Management"
Cohesion: 0.12
Nodes (31): _configured_collection(), get_vector_store_count(), get_vector_store_ids(), get_vector_store_item(), list_vector_store(), _normalize_docs(), _normalize_ids(), _normalize_metas() (+23 more)

### Community 5 - "API Testing Tools"
Cohesion: 0.07
Nodes (13): app(), client(), _fake_docs(), _fake_parsed_facts(), _mock_rag_for_stream(), test_run_step_accepts_valid_step(), _client_for(), _FakeOrchestrator (+5 more)

### Community 6 - "Reranking and Retrieval"
Cohesion: 0.24
Nodes (3): _get_reranker(), _rerank_docs(), _resolve_device()

### Community 7 - "Data Reset Tools"
Cohesion: 0.31
Nodes (3): _delete_dir(), main(), _reset_via_chromadb()

### Community 8 - "Fact Extraction Repair"
Cohesion: 0.09
Nodes (36): repair_json(), salvage_grounded_facts_json(), _strip_json_fences(), _strip_think_blocks(), test_repair_json_strips_fences_and_trailing_commas(), extraction(), _fake_llm(), _local_cfg() (+28 more)

### Community 9 - "API Error Handling"
Cohesion: 0.06
Nodes (25): main(), _sample_chunks(), is_retryable_api_error(), parse_grounded_facts_json(), ParsedFacts, _extract_grounded_facts_llm(), _extract_grounded_facts_local(), _grounded_facts_provider() (+17 more)

### Community 10 - "Evaluation Framework"
Cohesion: 0.09
Nodes (6): _FakeGemini, test_evaluate_model_falls_back_to_gemini_when_hf_unavailable(), test_evaluate_model_returns_ollama_evaluator(), test_evaluate_story_text_retries_once_on_unparseable_judge_output(), test_invoke_with_retry_uses_gemini_fallback_after_hf_transient_error(), test_ollama_unreachable_fails_explicitly_without_api_fallback()

### Community 11 - "Metadata Reporting"
Cohesion: 0.16
Nodes (15): main(), fetch_all(), format_report(), _pct(), problems(), _s(), summarize_metadata(), _md() (+7 more)

### Community 12 - "Configuration Management"
Cohesion: 0.18
Nodes (14): load_config(), _generation_prompts(), test_example_config_exposes_generation_precision_flag(), test_example_config_exposes_hf_grounded_facts_json_mode_flag(), test_example_config_exposes_ollama_generation_settings(), test_example_config_exposes_story_length_target_settings(), test_example_config_exposes_thinking_mode_generation_flags(), test_example_config_exposes_vllm_generation_settings() (+6 more)

### Community 13 - "Rule Validation"
Cohesion: 0.21
Nodes (17): section_failures(), _auto_stub(), _body(), _facts(), _loop_cfg(), _patch_chain(), _story(), test_clean_story_has_no_failures() (+9 more)

### Community 14 - "Orchestrator and Metrics"
Cohesion: 0.06
Nodes (28): format_comparison(), _install_overrides(), main(), _mean(), _run_one(), summarize(), summarize_runs(), generate_stream() (+20 more)

### Community 15 - "Draft Recovery Tests"
Cohesion: 0.11
Nodes (6): _auto_stub(), _cfg(), test_both_attempts_empty_raises_runtime_error(), test_fast_mode_empty_raises_runtime_error(), test_length_guard_falls_back_to_original_when_refine_raises(), test_thinking_mode_empty_retries_with_fast()

### Community 16 - "Web Framework Core"
Cohesion: 0.08
Nodes (31): create_app(), create_eval_status(), _default_n_results(), StatusResponse, story_evaluate(), story_evaluate_file(), story_generate(), StoryEvaluateFileRequest (+23 more)

### Community 17 - "Content Acquisition"
Cohesion: 0.12
Nodes (16): download_archive_book(), extract_books_info(), _find_file_for_formats(), receive_book(), search_in_archive(), download_book_archive(), _eval_to_dict(), evaluate_story_file_result() (+8 more)

### Community 18 - "Ingestion Metadata Tests"
Cohesion: 0.23
Nodes (12): _FakeCollection, ingest(), run(), test_crlf_only_difference_is_not_stale(), test_matching_story_json_supplies_chunks_sections_and_metadata(), test_missing_story_json_keeps_old_behaviour(), test_record_with_only_empty_chunks_falls_back_to_txt(), test_stale_story_json_uses_txt_chunks_but_keeps_story_metadata() (+4 more)

### Community 19 - "Embedding Utilities"
Cohesion: 0.18
Nodes (5): query_data(), embed_query(), embed_texts(), get_embed_model(), is_bge_model()

### Community 20 - "Prompt Engineering"
Cohesion: 0.14
Nodes (13): format_facts_for_prompt(), _get_generation_prompts(), build_refine_prompt(), build_story_prompt(), _flow_section_headers(), generate_from_facts(), _invoke_nonempty(), _load_generation_llm() (+5 more)

### Community 21 - "Length Preset Mapping"
Cohesion: 0.25
Nodes (3): length_presets(), _parse_target(), _words_per_minute()

### Community 22 - "Hybrid Retrieval"
Cohesion: 0.18
Nodes (8): _bm25_global_docs(), _bm25_rank_docs(), _bm25_tokens(), retrieve_docs(), _rrf_fuse(), _select_diverse_stories(), test_retrieval_returns_relevant_chunk_from_real_chroma(), test_story_type_filters_on_is_series()

### Community 23 - "Retrieval Evaluation"
Cohesion: 0.15
Nodes (7): test_evaluate_retrieval_uses_query_function(), query_fn(), test_example_fixture_has_realistic_diverse_cases(), test_make_retrieve_docs_query_fn_routes_through_real_pipeline(), fake_docs_to_chunks(), fake_retrieve_docs(), test_retrieve_docs_query_fn_output_is_scorable_end_to_end()

### Community 24 - "Retrieval Regression Tests"
Cohesion: 0.18
Nodes (4): _FakeDoc, test_retrieve_docs_reranks_before_diversity_selection(), fake_rerank(), test_retrieve_docs_skips_diversity_narrowing_before_rerank_sees_it()

### Community 25 - "Path and Config"
Cohesion: 0.21
Nodes (7): _load_config_cached(), _load_prompts_cached(), _normalize_base_path(), resolve_config_path(), overlay_api_keys_from_env(), try_load_dotenv(), test_resolve_config_path_honors_explicit_path()

### Community 26 - "Document Parsing"
Cohesion: 0.24
Nodes (5): extract_text_from_books(), extract_text_from_epub(), extract_text_from_pdf(), format_extracted_text(), _iter_files()

### Community 27 - "Integration Testing"
Cohesion: 0.12
Nodes (7): cfg(), ollama_url(), store(), _story_text(), test_agentic_loop_end_to_end_uses_ollama_judge(), test_global_bm25_recovers_story_dense_retrieval_missed(), test_ollama_evaluator_scores_story_via_fake_server()

### Community 28 - "Chroma Database Helpers"
Cohesion: 0.25
Nodes (6): _build_vectorstore(), embed_query(), _get_paths_and_names(), _merge_filters(), story_type_filter(), test_story_type_filter_maps_to_is_series()

### Community 29 - "Data Ingestion Pipeline"
Cohesion: 0.13
Nodes (14): main(), _flush(), main(), get_or_create_collection(), _chunk_text(), _embed_chunks(), _get_embed_model(), ingest_stories_dir() (+6 more)

### Community 30 - "Attribution Filtering"
Cohesion: 0.28
Nodes (5): attribution_violations(), extract_named_entities_heuristic(), _apply_attribution_gate(), without_sentence_starts(), _should_truncate()

### Community 31 - "Manifest Generation"
Cohesion: 0.32
Nodes (7): _read_json(), _set_tmp_config(), test_create_story_records_writes_editable_json(), test_records_to_ingest_manifest_omits_series_fields_when_not_series(), test_records_to_ingest_manifest_writes_jsonl(), test_story_json_to_manifest_workflow(), _write_series_template()

### Community 32 - "CLI and Discovery"
Cohesion: 0.07
Nodes (11): list_gemini_models(), main(), is_marker(), merge_file(), main(), main(), main(), _ensure_model() (+3 more)

### Community 34 - "Sectioned Generation Tests"
Cohesion: 0.16
Nodes (15): _auto_stub(), _body(), _run(), _run_arc(), _Scripted, test_5w1h_still_selectable(), test_arc_method_uses_its_own_outline_prompts_and_section_roles(), test_arc_prompt_set_matches_the_section_placeholders() (+7 more)

### Community 36 - "CUDA Compatibility Checks"
Cohesion: 0.32
Nodes (3): check_nvidia_smi(), check_pytorch_cuda(), run_cuda_compatibility_check()

### Community 49 - "Length Profile Logic"
Cohesion: 0.13
Nodes (7): _sections_below_min_sentences(), _sections_to_write(), _build_profile(), LengthProfile, sentence_count(), split_section_bodies(), trim_overlong()

### Community 50 - "Fact Attribution"
Cohesion: 0.15
Nodes (10): build_debug_attribution_stub(), extractive_facts(), GroundedFact, _lenient_fact(), _norm_chunk_id(), _resolve_chunk_id(), _scan_fact_objects(), generate_story_3step_langchain() (+2 more)

### Community 51 - "Story Arc Generation"
Cohesion: 0.18
Nodes (7): load_prompts(), _arc_prompts(), generate_arc(), _generate_sectioned(), _trim_to_sentence(), length_token_cap(), test_length_token_cap_ignores_missing_and_invalid_values()

## Knowledge Gaps
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `Configuration Management` to `CLI and Discovery`, `Streaming Generation`, `Agentic Loop Logic`, `Vector Store Management`, `Data Reset Tools`, `API Error Handling`, `Metadata Reporting`, `Orchestrator and Metrics`, `Web Framework Core`, `Content Acquisition`, `Fact Attribution`, `Embedding Utilities`, `Path and Config`, `Document Parsing`, `Data Ingestion Pipeline`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `run_agentic_story_loop()` connect `Agentic Loop Logic` to `Length Profile Logic`, `Configuration Management`, `Rule Validation`, `Orchestrator and Metrics`, `Content Acquisition`, `Prompt Engineering`, `Hybrid Retrieval`, `Integration Testing`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `story_type_filter()` connect `Chroma Database Helpers` to `Reranking and Retrieval`, `Hybrid Retrieval`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `load_config()` (e.g. with `main()` and `main()`) actually correct?**
  _`load_config()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `run_agentic_story_loop()` (e.g. with `Gen_mode` and `StoryType`) actually correct?**
  _`run_agentic_story_loop()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Should `Streaming Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.08309178743961353 - nodes in this community are weakly interconnected._
- **Should `Agentic Loop Logic` be split into smaller, more focused modules?**
  _Cohesion score 0.05154639175257732 - nodes in this community are weakly interconnected._