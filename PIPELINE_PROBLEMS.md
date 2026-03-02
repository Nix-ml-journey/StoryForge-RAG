# Pipeline — Flow and Function Reference

This document describes how the story generation and evaluation pipeline works: entry point, API, and each module’s flow and main functions. **Last updated**: 2025-02-10.

**Intended flow**: Run `main.py` → server starts. The user (or client) sends a **title** or query. That can drive **search** → **download** → **extract (OCR)** → **manual story separation** → **metadata from stories** → **check empty** → **manual metadata insert** → **merge** → **ingest** (vector store) → **generate** (story/summary) → **evaluate**. The user can either run the **full pipeline** in one go or call **each step via the API** (search only, then download, then extract, etc.).

**New flow (post-OCR)**: After OCR, the user **manually separates the story** from a single book (e.g. splits one extracted text into multiple `.txt` files and places them in **Story_input** / `Stories`). Then **Data/metadata.py** reads those separated `.txt` files and **creates one metadata JSON per txt** in **Metadata_input** (`Metadata`). Next, **check empty** runs (summary-empty count). Finally the **user does manual insert** for the metadata (fills Author, Title, Summary, etc. in the JSON files).

---

## 1. Entry point — `main.py`

**Flow**: Loads `setup.yaml` for `Port` (and paths used by other modules). Creates a FastAPI app, mounts five routers, exposes `GET /` (health/info and link to `/docs`), then starts uvicorn.

**Routers mounted**:
- `book_router`, `data_router`, `vector_store_router` (from `API.data_routers`)
- `orchestration_router`, `get_vector_router` (from `API.orchestration_routes`)

**Generation mode**: The API accepts `mode: "fast" | "thinking"` for story generation and run_pipeline; the orchestrator passes this to the generative module (SmolLM3 params from `setup.yaml`: `Generation_mode_fast` / `Generation_mode_thinking`).

---

## 2. API routes

### 2.1 `API/data_routers.py`

**Flow**: API-only. Three routers (book, data, vector_store) with async handlers. Each handler delegates to the **orchestrator**; no direct calls to Book_search, Vector_Store, etc. Request/response use Pydantic models. Download formats: pdf, epub.

**Routers and endpoints** (conceptually):
- **book_router**: search_book, download_book, status.
- **data_router**: metadata_check, metadata_update, clean_data, merged_data, status (clean_status, merged_status).
- **vector_store_router**: vector_store operations and status.

### 2.2 `API/orchestration_routes.py`

**Flow**: API-only. Two routers: **orchestration_router** and **get_vector_router**. All logic runs in the orchestrator (`_orchestrator`). Request bodies include optional `mode: "fast" | "thinking"` for generation.

**Main endpoints**:
- **POST /orchestration/run_pipeline** — Run pipeline steps (title, optional steps list, mode).
- **POST /orchestration/search_book**, **/download_book** — Book search and download.
- **POST /orchestration/story_generate** — Generate story (query, generation_type, save, n_results, mode).
- **POST /orchestration/story_evaluate**, **/story_evaluate_file**, **/summary_generate**, **/summary_evaluate** — Evaluation and summaries.
- **GET /orchestration/status**, **/docs** — Status and docs.
- **POST /get_vector_store** — Vector store query.

---

## 3. Book search

### 3.1 `Book_search/extract_text.py`

**Role**: Extract text from PDFs and EPUBs into plain text for the rest of the pipeline.

**Flow and main functions**:
- **extract_text_from_pdf(file_path)** — Opens PDF with fitz, gets text per page via `get_text("blocks_text")`, joins blocks; returns text or `None`; closes doc in `finally`.
- **extract_text_from_epub(file_path)** — Uses `EpubProcessor`, `export_chapters_markdown()`; reads `.md`/`.txt` from output dir; returns joined string or `None`.
- **format_ocr_text(text)** — Cleans and normalizes text (paragraphs, spaces).
- **extract_text_from_books()** — Reads from `BASE_PATH / downloaded_rawbook_dir`; for each PDF/EPUB, extracts and writes `.txt` under `BASE_PATH / extracted_text_dir`.
- **save_extracted_text()** — Ensures extracted-text dir exists; lists `.txt` files and returns the dir path.

After extraction, the user **manually separates** the story from a single book (splits into multiple `.txt` files) and places them in **Story_input** (`Stories`) for the metadata step.

### 3.2 `Book_search/fetch_book.py`

**Role**: Search and download books (e.g. Google Books, Archive). Outputs go to config folders.

**Flow**: When run as a script, loads `BASE_PATH` and folder names from config; builds `downloaded_rawbook_dir` and `downloaded_data_meta`; calls `download_archive_book(..., downloaded_rawbook_dir)` and `save_google_books(..., user_input, downloaded_data_meta)`. Config keys: `BASE_PATH`, `book_data`, `archive_url`, `Download_formats`, `archive_metadata_url`, `downloaded_rawbook_dir`, `downloaded_data_meta`.

### 3.3 `Data/metadata.py` (story-based metadata — new flow)

**Role**: Create metadata JSONs from **separated story** `.txt` files, check for empty summaries, and support manual metadata insert.

**Assumption**: The user has already run OCR (extract_text) and **manually separated** the story from a single book into multiple `.txt` files in **Story_input** (`Stories`).

**Flow and main functions**:
- **create_json_metadata_from_stories(BASE_PATH, story_input, metadata_input, database_schema)** — Reads all `.txt` in `Story_input`; for each file, creates one `.json` in `Metadata_input` with schema (ids, metadatas, chapters, series, documents). One metadata file per story txt.
- **check_summary_empty_count(BASE_PATH, metadata_input)** — Scans all metadata JSONs; returns `(total_count, empty_count, empty_list)` for Summary field. Used to see which files need manual filling.
- **is_summary_empty(data)** — Helper: true if Summary is missing or blank.
- **save_updated_metadata(file_path, updated_metadata)** — Writes updated metadata dict to file (e.g. after manual insert or API update).
- **retrieve_and_update_metadata(user_input)** — Loads one metadata JSON by filename; prompts for value and saves (CLI helper).
- **load_story**, **load_metadata**, **get_story_txt_files** — Helpers for listing/counting stories and metadata.

**Script flow** (when run as `python -m Data.metadata` or `python Data/metadata.py`): (1) Create metadata from stories → (2) Run check empty and print which summaries are empty → (3) Prompt user to do manual insert → (4) Run check empty again and print results.

---

## 4. Vector store and ingestion

**Data flow**: **Downloaded_rawbook_dir** / **Downloaded_data_meta** → **Extracted_text_dir** (OCR) → user places separated `.txt` in **Story_input** → **Data/metadata.py** creates one `.json` per `.txt` in **Metadata_input** → user fills metadata → **Data/data_merge** (reads Story_input + Metadata_input; writes **Merged_data_output**) → per-item JSON with `ID`, `Title`, `Author`, `Summary`, `Document` (content) → Chroma ingestion: `ids`, `metadatas`, `documents` built from merged JSONs.

### 4.1 `Vector_Store/chromadb.py`

**Role**: Store and query documents in Chroma. Ingestion uses the same embedding model as queries (SentenceTransformer).

**Flow and main functions**:
- **load_data(file_path)** — Loads one JSON file.
- **store_data(path)** — Accepts a **directory** (uses `load_merged_dir_for_chroma` + `Collection.add`) or a **file** (load + upsert). Uses `item.get("Document", item.get("Content", ""))` for text.
- **load_merged_dir_for_chroma(merge_output_dir)** — Reads all `.json` in dir; returns `(ids, metadatas, documents)`.
- **ingest_into_chroma(merge_output_dir=None)** — Defaults to config merge path; uses `ingest_with_sentence_transformer` (vector_store `create_embeddings`); adds to collection; returns count.
- **query_data**, **delete_data**, **check_data**, **get_data**, **update_data**, **get_all_data** — Standard collection operations.

### 4.2 `Vector_Store/vector_store.py`

**Role**: Create embeddings and encode queries. Used by chromadb for ingestion and by generative_ai for queries (one model for index and query).

**Flow**: Config uses `Merged_data_output`. **load_merged_data(BASE_PATH, Merge_Data_Output)** reads merged JSONs. **create_embeddings**, **encode_query** are used by chromadb and Generative_AI.

### 4.3 `Data/data_merge.py`

**Role**: Merge metadata and story text into one JSON per item for Chroma. Reads from **Story_input** (`.txt`) and **Metadata_input** (`.json`); matches by filename stem.

**Flow and main functions**:
- **Load_Story_Contents(BASE_PATH, Story_input)** — Reads all `.txt` in Story_input; returns `{ stem: story_text }`.
- **Load_Metadata_Contents(BASE_PATH, Metadata_input)** — Reads all `.json`; adds `_stem` from filename; returns list of dicts.
- **Merge_Metadata_and_Story(BASE_PATH, Story_input, Metadata_input, Merge_Data_Output)** — Loads stories and metadata; for each metadata record, attaches story by stem and sets `documents`; saves one JSON per item to Merged_data_output. Returns `(True, merged_count)` or `(False, 0)`.
- Merged JSON uses key **documents** for content; chromadb uses **Document** (fallback **Content**).

### 4.4 `Vector_Store/data_cleaning.py`

**Role**: Normalize metadata keys and clean values for merge and Chroma.

**Flow and main functions**:
- **load_Metadata(BASE_PATH, Metadata_input)** — Reads all `.json` in metadata dir; returns list of dicts.
- **clean_metadata(metadata)** — Normalizes keys (`file_name`→filename, `title`→Title, `author`/`authors`→Author, etc.); strips strings.
- **save_cleaned_metadata(BASE_PATH, cleaned_metadata, output_folder=None)** — Writes one JSON per key to output folder (default Metadata_input).
- **run_cleaning(BASE_PATH, Metadata_input)** — Loads raw metadata; flattens `books`; cleans each record; saves. Returns `(True, count)` or `(False, 0)`. Typical use: read from Extracted_metadata_dir, write to Cleaned_Metadata for data_merge.

---

## 5. Generative AI (`Generative_AI/generative_ai.py`)

**Role**: Query Chroma with the user’s prompt, build context from results, and generate story (or summary) with the generation model (e.g. SmolLM3). Supports two modes: **fast** (shorter, quicker) and **thinking** (longer, more careful).

**Flow and main functions**:
- **Config**: Uses `Chroma_path`, `Chroma_collection_name`, `GENERATIVE_MODEL`, `Generated_story_output`, `BASE_PATH`; generation params from `Generation_mode_fast` / `Generation_mode_thinking` (max_tokens, temperature, top_p).
- **Embeddings**: Imports `encode_query` and `model as embedding_model` from `Vector_Store.vector_store`; same model as Chroma ingestion.
- **get_generative_model(GENERATIVE_MODEL)** — Returns `(model, tokenizer)`.
- **get_generation_params(mode)** — Returns params for `"fast"` or `"thinking"` from setup.yaml.
- **generate_full_story(query, n_results=5, mode="fast")** — Encodes query, queries Chroma, builds context, calls `generate_response` with mode-specific params; returns generated text.
- **generate_response(..., max_tokens, temperature, top_p)** — Runs the language model with given params.
- **save_generated_story(content)** — Saves to `BASE_PATH / Generated_story_output` with timestamp.

---

## 6. Evaluation (`Evaluation/evaluation.py`)

**Role**: Evaluate generated stories and summaries (e.g. via Gemini), return scores and conclusions.

**Flow**: Reads `setup.yaml` and `prompts.yaml`; uses `Gemini_api_key`, `Gemini_evaluation_model`, output paths (`Generated_story_output`, `Generated_summary_output`, `Evaluated_stories_output`), `BASE_PATH`. **evaluate_model()** validates API key and model. Logic: find story/summary by date or path; run evaluation; write results to `BASE_PATH / Evaluated_stories_output`. The orchestrator and API routes call evaluation functions when the user triggers evaluate endpoints.

---

## 7. Pipeline orchestration

### 7.1 `orchestrator.py`

**Role**: Holds all pipeline logic. The API only receives HTTP requests and calls orchestrator methods; the orchestrator talks to Book_search, Vector_Store, Generative_AI, and (when wired) Evaluation.

**Flow**: Loads config in `__init__`. Defines **Gen_mode** (FAST / THINKING) for SmolLM3. Implements methods the API calls; each returns a dict with the fields the response models expect.

**Methods** (conceptually):
- **run_pipeline(title, steps=None, mode=Gen_mode.FAST)** — Runs requested steps (default: fetch, extract, clean, merge, ingest, generate, evaluate).
- **search_book**, **download_book** — Book search and download.
- **metadata_check**, **metadata_update**, **clean_data**, **merge_data** — Data and metadata operations (delegate to Data.metadata for story-based check/update, Vector_Store.data_cleaning, Data.data_merge).
- **vector_store_query**, **vector_store_insert**, **vector_store_update**, **vector_store_delete** — Chroma operations.
- **generate_story(query, generation_type, save, n_results, mode)** — Calls Generative_AI `generate_full_story` with mode; optionally saves.
- **evaluate_story**, **evaluate_story_file**, **generate_summary**, **evaluate_summary** — Summary and evaluation (orchestrator can wire to Evaluation module).
- **query_vector_store** — Query Chroma (e.g. for get_vector_store endpoint).

**Wiring**: Book_search (fetch_book: receive_book, extract_books_info; extract_text), Data (metadata: create_json_metadata_from_stories, check_summary_empty_count, save_updated_metadata; data_merge.Merge_Metadata_and_Story), Vector_Store (data_cleaning.run_cleaning, chromadb: query_data, delete_data, ingest_into_chroma), Generative_AI (generate_full_story with mode, save_generated_story).

### 7.2 `API/orchestration_routes.py` (handlers)

Handlers receive request bodies, call `_orchestrator` methods (e.g. `generate_story`, `run_pipeline`), and return the dict as the response model. Mode is parsed from the request (`_parse_gen_mode`) and passed to the orchestrator.

### 7.3 `whole_evaluation.py`

Placeholder script. Intended to run the full evaluation pipeline (e.g. evaluate all generated stories/summaries for a date). The real logic lives in `Evaluation.evaluation`; this file would invoke it.

---

## 8. Configuration (`setup.yaml`)

**Role**: Single config for paths, ports, API keys, and model params.

**Paths**: `BASE_PATH`, `Downloaded_rawbook_dir`, `Downloaded_data_meta`, `Extracted_text_dir`, `Extracted_metadata_dir`, `Story_input`, `Metadata_input`, `Merged_data_output`, `Chroma_path`, `Generated_story_output`, etc.

**Keys used by modules**: Book_search and Vector_Store use path keys above; chromadb uses `Chroma_path`, `Chroma_collection_name`; generative_ai uses `GENERATIVE_MODEL`, `Generated_story_output`, `Generation_mode_fast`, `Generation_mode_thinking` (and nested params); Evaluation uses `Gemini_api_key`, `Gemini_evaluation_model`, output folders. **Chroma_json_template** is a valid YAML string describing the target Chroma item shape (`ids`, `metadatas`, `documents`).

---

## 9. End-to-end pipeline flow

**Step 1 — Book search and download**  
Client sends title/query. API calls orchestrator → Book_search: search (e.g. Google Books), then download. PDFs and metadata are written to config folders (Downloaded_books, Downloaded_data_meta).

**Step 2 — Text extraction (OCR)**  
For each downloaded PDF/EPUB, run extraction (`extract_text_from_pdf` / `extract_text_from_epub`). Text is written under Extracted_text_dir. Merge and vector store expect text files, not raw PDFs.

**Step 3 — Manual story separation**  
The user **manually separates** the story from a single book: e.g. takes the extracted text (or one book’s text) and splits it into multiple `.txt` files (one story or chapter per file), then places those files in **Story_input** (`Stories`). No code runs in this step.

**Step 4 — Metadata from stories**  
Run **Data/metadata.py** (or the equivalent API/orchestrator step). **create_json_metadata_from_stories** reads all `.txt` in Story_input and creates **one metadata `.json` per `.txt`** in Metadata_input (`Metadata`), with empty Author, Title, Summary, etc.

**Step 5 — Check empty**  
Still in the metadata step (or right after): **check_summary_empty_count** runs and reports how many metadata files have empty Summary. The user can see which files need filling.

**Step 6 — Manual metadata insert**  
The user **manually fills** the metadata JSONs (Author, Title, Summary, chapters, series, etc.) in the `Metadata` folder. Optionally, **save_updated_metadata** or API **metadata_update** can be used to write back changes.

**Step 7 — Merge**  
`Merge_Metadata_and_Story` (data_merge) reads Metadata and story text (Story_input); matches by filename (stem); writes one JSON per item to Merged_data_output with ID, Title, Author, Summary, Document. Data_cleaning can be used beforehand if normalizing raw metadata from another source.

**Step 8 — Vector store ingestion**  
`ingest_into_chroma` reads all JSONs from Merged_data_output, creates embeddings with SentenceTransformer, adds to Chroma. Same embedding model is used for queries in generation.

**Step 9 — Story generation**  
Client sends query and optional `mode` (fast/thinking). Orchestrator calls `generate_full_story(query, n_results, mode)`. Generative_AI encodes query, queries Chroma, builds context, runs the language model with mode-specific params, returns (and optionally saves) the story.

**Step 10 — Evaluation**  
Client requests evaluation for a story (and optionally summary). Orchestrator calls Evaluation module; results (scores, conclusion) are returned and can be written to Evaluated_stories_output.

**Chaining**: The user can (1) run the **full pipeline** (one request with title/steps) or (2) call **each step separately** via the API (search, then download, then extract, then **place separated stories in Stories**, then run metadata-from-stories + check empty, then manual insert, then merge, ingest, generate, evaluate).

---

## Summary — Component overview

| Component | Role / flow |
|-----------|-------------|
| main.py | Load config (port); mount book_router, data_router, vector_store_router, orchestration_router, get_vector_router; GET /; run uvicorn. |
| data_routers.py | Three routers (book, data, vector_store); async handlers; delegate to orchestrator; pdf/epub download. |
| orchestration_routes.py | orchestration_router + get_vector_router; all handlers delegate to orchestrator; request/response include mode (fast/thinking) for generation. |
| extract_text.py | Extract text from PDF/EPUB; format; write to extracted_text_dir. |
| fetch_book.py | Search and download books; write to downloaded_rawbook_dir and downloaded_data_meta. |
| Data/metadata.py | create_json_metadata_from_stories (one .json per .txt in Story_input), check_summary_empty_count, save_updated_metadata; script: create → check empty → prompt manual insert → check again. |
| chromadb.py | load_data, store_data (dir or file), load_merged_dir_for_chroma, ingest_into_chroma (SentenceTransformer); query/delete/get/update. |
| vector_store.py | create_embeddings, encode_query; used by chromadb and generative_ai. |
| Data/data_merge.py | Load_Story_Contents, Load_Metadata_Contents, Merge_Metadata_and_Story → one JSON per item (documents key); returns (True, count). |
| data_cleaning.py | load_Metadata, clean_metadata, save_cleaned_metadata, run_cleaning; key normalization for merge. |
| generative_ai.py | Query Chroma, build context, get_generation_params(mode), generate_full_story(mode), save_generated_story. |
| evaluation.py | Evaluate story/summary (e.g. Gemini); output to Evaluated_stories_output. |
| orchestrator.py | Implements all API-called methods; wires to Book_search, Vector_Store, Generative_AI (and Evaluation when used); Gen_mode for SmolLM3. |
| whole_evaluation.py | Placeholder for a script that runs the full evaluation pipeline. |
| setup.yaml | BASE_PATH, port, paths, API keys, model names, Generation_mode_fast/thinking params, Chroma_json_template. |

---

## 10. Problems to fix for the new flow

These files are still wired to the **old** flow (metadata from downloaded books, `Merge_Data`, etc.). They need to be updated so the **new flow** (manual story separation → metadata from stories → check empty → manual insert → merge) works end-to-end.

| File | Problem | What to do |
|------|---------|------------|
| **Orchestrator/response_parameter.py** | Imports **`Book_search.metadata`** (`check_empty`, `metadata_extraction`, `save_updated_metadata`). **`Book_search/metadata.py` does not exist**; the new flow uses **Data/metadata.py**. | Switch imports to **Data.metadata**: `check_summary_empty_count`, `create_json_metadata_from_stories`, `save_updated_metadata`. |
| **Orchestrator/response_parameter.py** | **`metadata_check_result`** calls `check_empty(Path(folder_path), file_name)`, which does not exist. New flow uses **`check_summary_empty_count(BASE_PATH, metadata_input)`**, which returns `(total, empty_count, empty_list)`. | Implement metadata_check_result with `check_summary_empty_count`; pass `base_path` and `metadata_input` (e.g. from config or request). Return shape: e.g. `success = (empty_count == 0)`, `issues` = empty_list, `number_books` = total, `empty_number` = empty_count. |
| **Orchestrator/response_parameter.py** | **`metadata_extract_result`** calls **`metadata_extraction(raw_folder, extracted_metadata_dir)`** (old flow: extract from downloaded metadata). New flow creates metadata **from Story_input** (one .json per .txt). | Replace with **`create_json_metadata_from_stories(BASE_PATH, story_input, metadata_input, database_schema)`**. Signature can stay or change to `(base_path, story_input, metadata_input, database_schema)` from config; no raw_folder. |
| **Orchestrator/response_parameter.py** | Imports **`Merge_Data`** from **`Vector_Store.data_merge`**. **`Vector_Store/data_merge.py` does not exist**; merge lives in **Data/data_merge.py** as **`Merge_Metadata_and_Story`**. | Import **`Merge_Metadata_and_Story`** from **`Data.data_merge`**. In **`merge_data_result`**, call **`Merge_Metadata_and_Story(base_path, story_dir, meta_dir, out_dir)`** (note: story_dir then meta_dir). Adjust argument order if the current `merge_data_result(base_path, meta_dir, story_dir, out_dir)` is required by the API. |
| **Orchestrator/orchestrator.py** | **`vector_store_insert`** and **`vector_store_update`** pass **`chapters, series`** into `parameters.vector_insert_result` / `parameters.vector_update_result`. Those facades only accept **(ids, metadata)**; **chapters and series are undefined** → **NameError** at runtime. | Remove the extra `chapters, series` arguments from the orchestrator calls, or add them to the facade signatures if the API needs them. |
| **Orchestrator/orchestrator.py** | **`run_pipeline`** has no **"metadata"** step. New flow requires: after extract → user separates stories → **create metadata from stories** → **check empty** → **manual insert** → then merge. | Either add a **"metadata"** step that calls `create_json_metadata_from_stories` (and optionally `check_summary_empty_count`) so the API can drive it, or document that the user must run **Data/metadata.py** (and manual insert) before calling merge. |
| **whole_evaluation.py** | File is **empty**. Intended to run the full evaluation pipeline (e.g. evaluate all generated stories/summaries for a date). | Implement by calling **Evaluation.evaluation** (e.g. `orchestrator_evaluation(selected_date)` or equivalent); or keep as a documented placeholder. |

**Reference (new flow–aligned code):** **Orchestrator/orchestration_facades_Example.py** and **Orchestrator/orchestrator_EXAMPLE.py** already use **Data.metadata** and **Data.data_merge.Merge_Metadata_and_Story**. You can mirror their imports and logic in **response_parameter.py** and **orchestrator.py** when fixing the items above.
