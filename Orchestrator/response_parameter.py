import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from Book_search.extract_text import extract_text_from_books
from Book_search.fetch_book import download_archive_book, extract_books_info, receive_book, search_in_archive
from Data.metadata import (
    check_summary_empty_count,
    save_updated_metadata,
    create_json_metadata_from_stories,
    BASE_PATH as DATA_BASE_PATH,
    story_input as DATA_STORY_INPUT,
    metadata_input as DATA_METADATA_INPUT,
    database_schema as DATA_DATABASE_SCHEMA,
)
from Evaluation.evaluation import evaluate_generated_story, evaluate_generated_summary, evaluate_model, save_evaluation_results
from Generative_AI import generative_ai
from Generative_AI.generative_ai import Gen_mode
from Vector_Store.chromadb import Collection, delete_data, query_data, update_data, ingest_into_chroma
from Data.data_merge import (
    Merge_Metadata_and_Story,
    create_summaries_for_merged_data,
    check_number_summaries,
)
import Data.data_merge as data_merge
from Evaluation.evaluation import evaluate_model, evaluate_generated_story, evaluate_generated_summary, save_evaluation_results

LOG = logging.getLogger(__name__)
FAIL = {"success": False, "results": [], "metadata": {}, "urls": []}
FAIL_SAVE = {"success": False, "saved": False, "saved_path": None}
EVAL_FAIL = {"success": False, "evaluation_type": "list[str]", "average_score": 0.0, "scores": {}, "summary": "", "suggestions": [], "metadata": {}, "conclusion": ""}

def search_books(api_key: str, query: str, n_results: int = 20) -> dict:
    try:
        r = receive_book(api_key, query)
        if not r:
            return FAIL
        books = extract_books_info(r)[:n_results]
        results = [{"id": b.get("id"), "title": b.get("title"), "authors": b.get("authors")} for b in books]
        urls = [{"url": (b.get("selfLink") or b.get("infoLink")), "source": "Google Book"} for b in books if b.get("selfLink") or b.get("infoLink")]
        return {"success": True, "results": results, "metadata": {}, "urls": urls}
    except Exception as e:
        LOG.exception("search_books failed")
        return FAIL

def _identifier_already_downloaded(output_books: Path, identifier: str) -> bool:
    if not identifier or not output_books.exists():
        return False
    prefix = f"archive__{identifier}__"
    for f in output_books.iterdir():
        if f.is_file() and f.name.startswith(prefix):
            return True
    return False

def download_book_archive(base_path: str, query: str, formats: list, raw_dir: str, meta_dir: str, archive_url: str) -> dict:
    if not archive_url:
        return FAIL_SAVE
    try:
        output_books = Path(base_path) / raw_dir
        output_meta = Path(base_path) / meta_dir
        book_info = {"title": query, "authors": []}
        docs = search_in_archive(book_info, archive_url)
        if not docs:
            return FAIL_SAVE
        d = None
        downloaded_file = None
        for doc in docs:
            identifier = doc.get("identifier")
            if not identifier:
                continue
            if _identifier_already_downloaded(output_books, identifier):
                LOG.info("Skipping result %s (already in %s), trying next.", identifier, raw_dir)
                continue
            downloaded_file = download_archive_book([doc], output_books, formats=formats)
            if downloaded_file:
                d = doc
                break
        if not downloaded_file or not d:
            return FAIL_SAVE
        output_meta.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_file = output_meta / f"archive_books_{ts}.json"
        creator = d.get("creator")
        creator = creator[0] if isinstance(creator, list) and creator else (creator or "Unknown")
        book_path = Path(downloaded_file) if not isinstance(downloaded_file, Path) else downloaded_file
        txt_filename = book_path.stem + ".txt"
        meta = {
            "search_query": query,
            "source": "archive.org",
            "timestamp": ts,
            "books": [{"title": d.get("title", query), "authors": [creator] if creator else [], "identifier": d.get("identifier"), "filename": txt_filename, "file_path": str(downloaded_file)}],
        }
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"success": True, "saved": True, "saved_path": str(meta_file)}
    except Exception as e:
        LOG.exception("download_book_archive failed")
        return FAIL_SAVE

def metadata_check_result(folder_path: str, file_name: str, number_books: int, empty_number: int) -> dict:
    try:
        total, empty_count, empty_list = check_summary_empty_count(Path(folder_path), file_name)
        return {"success": empty_count == 0, "issues": empty_list, "number_books": total, "empty_number": empty_count}
    except Exception as e:
        LOG.exception("metadata_check failed")
        return {"success": False, "issues": [str(e)], "number_books": number_books, "empty_number": empty_number}

def metadata_update_result(metadata: dict, folder_path: str, file_name: str) -> dict:
    try:
        save_updated_metadata(Path(folder_path) / file_name, metadata)
        return {"success": True, "metadata": metadata, "file_name": file_name}
    except Exception as e:
        LOG.exception("metadata_update failed")
        return {"success": False, "metadata": metadata, "file_name": file_name, "error": str(e)}

def create_metadata_template_result(number_books: int = 0, empty_number: int = 0) -> dict:
    try:
        template_count = create_json_metadata_from_stories(
            DATA_BASE_PATH, DATA_STORY_INPUT, DATA_METADATA_INPUT, DATA_DATABASE_SCHEMA
        )
        return {"success": True, "message": "Metadata templates created from stories", "template_count": template_count, "number_books": number_books, "empty_number": empty_number}
    except Exception as e:
        LOG.exception("create_metadata_template failed")
        return {"success": False, "message": str(e), "template_count": 0, "number_books": number_books, "empty_number": empty_number}

def metadata_extract_result(raw_folder: Path, extracted_metadata_dir: str, number_books: int, empty_number: int) -> dict:
    return create_metadata_template_result(number_books, empty_number)

def extract_text_result() -> dict:
    try:
        extract_text_from_books()
        return {"success": True, "message": "Text extraction completed"}
    except Exception as e:
        LOG.exception("extract_text failed")
        return {"success": False, "message": str(e), "error": str(e)}

def merge_data_result(base_path: str, story_dir: str, meta_dir: str, out_dir: str) -> dict:
    try:
        ok, total = Merge_Metadata_and_Story(base_path, story_dir, meta_dir, out_dir)
        return {"success": ok, "total_results": total}
    except Exception as e:
        LOG.exception("merge_data failed")
        return {"success": False, "total_results": 0, "error": str(e)}

def ingest_result(base_path: str, merged_output: str) -> dict:
    try:
        merge_dir = Path(base_path) / merged_output
        count = ingest_into_chroma(merge_output_dir=merge_dir)
        return {"success": True, "count": count}
    except Exception as e:
        LOG.exception("ingest failed")
        return {"success": False, "count": 0, "error": str(e)}

def create_summaries_result(base_path: str, merged_output: str) -> dict:
    try:
        ok, count = create_summaries_for_merged_data(
            base_path, merged_output,
            data_merge.Gemini_summary_model, data_merge.Gemini_api_key,
            data_merge.prompt_template, data_merge.refine_prompt,
        )
        return {"success": ok, "summaries_created": count}
    except Exception as e:
        LOG.exception("create_summaries failed")
        return {"success": False, "summaries_created": 0, "error": str(e)}

def create_summaries_and_check_result() -> dict:
    try:
        result = check_number_summaries(check_summary_empty_count, create_summaries_for_merged_data)
        if result is None:
            return {"success": False, "summaries_created": 0, "empty_count": 0, "created_count": 0, "counts_match": False, "error": "check_number_summaries returned None"}
        if len(result) == 2:
            ok, zero = result
            return {"success": ok, "summaries_created": zero, "empty_count": 0, "created_count": zero, "counts_match": False, "error": None if ok else "create_summaries failed"}
        ok, empty_count, created_count = result
        return {
            "success": ok,
            "summaries_created": created_count,
            "empty_count": empty_count,
            "created_count": created_count,
            "counts_match": empty_count == created_count,
            "error": None if ok else f"count mismatch: empty={empty_count}, created={created_count}",
        }
    except Exception as e:
        LOG.exception("create_summaries_and_check failed")
        return {"success": False, "summaries_created": 0, "empty_count": 0, "created_count": 0, "counts_match": False, "error": str(e)}

def query_vector_result(query: str, n_results: int = 5) -> dict:
    try:
        res = query_data(query, n_results=n_results, query_type="Title")
        if res is None:
            return {"success": False, "results": []}
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        results = [{"document": d, "metadata": m} for d, m in zip(docs, metas)]
        return {"success": True, "results": results}
    except Exception as e:
        LOG.exception("query_vector_result failed")
        return {"success": False, "results": [], "error": str(e)}

def vector_insert_result(ids: list[str], metadata: dict) -> dict:
    try:
        if not ids:
            return {"success": False}
        doc = metadata.get("document", metadata.get("content", ""))
        meta = {"Author": metadata.get("Author", ""), "Title": metadata.get("Title", ""), "Summary": metadata.get("Summary", ""), "query_type": "Title"}
        Collection.add(ids=ids, metadatas=[meta] * len(ids), documents=[doc] * len(ids))
        return {"success": True}
    except Exception as e:
        LOG.exception("vector_insert_result failed")
        return {"success": False, "ids": ids, "metadata": metadata, "error": str(e)}

def vector_update_result(ids: list[str], metadata: dict) -> dict:
    try:
        new_data = metadata.get("document", metadata.get("content", ""))
        for id in ids:
            update_data(id, new_data)
        return {"success": True}
    except Exception as e:
        LOG.exception("vector_store_update failed")
        return {"success": False, "ids": ids, "metadata": metadata, "error": str(e)}

def vector_delete_result(ids: list[str]) -> dict:
    try:
        for id in ids:
            delete_data(id)
        return {"success": True}
    except Exception as e:
        LOG.exception("vector_delete_result failed")
        return {"success": False, "ids": ids, "error": str(e)}

def generate_story_result(query: str, save: bool, n_results: int, mode: Gen_mode) -> dict:
    try:
        gen_params = generative_ai.get_generation_params(mode)
        content = generative_ai.generate_full_story(query, n_results=n_results, mode=mode)
        if not content:
            return {"success": False, "content": "", "saved": False, "saved_path": None, "timestamp": ""}
        saved_path = generative_ai.save_generated_story(content) if save else None
        return {"success": True, "content": content, "saved": save and saved_path is not None, "saved_path": str(saved_path) if saved_path else None, "timestamp": datetime.now().isoformat(), "mode": mode.value, "gen_params": gen_params}
    except Exception as e:
        LOG.exception("generate_story failed")
        return {"success": False, "content": "", "saved": False, "saved_path": None, "timestamp": ""}

def generate_summary_result(story_path: str, base_path: str, summary_output_dir: str) -> dict:
    try:
        path = Path(story_path)
        if not path.exists():
            return {"success": False, "summary": "", "saved": False, "saved_path": None, "timestamp": datetime.now().isoformat()}
        story_content = path.read_text(encoding="utf-8")
        if not story_content.strip():
            return {"success": False, "summary": "", "saved": False, "saved_path": None, "timestamp": datetime.now().isoformat()}
        stem = "summary_tmp"
        tmpdir = tempfile.mkdtemp()
        try:
            data = {
                "_stem": stem,
                "ids": f"id_{stem}",
                "metadatas": {"Author": "", "Title": "", "Summary": ""},
                "chapters": {"Chapter_name": "", "Chapter_number": 0, "Is_series": False},
                "documents": story_content,
            }
            json_path = Path(tmpdir) / f"{stem}.json"
            json_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
            base = Path(tmpdir).parent
            out_name = Path(tmpdir).name
            ok, _ = create_summaries_for_merged_data(
                str(base), out_name,
                data_merge.Gemini_summary_model, data_merge.Gemini_api_key,
                data_merge.prompt_template, data_merge.refine_prompt,
            )
            if not ok:
                return {"success": False, "summary": "", "saved": False, "saved_path": None, "timestamp": datetime.now().isoformat()}
            data2 = json.loads(json_path.read_text(encoding="utf-8"))
            summary = (data2.get("metadatas") or {}).get("Summary", "")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if not summary:
            return {"success": False, "summary": "", "saved": False, "saved_path": None, "timestamp": datetime.now().isoformat()}
        out_dir = Path(base_path) / summary_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{datetime.now().strftime('%Y%m%d')}_summary_{path.stem}.txt"
        out_path.write_text(summary, encoding="utf-8")
        return {"success": True, "summary": summary, "saved": True, "saved_path": str(out_path), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        LOG.exception("generate_summary failed")
        return {"success": False, "summary": "", "saved": False, "saved_path": None, "timestamp": datetime.now().isoformat()}              

def _eval_to_dict(data: Optional[dict]) -> dict:
    if not data:
        return EVAL_FAIL
    return {"success": True, "average_score": data.get("average_score", 0.0), "scores": data.get("scores", {}), "summary": data.get("summary", ""), "suggestions": data.get("suggestions", []), "metadata": data.get("metadata", {}), "conclusion": data.get("conclusion", "")}

def evaluate_story_text_result(story_text: str) -> dict:
    try:
        model = evaluate_model()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(story_text)
            path = f.name
        try:
            data = evaluate_generated_story(model, path)
            return _eval_to_dict(data)
        finally:
            Path(path).unlink(missing_ok=True)
    except Exception as e:
        LOG.exception("evaluate_story failed")
        return {**EVAL_FAIL, "conclusion": str(e)}

def evaluate_story_file_result(story_path: str, save: bool) -> dict:
    try:
        model = evaluate_model()
        data = evaluate_generated_story(model, story_path)
        if not data:
            return EVAL_FAIL
        if save:
            save_evaluation_results(data, result_type="story", source_stem=Path(story_path).stem)
        return _eval_to_dict(data)
    except Exception as e:
        LOG.exception("evaluate_story_file failed")
        return {**EVAL_FAIL, "conclusion": str(e)}

def evaluate_summary_result(summary_path: str, story_path: Optional[str], save: bool) -> dict:
    try:
        model = evaluate_model()
        data = evaluate_generated_summary(model, summary_path, story_path=story_path)
        if not data:
            return EVAL_FAIL
        if save:
            save_evaluation_results(data, result_type="summary", source_stem=Path(summary_path).stem)
        return _eval_to_dict(data)
    except Exception as e:
        LOG.exception("evaluate_summary failed")
        return {**EVAL_FAIL, "conclusion": str(e)}