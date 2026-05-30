import json
import logging
import shutil
import tempfile
from storyforge.data.step1_prepare_and_enrich import run_step1_prepare_and_enrich
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from storyforge.book_search.extract_text import extract_text_from_books
from storyforge.book_search.fetch_book import download_archive_book, extract_books_info, receive_book, search_in_archive
from storyforge.evaluation.evaluation import (
    evaluate_generated_story,
    evaluate_generated_summary,
    evaluate_model,
    save_evaluation_results,
)
from storyforge.rag import generative_ai
from storyforge.rag.generative_ai import Gen_mode, StoryType
from storyforge.rag.agentic_loop import run_agentic_story_loop
from storyforge.rag.langchain_rag import generate_story_3step_langchain
from storyforge.vector_store.chromadb import Collection, delete_data, query_data, update_data
from storyforge.vector_store.chromadb import reset_vector_store_dir, set_active_collection
from storyforge.vector_store.ingest_stories import ingest_stories_dir
"""
Orchestrator helpers: books, ingest, 3-step RAG, and agentic generation.

Pipeline: story JSON → Chroma ingest → retrieve / extract / generate (or agentic loop).
"""

LOG = logging.getLogger(__name__)
FAIL = {"success": False, "results": [], "metadata": {}, "urls": []}
FAIL_SAVE = {"success": False, "saved": False, "saved_path": None}
EVAL_FAIL = {
    "success": False,
    "evaluation_type": "list[str]",
    "average_score": 0.0,
    "scores": {},
    "summary": "",
    "suggestions": [],
    "metadata": {},
    "conclusion": "",
}


def search_books(api_key: str, query: str, n_results: int = 20) -> dict:
    try:
        r = receive_book(api_key, query)
        if not r:
            return FAIL
        books = extract_books_info(r)[:n_results]
        results = [{"id": b.get("id"), "title": b.get("title"), "authors": b.get("authors")} for b in books]
        urls = [
            {"url": (b.get("selfLink") or b.get("infoLink")), "source": "Google Book"}
            for b in books
            if b.get("selfLink") or b.get("infoLink")
        ]
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


def download_book_archive(
    base_path: str, query: str, formats: list, raw_dir: str, meta_dir: str, archive_url: str
) -> dict:
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
            "books": [
                {
                    "title": d.get("title", query),
                    "authors": [creator] if creator else [],
                    "identifier": d.get("identifier"),
                    "filename": txt_filename,
                    "file_path": str(downloaded_file),
                }
            ],
        }
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"success": True, "saved": True, "saved_path": str(meta_file)}
    except Exception as e:
        LOG.exception("download_book_archive failed")
        return FAIL_SAVE


def extract_text_result() -> dict:
    try:
        extract_text_from_books()
        return {"success": True, "message": "Text extraction completed"}
    except Exception as e:
        LOG.exception("extract_text failed")
        return {"success": False, "message": str(e), "error": str(e)}


def reset_vector_store_result(*, new_collection_name: str = "StoryForgeRag_v1") -> dict:
    res = reset_vector_store_dir(new_collection_name=new_collection_name)
    return res


def ingest_stories_result(*, collection_name: str = "StoryForgeRag_v1") -> dict:
    try:
        set_active_collection(collection_name)
        res = ingest_stories_dir(collection_name=collection_name)
        return {
            "success": res.success,
            "files_seen": res.files_seen,
            "chunks_written": res.chunks_written,
            "collection_name": res.collection_name,
            "error": res.error,
        }
    except Exception as e:
        LOG.exception("ingest_stories_result failed")
        return {
            "success": False,
            "files_seen": 0,
            "chunks_written": 0,
            "collection_name": collection_name,
            "error": str(e),
        }


def step1_prepare_and_enrich_result(
    *,
    limit: int = 0,
    overwrite_summary: bool = False,
    overwrite_sections: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run prepare + enrich on data/story_json (see scripts/prepare_story_records.py)."""
    try:
        run_step1_prepare_and_enrich(
            root=Path(__file__).resolve().parents[3],
            limit=limit,
            overwrite_summary=overwrite_summary,
            overwrite_sections=overwrite_sections,
            dry_run=dry_run,
            enable_tqdm=True,
        )
        return {"success": True}
    except Exception as e:
        LOG.exception("step1_prepare_and_enrich_result failed")
        return {"success": False, "error": str(e)}


def query_vector_result(query: str, n_results: int = 5, query_type: str = "content") -> dict:
    try:
        res = query_data(query, n_results=n_results, query_type=query_type)
        if res is None:
            return {"success": False, "results": []}
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        results = []
        for rid, d, m in zip(ids, docs, metas):
            md = m or {}
            results.append(
                {
                    "id": rid,
                    "text": d,
                    "metadata": {
                        "series_id": md.get("series_id", ""),
                        "series_name": md.get("series_name", ""),
                        "volume_number": md.get("volume_number", 0),
                        "chapter_id": md.get("chapter_id", ""),
                        "chapter_number": md.get("chapter_number", 0),
                        "chapter_name": md.get("chapter_name", ""),
                        "section": md.get("section", ""),
                        "character": md.get("character", ""),
                    },
                }
            )
        return {"success": True, "results": results}
    except Exception as e:
        LOG.exception("query_vector_result failed")
        return {"success": False, "results": [], "error": str(e)}


def vector_insert_result(ids: list[str], metadata: dict) -> dict:
    try:
        if not ids:
            return {"success": False}
        doc = metadata.get("document", metadata.get("content", ""))
        meta = {
            "Author": metadata.get("Author", ""),
            "Title": metadata.get("Title", ""),
            "Summary": metadata.get("Summary", ""),
            "query_type": "Title",
        }
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
        failed = [id for id in ids if not delete_data(id)]
        if failed:
            return {
                "success": False,
                "ids": ids,
                "error": f"Delete failed for: {failed}. Check server logs; ids must match Chroma exactly (see GET /vector_store/ids).",
            }
        return {"success": True}
    except Exception as e:
        LOG.exception("vector_delete_result failed")
        return {"success": False, "ids": ids, "error": str(e)}


def generate_story_result(
    query: str,
    save: bool,
    n_results: int,
    mode: Gen_mode,
    story_type: Optional[StoryType] = None,
    debug: bool = False,
) -> dict:
    try:
        temperature, top_p = generative_ai.get_mode_sampling(mode)
        gen_params = {
            "temperature": temperature,
            "top_p": top_p,
            "three_layer": True,
            "story_type": story_type.value if story_type else None,
        }
        # 3-step RAG: Chroma retrieve → HF facts → local story generation
        out = generate_story_3step_langchain(
            query,
            n_stories=3,
            chunks_per_story=2,
            show_progress=True,
            debug=debug,
        )
        content = out.content
        if not content:
            return {"success": False, "content": "", "saved": False, "saved_path": None, "timestamp": ""}
        saved_path = generative_ai.save_generated_story(content) if save else None
        payload = {
            "success": True,
            "content": content,
            "saved": save and saved_path is not None,
            "saved_path": str(saved_path) if saved_path else None,
            "timestamp": datetime.now().isoformat(),
            "mode": mode.value,
            "gen_params": gen_params,
        }
        if debug:
            payload.update(
                {
                    "retrieval_context": out.retrieval_context,
                    "grounded_extraction": out.grounded_extraction,
                    "retrieval_chunks": list(out.retrieval_chunks),
                    "grounded_facts": list(out.grounded_facts),
                    "debug_attribution": out.debug_attribution or {},
                }
            )
        return payload
    except Exception as e:
        LOG.exception("generate_story failed")
        return {"success": False, "content": "", "saved": False, "saved_path": None, "timestamp": ""}


def generate_story_agentic_result(
    query: str,
    save: bool,
    mode: Gen_mode,
    story_type: Optional[StoryType] = None,
    debug: bool = False,
) -> dict:
    """
    Agentic generation: loop until accept or max iterations.

    See agentic_loop.run_agentic_story_loop. Returns content, scores, and iteration log.
    """
    try:
        temperature, top_p = generative_ai.get_mode_sampling(mode)
        result = run_agentic_story_loop(
            query,
            mode=mode,
            story_type=story_type or StoryType.MIX,
            debug=debug,
        )
        content = result.content
        if not content:
            return {"success": False, "content": "", "saved": False, "saved_path": None, "timestamp": ""}
        saved_path = generative_ai.save_generated_story(content) if save else None
        payload = {
            "success": True,
            "content": content,
            "saved": save and saved_path is not None,
            "saved_path": str(saved_path) if saved_path else None,
            "timestamp": datetime.now().isoformat(),
            "mode": mode.value,
            "accepted": result.accepted,
            "stop_reason": result.stop_reason,
            "iterations_run": len(result.iterations),
            "final_average": result.final_average,
            "iterations": result.iterations,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "agentic": True,
                "story_type": story_type.value if story_type else None,
            },
        }
        if debug:
            payload.update(
                {
                    "retrieval_context": result.retrieval_context,
                    "grounded_extraction": result.grounded_extraction,
                    "retrieval_chunks": list(result.retrieval_chunks),
                    "grounded_facts": list(result.grounded_facts),
                    "final_scores": result.final_scores,
                }
            )
        return payload
    except Exception as e:
        LOG.exception("generate_story_agentic failed")
        return {"success": False, "content": "", "saved": False, "saved_path": None, "timestamp": "", "error": str(e)}


def generate_summary_result(story_path: str, base_path: str, summary_output_dir: str) -> dict:
    try:
        path = Path(story_path)
        if not path.exists():
            return {
                "success": False,
                "summary": "",
                "saved": False,
                "saved_path": None,
                "timestamp": datetime.now().isoformat(),
            }
        story_content = path.read_text(encoding="utf-8")
        if not story_content.strip():
            return {
                "success": False,
                "summary": "",
                "saved": False,
                "saved_path": None,
                "timestamp": datetime.now().isoformat(),
            }
        from storyforge.config.config import load_config
        from storyforge.data.hf_summary import get_hf_token, summarize_long_text

        cfg = load_config()
        model = str(cfg.get("HF_summary_model") or "sshleifer/distilbart-cnn-12-6")
        token = get_hf_token(cfg)
        summary = summarize_long_text(token, model, story_content)
        if not summary:
            return {
                "success": False,
                "summary": "",
                "saved": False,
                "saved_path": None,
                "timestamp": datetime.now().isoformat(),
            }
        out_dir = Path(base_path) / summary_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{datetime.now().strftime('%Y%m%d')}_summary_{path.stem}.txt"
        out_path.write_text(summary, encoding="utf-8")
        return {
            "success": True,
            "summary": summary,
            "saved": True,
            "saved_path": str(out_path),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        LOG.exception("generate_summary failed")
        return {
            "success": False,
            "summary": "",
            "saved": False,
            "saved_path": None,
            "timestamp": datetime.now().isoformat(),
        }


def _eval_to_dict(data: Optional[dict]) -> dict:
    if not data:
        return EVAL_FAIL
    return {
        "success": True,
        "average_score": data.get("average_score", 0.0),
        "scores": data.get("scores", {}),
        "summary": data.get("summary", ""),
        "suggestions": data.get("suggestions", []),
        "metadata": data.get("metadata", {}),
        "conclusion": data.get("conclusion", ""),
    }


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
