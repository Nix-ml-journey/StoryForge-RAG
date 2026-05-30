from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from storyforge.config.config import load_config

LOG = logging.getLogger(__name__)


def _maybe_tqdm(it, **kwargs):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(it, **kwargs)
    except Exception:
        return it


def receive_book(google_book_api_key: str, user_input: str) -> dict[str, Any] | None:
    encoded_input = quote(user_input)
    url = f"https://www.googleapis.com/books/v1/volumes?q=title:{encoded_input}&key={google_book_api_key}&maxResults=20"

    try:
        response = requests.get(url, timeout=10)
        LOG.info("Google Books status=%s", response.status_code)

        if response.status_code == 200:
            try:
                data = response.json()
                if "error" in data:
                    LOG.error("Google Books API error: %s", data.get("error", {}).get("message", "Unknown error"))
                    return None
                return data
            except json.JSONDecodeError:
                LOG.error("Invalid JSON response from Google Books")
                return None
        elif response.status_code == 401 or response.status_code == 403:
            LOG.error("Google Books API key is invalid/unauthorized")
            return None
        else:
            LOG.error("Google Books request failed: status=%s", response.status_code)
            return None

    except requests.exceptions.RequestException as e:
        LOG.error("Google Books network error: %s", str(e))
        return None


def extract_books_info(search_results: dict[str, Any]) -> list[dict[str, Any]]:
    if not search_results or "items" not in search_results:
        return []

    cfg = load_config()
    book_fields = cfg.get("Book_data") or ["title", "authors", "description"]
    books = []
    for item in search_results["items"]:
        volume_info = item.get("volumeInfo", {})
        access_info = item.get("accessInfo", {})

        current_book = {key: volume_info.get(key, None) for key in book_fields}
        current_book["id"] = item.get("id")
        current_book["selfLink"] = item.get("selfLink")
        current_book["Links_to_access"] = {}
        current_book["accessInfo"] = access_info

        current_book["webReaderLink"] = access_info.get("webReaderLink")
        current_book["viewability"] = access_info.get("viewability")
        current_book["publicDomain"] = access_info.get("publicDomain")
        current_book["previewLink"] = volume_info.get("previewLink")
        current_book["infoLink"] = volume_info.get("infoLink")

        identifiers = volume_info.get("industryIdentifiers", [])
        for identifier in identifiers:
            if identifier.get("type") == "ISBN_13":
                current_book["isbn"] = identifier.get("identifier")
                break
            elif identifier.get("type") == "ISBN_10" and not current_book.get("isbn"):
                current_book["isbn"] = identifier.get("identifier")
                break

        for fmt in ["pdf", "epub"]:
            download_link = access_info.get(fmt, {}).get("downloadLink")
            if download_link:
                current_book["Links_to_access"][fmt] = download_link

        books.append(current_book)

    return books


def _identifier_already_downloaded(output_books: Path, identifier: str) -> bool:
    if not identifier or not output_books.exists():
        return False
    prefix = f"archive__{identifier}__"
    for f in output_books.iterdir():
        if f.is_file() and f.name.startswith(prefix):
            return True
    return False


def search_in_archive(book_info: dict[str, Any], archive_url: str) -> list[dict[str, Any]] | None:

    title = str(book_info.get("title") or "Unknown")
    authors = book_info.get("authors") or []

    author_str = None
    if isinstance(authors, list) and len(authors) > 0:
        author_str = authors[0]
    elif isinstance(authors, str):
        author_str = authors

    search_query = f'title:"{title}" AND mediatype:texts'

    params = {
        "q": search_query,
        "fl": "identifier,title,creator,downloads",
        "sort": "downloads desc",
        "rows": 10,
        "output": "json",
    }

    try:
        LOG.info("Searching archive.org: %s", title)
        response = requests.get(archive_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        docs = data.get("response", {}).get("docs", [])
        num_found = data.get("response", {}).get("numFound", 0)

        if docs:
            return docs

        if author_str:
            search_query = f'title:"{title}" AND creator:"{author_str}" AND mediatype:texts'
            params["q"] = search_query

            response = requests.get(archive_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            docs = data.get("response", {}).get("docs", [])
            num_found = data.get("response", {}).get("numFound", 0)

            return docs or None
        return None

    except Exception as e:
        LOG.error("archive.org search failed: %s", str(e))
        return None


_FORMAT_MAP = {
    "pdf": ([".pdf"], ["PDF"]),
    "epub": ([".epub"], ["EPUB"]),
}


def _find_file_for_formats(files, formats):
    for fmt in formats:
        fmt_info = _FORMAT_MAP.get(fmt.lower())
        for file_info in files:
            name = file_info.get("name", "")
            ftype = file_info.get("format", "")
            if fmt_info:
                exts, types = fmt_info
                if any(name.lower().endswith(e) for e in exts) or ftype in types:
                    return name
            elif name.lower().endswith(f".{fmt.lower()}"):
                return name
    return None


def download_archive_book(docs, output_dir, formats=None):
    cfg = load_config()
    if formats is None:
        formats = cfg.get("Download_formats") or ["pdf", "epub"]

    if not docs or len(docs) == 0:
        logging.error("No archive search results found")
        return None

    archive_book = docs[0]
    identifier = archive_book.get("identifier")
    title = archive_book.get("title", "Unknown")

    if not identifier:
        logging.error("No identifier found for the archive book")
        return None

    LOG.info("Downloading archive.org item: %s - %s", identifier, title)

    try:
        archive_metadata_url = str(cfg.get("Archive_metadata_url") or "")
        metadata_url = f"{archive_metadata_url}{identifier}"
        metadata_response = requests.get(metadata_url, timeout=10)
        metadata_response.raise_for_status()

        metadata_data = metadata_response.json()
        files = metadata_data.get("files", [])

        matched_file = _find_file_for_formats(files, formats)

        if not matched_file:
            logging.warning(f"No file in formats {formats} for identifier: {identifier}")
            return None

        LOG.info("Found file: %s", matched_file)

        download_url = f"https://archive.org/download/{identifier}/{matched_file}"

        LOG.info("Downloading: %s", download_url)
        response = requests.get(download_url, timeout=60, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        safe_title = re.sub(r'[<>:"/\\|?*]', "_", title)
        safe_title = safe_title.replace(" ", "_")
        safe_title = safe_title[:100]

        ext = Path(matched_file).suffix or f".{formats[0]}"
        filename = f"archive__{identifier}__{safe_title}{ext}"
        file_path = output_path / filename

        with open(file_path, "wb") as f:
            pbar = None
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=filename, ncols=100, unit_divisor=1024)
            except Exception:
                pbar = None
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024 * 10):
                    if chunk:
                        f.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))
            finally:
                if pbar:
                    pbar.close()

        file_size = file_path.stat().st_size
        LOG.info("Saved %s (%s bytes)", filename, file_size)

        return file_path

    except requests.exceptions.RequestException as e:
        LOG.error("Network error downloading from Archive.org: %s", str(e))
        return None
    except json.JSONDecodeError:
        LOG.error("Invalid JSON response from Archive.org metadata API")
        return None
    except Exception as e:
        LOG.error("Error downloading from Archive.org: %s", str(e))
        return None


def download_archive_book_and_save_meta(
    *,
    base_path: str | Path | None,
    query: str,
    formats: list[str],
    raw_dir: str,
    meta_dir: str,
    archive_url: str,
) -> dict[str, Any]:
    """
    Convenience helper compatible with existing orchestrator:
    searches archive.org then downloads one file + writes a small metadata json.
    """
    cfg = load_config()
    base = Path(base_path or cfg.get("BASE_PATH") or ".").resolve()
    output_books = (base / raw_dir).resolve()
    output_meta = (base / meta_dir).resolve()

    docs = search_in_archive({"title": query, "authors": []}, archive_url) or []
    if not docs:
        return {"success": False, "saved": False, "saved_path": None}

    downloaded = None
    identifier = ""
    # Try results in order, showing progress when tqdm is available.
    for doc in _maybe_tqdm(docs, desc="Archive download candidates", unit="item"):
        identifier = str((doc or {}).get("identifier") or "").strip()
        if not identifier:
            continue
        if _identifier_already_downloaded(output_books, identifier):
            continue
        downloaded = download_archive_book([doc], output_books, formats=formats)
        if downloaded:
            break

    if not downloaded:
        return {"success": False, "saved": False, "saved_path": None}

    output_meta.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_file = output_meta / f"archive_books_{ts}.json"
    meta_file.write_text(
        json.dumps(
            {
                "search_query": query,
                "source": "archive.org",
                "timestamp": ts,
                "books": [{"title": query, "authors": [], "identifier": identifier, "file_path": str(downloaded)}],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"success": True, "saved": True, "saved_path": str(meta_file)}


if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be used via the FastAPI/orchestrator wrappers, not as an interactive CLI.\n"
        "Use the API endpoints or Orchestrator methods to search/download, then run extraction.\n"
        "After extraction, manually clean/split texts into data/stories/, then run Step 1."
    )
