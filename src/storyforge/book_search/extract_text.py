from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
from epub_to_text import EpubProcessor

from storyforge.config.config import load_config

LOG = logging.getLogger(__name__)


def _iter_files(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes], key=lambda p: p.name)


def extract_text_from_pdf(path: Path) -> str:
    doc = fitz.open(path)
    try:
        parts: list[str] = []
        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                parts.append(str(b[4] or ""))
        return "\n".join(parts).strip()
    finally:
        doc.close()


def extract_text_from_epub(path: Path, *, scratch_dir: Path) -> str:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    book_name = path.stem
    output_dir = scratch_dir / book_name
    output_dir.mkdir(parents=True, exist_ok=True)

    epub_processor = EpubProcessor(str(path), str(output_dir))
    epub_processor.export_chapters_markdown()

    parts: list[str] = []
    for root, _, files in os.walk(output_dir):
        for f in sorted(files):
            if f.endswith((".md", ".txt")):
                parts.append((Path(root) / f).read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join([p.strip() for p in parts if p.strip()]).strip()


def format_extracted_text(text: str) -> str:
    """
    Light formatting to make manual cleanup easier.
    We do NOT attempt story separation — user will do that manually.
    """
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return ""

    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""

    paras: list[str] = []
    cur: list[str] = []

    for ln in lines:
        is_headingish = bool(re.match(r"^[A-Z0-9][A-Z0-9\s\-–—]{5,}$", ln))
        is_page_num = bool(re.match(r"^\d{1,4}$", ln))
        if (is_headingish or is_page_num) and cur:
            paras.append(" ".join(cur).strip())
            cur = []
        cur.append(ln)

        if ln.endswith((".", "?", "!")) and len(" ".join(cur)) >= 400:
            paras.append(" ".join(cur).strip())
            cur = []

    if cur:
        paras.append(" ".join(cur).strip())

    out = "\n\n".join([p for p in paras if p]).strip()
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def extract_text_from_books(
    *,
    base_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    formats: Iterable[str] = ("pdf", "epub"),
) -> dict[str, Any]:
    """
    Extract all supported books in the downloaded folder into plain `.txt`.

    After this step, you manually clean/split and move final `.txt` stories into `data/stories/`.
    """
    cfg = load_config()
    base = Path(base_path or cfg.get("BASE_PATH") or ".").resolve()
    in_dir = Path(input_dir) if input_dir else (base / (cfg.get("Downloaded_rawbook_dir") or "Downloaded_RawBook"))
    out_dir = Path(output_dir) if output_dir else (base / (cfg.get("Raw_extracted_dir") or cfg.get("Extracted_text_dir") or "Raw_Extracted"))
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allow = {f".{str(x).lower().lstrip('.')}" for x in formats}
    pdfs = _iter_files(in_dir, (".pdf",)) if ".pdf" in allow else []
    epubs = _iter_files(in_dir, (".epub",)) if ".epub" in allow else []

    extracted = 0
    skipped = 0
    errors: list[str] = []

    def _maybe_tqdm(it, **kwargs):
        try:
            from tqdm import tqdm  # type: ignore

            return tqdm(it, **kwargs)
        except Exception:
            return it

    for p in _maybe_tqdm(pdfs, desc="Extract PDF", unit="file"):
        out_path = out_dir / f"{p.stem}.txt"
        try:
            text = extract_text_from_pdf(p)
            text = format_extracted_text(text)
            out_path.write_text(text + "\n", encoding="utf-8")
            extracted += 1
        except Exception as e:
            errors.append(f"{p.name}: {type(e).__name__}: {e}")
            skipped += 1

    scratch = out_dir / "_epub_scratch"
    for p in _maybe_tqdm(epubs, desc="Extract EPUB", unit="file"):
        out_path = out_dir / f"{p.stem}.txt"
        try:
            text = extract_text_from_epub(p, scratch_dir=scratch)
            text = format_extracted_text(text)
            out_path.write_text(text + "\n", encoding="utf-8")
            extracted += 1
        except Exception as e:
            errors.append(f"{p.name}: {type(e).__name__}: {e}")
            skipped += 1

    return {
        "success": True,
        "input_folder": str(in_dir),
        "output_folder": str(out_dir),
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
    }
