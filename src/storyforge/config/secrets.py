from __future__ import annotations

import logging
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ROOT / ".env")
    load_dotenv(_ROOT / ".env.local")


def overlay_api_keys_from_env(config: dict | None) -> None:
    if not config:
        return
    try_load_dotenv()

    g = os.environ.get("STORYFORGE_GEMINI_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    if g:
        config["Gemini_api_key"] = g
        logging.debug("Gemini_api_key taken from environment (not logged).")

    b = (
        os.environ.get("STORYFORGE_GOOGLE_BOOKS_API_KEY", "").strip()
        or os.environ.get("GOOGLE_BOOKS_API_KEY", "").strip()
    )
    if b:
        config["Google_book_api_key"] = b
        logging.debug("Google_book_api_key taken from environment (not logged).")

    hf = (
        os.environ.get("STORYFORGE_HF_API_KEY", "").strip()
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
    )
    if hf:
        config["facehugging_api"] = hf
        logging.debug("facehugging_api taken from environment (not logged).")
