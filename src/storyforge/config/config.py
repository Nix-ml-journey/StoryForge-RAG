from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import os
import yaml

from storyforge.config.secrets import overlay_api_keys_from_env

# Project root (repo top level): src/storyforge/config → three parents up.
ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_FILE = ROOT_DIR / "setup.yaml"
EXAMPLE_CONFIG_FILE = ROOT_DIR / "setup.example.yaml"
PROMPTS_FILE = ROOT_DIR / "prompts.yaml"


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    env_path = (os.environ.get("STORYFORGE_CONFIG_PATH") or "").strip()
    if env_path:
        return Path(env_path)
    if config_path:
        return Path(config_path)
    if CONFIG_FILE.exists():
        return CONFIG_FILE
    return EXAMPLE_CONFIG_FILE


def _normalize_base_path(config: dict[str, Any], source_path: Path) -> None:
    base_path = str(config.get("BASE_PATH") or "").strip()
    is_placeholder = "path/to/your/project" in base_path.replace("\\", "/")
    # Blank or placeholder BASE_PATH → use repo root.
    if (not base_path) or is_placeholder:
        config["BASE_PATH"] = str(ROOT_DIR)


@lru_cache(maxsize=8)
def _load_config_cached(config_path_key: str, overlay_keys: bool) -> dict[str, Any]:
    path = resolve_config_path(config_path_key or None)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file. Expected {CONFIG_FILE} or {EXAMPLE_CONFIG_FILE}.")
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    _normalize_base_path(config, path)
    if overlay_keys:
        overlay_api_keys_from_env(config)
    return config


def load_config(config_path: str | Path | None = None, *, overlay_keys: bool = True) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    return deepcopy(_load_config_cached(str(path), overlay_keys))


@lru_cache(maxsize=1)
def _load_prompts_cached() -> dict[str, Any]:
    if not PROMPTS_FILE.exists():
        raise FileNotFoundError(f"Missing prompts file: {PROMPTS_FILE}")
    with open(PROMPTS_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_prompts() -> dict[str, Any]:
    return deepcopy(_load_prompts_cached())
