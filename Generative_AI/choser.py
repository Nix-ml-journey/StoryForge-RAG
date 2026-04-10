"""
Structure picker: chooses a narrative flow (structural shift) after Layer 1 and before Layer 2.
Layer 2 then maps 5W1H into exactly that order so the model only does mapping, not structure choice.

This module also filters which structures are allowed per generation mode:
- FAST  → only linear-friendly structures (easy for single-pass mapping).
- THINKING → all structures, including non-linear/reverse/obituary styles that
  require stronger causal reasoning.
"""
import random
import yaml
from pathlib import Path
from typing import Any, Optional, Union

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT_DIR = Path(__file__).resolve().parent.parent
FLOW_STRUCTURE_FILE = ROOT_DIR / "flow_structure.yaml"

_CACHED_STRUCTURES: Optional[list[dict[str, Any]]] = None


def load_flow_structures(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load flow structures from YAML. Returns list of {name, flow, sections, style}."""
    global _CACHED_STRUCTURES
    if _CACHED_STRUCTURES is not None:
        return _CACHED_STRUCTURES
    p = path or FLOW_STRUCTURE_FILE
    if not p.exists():
        logging.warning("flow_structure.yaml not found at %s; using default linear structure only.", p)
        _CACHED_STRUCTURES = [_default_linear_structure()]
        return _CACHED_STRUCTURES
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "flow_structures" in data:
        structures = data["flow_structures"]
    elif isinstance(data, list):
        structures = data
    else:
        logging.warning("flow_structure.yaml has unexpected format; using default linear structure only.")
        _CACHED_STRUCTURES = [_default_linear_structure()]
        return _CACHED_STRUCTURES
    if not structures or not isinstance(structures, list):
        _CACHED_STRUCTURES = [_default_linear_structure()]
        return _CACHED_STRUCTURES
    valid = []
    for s in structures:
        if not isinstance(s, dict):
            continue
        if "sections" in s and len(s["sections"]) == 5 and "flow" in s:
            valid.append(s)
        else:
            logging.warning("Skipping invalid structure entry: missing flow or 5 sections.")
    if not valid:
        _CACHED_STRUCTURES = [_default_linear_structure()]
        return _CACHED_STRUCTURES
    _CACHED_STRUCTURES = valid
    return _CACHED_STRUCTURES


def get_default_structure() -> dict[str, Any]:
    return _default_linear_structure()


def _default_linear_structure() -> dict[str, Any]:
    return {
        "name": "The Standard (Linear)",
        "flow": "1-2-3-4-5",
        "sections": [
            "WHO, WHERE, WHEN (The Setup)",
            "WHAT (The Problem Starts)",
            "TWIST/COMPLICATION (The Challenge)",
            "HOW (The Big Action/Climax)",
            "WHY/OUTCOME (The Moral and Conclusion)",
        ],
        "style": "Clear, logical, and easy to follow.",
    }


def _filter_structures_for_mode(
    structures: list[dict[str, Any]],
    mode: Optional[Union[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Restrict non-linear / reverse-causal structures in FAST mode.

    - FAST mode: prefer a small, linear-friendly subset. If none found, fall back
      to all structures except explicitly non-linear ones.
    - THINKING mode or unknown: allow all structures.

    Mode can be:
    - Gen_mode enum instance (value like "FAST"/"THINKING")
    - plain string "fast"/"thinking"/"FAST"/"THINKING"
    """
    if not structures or mode is None:
        return structures

    # Normalize mode to a simple lowercase string ("fast" / "thinking" / other)
    mode_str = ""
    try:
        # Gen_mode is a str Enum; .value already contains the config string.
        val = getattr(mode, "value", mode)
        mode_str = str(val).strip().lower()
    except Exception:
        mode_str = str(mode).strip().lower()

    is_restricted = "fast" in mode_str or "short" in mode_str
    if not mode_str or not is_restricted:
        # THINKING or anything non-fast/non-short → allow all structures.
        return structures

    # Names of structures that are safe for FAST mode (linear / straightforward arcs).
    fast_allowed_names = {
        "The Standard (Linear)",
        "The Action Hook (In Media Res)",
        "The Sinking Ship (Rising Tension)",
        "The Heist (The Plan)",
        "The Impact (Outcome Focused)",
        "The Slow Burn (Delayed Conflict)",
    }

    # Names of explicitly complex/non-linear structures that should never be
    # used in FAST mode. Some of these may not exist yet but are reserved.
    nonlinear_names = {
        "Aftermath Unspooling",
        "The Mystery (Reverse)",
        "The Chaos Start (Disoriented)",
        "Obituary",
        "Verdict First",
    }

    # First pass: prefer the explicitly whitelisted FAST structures.
    fast_candidates = [s for s in structures if s.get("name") in fast_allowed_names]
    if fast_candidates:
        return fast_candidates

    # Fallback: allow everything except known non-linear structures.
    linearish = [s for s in structures if s.get("name") not in nonlinear_names]
    return linearish or structures


def choose_flow_structure(
    flow_structures: Optional[list[dict[str, Any]]] = None,
    path: Optional[Path] = None,
    mode: Optional[Union[str, Any]] = None,
) -> dict[str, Any]:
    """
    Pick one structural shift at random. Used after Layer 1 and before Layer 2.
    Returns a dict with: name, flow, sections (5 header strings in narrative order), style.
    """
    structures = flow_structures if flow_structures is not None else load_flow_structures(path)
    structures = _filter_structures_for_mode(structures, mode=mode)
    if not structures:
        return _default_linear_structure()
    return random.choice(structures)


def build_section_headers_and_mapping(structure: dict[str, Any]) -> tuple[str, str]:
    """
    Build the two strings to inject into Layer 2 prompts:
    - section_headers: one header per line, [SECTION 1: ...], [SECTION 2: ...], ...
    - section_mapping: short mapping lines for the system prompt.
    """
    sections = structure.get("sections") or []
    if len(sections) != 5:
        default = _default_linear_structure()
        sections = default["sections"]
    headers = "\n".join(f"[SECTION {i + 1}: {s}]" for i, s in enumerate(sections))
    mapping = "\n".join(
        f"- Section {i + 1}: Map 5W1H into {s}."
        for i, s in enumerate(sections)
    )
    return headers, mapping


if __name__ == "__main__":
    load_flow_structures()
    chosen = choose_flow_structure()
    print("Chosen:", chosen.get("name"), "| Flow:", chosen.get("flow"))
    h, m = build_section_headers_and_mapping(chosen)
    print("Headers:\n", h)
    print("Mapping:\n", m)
