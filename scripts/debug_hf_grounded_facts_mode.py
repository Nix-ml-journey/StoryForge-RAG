"""
Probe Step-2 Hugging Face grounded-facts extraction mode.

This script checks whether HF grounded-facts extraction can use strict JSON mode
(`response_format={"type":"json_object"}`) or must fall back to a normal chat call.

Usage:
  py scripts/debug_hf_grounded_facts_mode.py
  py scripts/debug_hf_grounded_facts_mode.py --query "A warrior monk in the desert"
  py scripts/debug_hf_grounded_facts_mode.py --disable-json-mode
  py scripts/debug_hf_grounded_facts_mode.py --show-raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import InferenceClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyforge.config.config import load_config, load_prompts  # noqa: E402
from storyforge.rag.attribution import parse_grounded_facts_json  # noqa: E402


def _hf_token(cfg: dict) -> str:
    import os

    return (
        str(cfg.get("facehugging_api") or "").strip()
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_API_KEY", "").strip()
    )


def _extract_message_text(resp) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()  # type: ignore[attr-defined]
    except Exception:
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg.get("content") or "").strip()
        return str(resp).strip()


def _build_prompt(query: str) -> tuple[str, str]:
    prompts = (load_prompts() or {}).get("generation") or {}
    system = str(prompts.get("grounded_facts_system") or "Return grounded facts as JSON.")
    user_template = str(
        prompts.get("grounded_facts_user")
        or "QUERY:\n{query}\n\nCHUNKS:\n{retrieval_chunks}\n\nReturn JSON facts."
    )
    sample_chunks = (
        "[CHUNK demo_1 | The Warrior Monk]\n"
        "Amun trained with the monks in Kongshan Temple after crossing the desert.\n\n"
        "[CHUNK demo_2 | The Warrior Monk]\n"
        "He returned to defend Eldoria when raiders threatened the northern gate."
    )
    user = user_template.format(query=query, retrieval_chunks=sample_chunks)
    return system, user


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug HF grounded-facts JSON mode/fallback behavior.")
    parser.add_argument("--query", default="A warrior monk who returns to defend Eldoria", help="Probe query.")
    parser.add_argument("--model", default="", help="Optional HF model override.")
    parser.add_argument("--disable-json-mode", action="store_true", help="Force no response_format.")
    parser.add_argument("--show-raw", action="store_true", help="Print full raw model response.")
    args = parser.parse_args()

    cfg = load_config()
    token = _hf_token(cfg)
    if not token:
        print("ERROR: Missing Hugging Face token in setup/env.")
        return 1

    model_id = args.model or cfg.get("HF_grounded_facts_model") or cfg.get("HF_evaluation_model") or "Qwen/Qwen2.5-1.5B-Instruct"
    max_new = int(cfg.get("HF_grounded_facts_max_new_tokens") or 300)
    temperature = float(cfg.get("HF_grounded_facts_temperature") or 0.1)
    config_json_mode = str(cfg.get("HF_grounded_facts_json_mode") or "true").strip().lower() not in ("false", "0", "no")
    json_mode = config_json_mode and (not args.disable_json_mode)

    system, user = _build_prompt(args.query)
    client = InferenceClient(token=token)
    request = {
        "model": str(model_id),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_new,
        "temperature": temperature,
    }

    mode_used = "disabled"
    if json_mode:
        request["response_format"] = {"type": "json_object"}
        mode_used = "json_mode"

    try:
        resp = client.chat_completion(**request)
    except TypeError:
        if "response_format" not in request:
            raise
        request.pop("response_format", None)
        resp = client.chat_completion(**request)
        mode_used = "fallback_no_response_format"

    raw = _extract_message_text(resp)
    parsed = parse_grounded_facts_json(raw)

    print(f"MODEL: {model_id}")
    print(f"JSON_MODE_CONFIG: {config_json_mode}")
    print(f"MODE_USED: {mode_used}")
    print(f"FACTS_PARSED: {len(parsed.facts)}")
    print("RAW_PREVIEW:")
    print((raw or "").strip()[:600] or "<empty>")
    if args.show_raw:
        print("\nRAW_FULL:")
        print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
