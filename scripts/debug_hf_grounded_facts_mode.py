"""
Probe Step-2 Hugging Face grounded-facts extraction (extraction.py).

Calls the real `storyforge.rag.extraction._hf_chat_extract_json()` -- the
exact function `extract_grounded_facts()` uses in production -- through a
thin spy around InferenceClient that records the outgoing request kwargs and
the raw response object, so this probe can never silently drift out of sync
with what production actually sends (as an earlier version of this script
did: it hand-built its own request dict and never included `extra_body`,
so it looked like the enable_thinking fix wasn't taking effect when in fact
the probe just wasn't using it).

Usage:
  py scripts/debug_hf_grounded_facts_mode.py
  py scripts/debug_hf_grounded_facts_mode.py --query "A warrior monk in the desert"
  py scripts/debug_hf_grounded_facts_mode.py --disable-json-mode
  py scripts/debug_hf_grounded_facts_mode.py --force-enable-thinking   # compare against thinking left on
  py scripts/debug_hf_grounded_facts_mode.py --show-raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyforge.config.config import load_config  # noqa: E402
from storyforge.rag.attribution import parse_grounded_facts_json  # noqa: E402


def _sample_chunks() -> str:
    return (
        "[CHUNK demo_1 | The Warrior Monk]\n"
        "Amun trained with the monks in Kongshan Temple after crossing the desert.\n\n"
        "[CHUNK demo_2 | The Warrior Monk]\n"
        "He returned to defend Eldoria when raiders threatened the northern gate."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug HF grounded-facts extraction by calling the real extraction.py code path."
    )
    parser.add_argument("--query", default="A warrior monk who returns to defend Eldoria", help="Probe query.")
    parser.add_argument("--model", default="", help="Optional HF model override (HF_grounded_facts_model).")
    parser.add_argument("--disable-json-mode", action="store_true", help="Force HF_grounded_facts_json_mode off.")
    parser.add_argument(
        "--force-enable-thinking",
        action="store_true",
        help="Force HF_grounded_facts_disable_thinking off, to compare against the default (thinking disabled).",
    )
    parser.add_argument("--show-raw", action="store_true", help="Print full raw model response / reasoning trace.")
    args = parser.parse_args()

    cfg = dict(load_config())
    if args.model:
        cfg["HF_grounded_facts_model"] = args.model
    if args.disable_json_mode:
        cfg["HF_grounded_facts_json_mode"] = False
    if args.force_enable_thinking:
        cfg["HF_grounded_facts_disable_thinking"] = False

    if not (cfg.get("facehugging_api") or "").strip():
        print("ERROR: Missing Hugging Face token in setup/env (facehugging_api).")
        return 1

    from storyforge.rag.extraction import _get_generation_prompts

    prompts = _get_generation_prompts()
    system = prompts["facts_system"]
    user = prompts["facts_user"].format(query=args.query, retrieval_chunks=_sample_chunks())

    # Spy around InferenceClient: forwards every call to a real client so the
    # HTTP request and response are genuine, but records the outgoing kwargs
    # and the response object for inspection. This runs through the actual
    # extraction._hf_chat_extract_json() code (model selection, json_mode
    # logic, disable_thinking logic, TypeError retry) -- not a reimplementation.
    from huggingface_hub import InferenceClient as _RealInferenceClient

    captured_requests: list[dict] = []
    captured_responses: list[object] = []

    class _SpyInferenceClient:
        def __init__(self, token=None, **kw):
            self._real = _RealInferenceClient(token=token, **kw)

        def chat_completion(self, **kwargs):
            captured_requests.append(dict(kwargs))
            resp = self._real.chat_completion(**kwargs)
            captured_responses.append(resp)
            return resp

    from storyforge.rag.extraction import _hf_chat_extract_json

    with patch("storyforge.rag.extraction.InferenceClient", _SpyInferenceClient):
        raw = _hf_chat_extract_json(cfg=cfg, system=system, user=user)

    parsed = parse_grounded_facts_json(raw)

    # Diagnostics from the final (last) request/response actually sent.
    last_request = captured_requests[-1] if captured_requests else {}
    last_resp = captured_responses[-1] if captured_responses else None
    finish_reason = None
    reasoning_content = None
    usage = None
    if last_resp is not None:
        try:
            choice = last_resp.choices[0]  # type: ignore[attr-defined]
            finish_reason = getattr(choice, "finish_reason", None)
            msg = getattr(choice, "message", None)
            reasoning_content = getattr(msg, "reasoning_content", None) if msg is not None else None
            usage = getattr(last_resp, "usage", None)
        except Exception:
            pass

    print(f"MODEL: {cfg.get('HF_grounded_facts_model')}")
    print(f"CALLS_MADE: {len(captured_requests)} (>1 means a TypeError retry happened -- see extraction.py)")
    print(f"REQUEST_HAD_RESPONSE_FORMAT: {'response_format' in last_request}")
    print(f"REQUEST_HAD_EXTRA_BODY: {last_request.get('extra_body')}")
    print(f"MAX_TOKENS_REQUESTED: {last_request.get('max_tokens')}")
    print(f"FINISH_REASON: {finish_reason}")
    print(f"USAGE: {usage}")
    print(f"FACTS_PARSED: {len(parsed.facts)}")
    print("RAW_PREVIEW:")
    print((raw or "").strip()[:600] or "<empty>")
    if reasoning_content:
        print("\nREASONING_CONTENT_PREVIEW (hidden thinking tokens -- not the JSON answer):")
        print(str(reasoning_content)[:600])
    if args.show_raw:
        print("\nRAW_FULL:")
        print(raw)
        if reasoning_content:
            print("\nREASONING_CONTENT_FULL:")
            print(reasoning_content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
