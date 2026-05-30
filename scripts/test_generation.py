"""
test_generation.py
------------------
Manual smoke-tests for the story generation endpoint.

Usage (server must be running in another terminal: python main.py):
    python scripts/test_generation.py              # run all tests
    python scripts/test_generation.py --test 1     # run a single test by number
    python scripts/test_generation.py --debug      # show full response including grounded facts

Tests cover:
    1. Basic generation     — minimal request, just a query
    2. Character match      — query about a known character (Amun)
    3. Debug mode           — same query + debug=True to inspect retrieval
    4. n_results variation  — bump n_results to 6 for richer context
    5. Summary mode         — generation_type="summary"
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time

import requests

BASE_URL = "http://localhost:8000"
GENERATE_URL = f"{BASE_URL}/create-eval/story_generate"

# ── Test cases ──────────────────────────────────────────────────────────────

TESTS = [
    {
        "id": 1,
        "name": "Basic generation — generic fantasy query",
        "payload": {
            "query": "A warrior who seeks redemption after a great mistake",
            "generation_type": "full_story",
            "save": False,
            "debug": False,
            "n_results": 3,
            "mode": "fast",
            "story_type": "mix",
        },
    },
    {
        "id": 2,
        "name": "Character match — Amun (half-elf warrior monk)",
        "payload": {
            "query": "A half-elf raised in the desert who trains with warrior monks",
            "generation_type": "full_story",
            "save": False,
            "debug": False,
            "n_results": 3,
            "mode": "fast",
            "story_type": "mix",
        },
    },
    {
        "id": 3,
        "name": "Debug mode — inspect retrieval + grounded facts",
        "payload": {
            "query": "A pirate who builds a lighthouse to defy a sea goddess",
            "generation_type": "full_story",
            "save": False,
            "debug": True,
            "n_results": 3,
            "mode": "fast",
            "story_type": "mix",
        },
    },
    {
        "id": 4,
        "name": "Richer context — n_results=6 with reranker",
        "payload": {
            "query": "A mage haunted by a catastrophic mistake seeks to atone",
            "generation_type": "full_story",
            "save": False,
            "debug": False,
            "n_results": 6,
            "mode": "fast",
            "story_type": "mix",
        },
    },
    {
        "id": 5,
        "name": "Summary mode — short summary output",
        "payload": {
            "query": "A paladin calls on divine power in a desperate battle",
            "generation_type": "summary",
            "save": False,
            "debug": False,
            "n_results": 3,
            "mode": "fast",
            "story_type": "mix",
        },
    },
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def check_server() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def print_divider(char: str = "─", width: int = 64) -> None:
    print(char * width)


def run_test(test: dict, show_debug: bool = False) -> bool:
    print_divider("═")
    print(f"TEST {test['id']}: {test['name']}")
    print_divider()
    print(f"Query      : {test['payload']['query']}")
    print(f"Mode       : {test['payload']['mode']}  |  "
          f"n_results: {test['payload']['n_results']}  |  "
          f"type: {test['payload']['generation_type']}")
    print_divider()

    start = time.time()
    try:
        resp = requests.post(GENERATE_URL, json=test["payload"], timeout=600)
        elapsed = time.time() - start
    except requests.Timeout:
        print("[FAIL] Request timed out after 300 s")
        return False
    except requests.ConnectionError as e:
        print(f"[FAIL] Connection error: {e}")
        return False

    if resp.status_code != 200:
        print(f"[FAIL] HTTP {resp.status_code}")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:500])
        return False

    data = resp.json()

    if not data.get("success"):
        print(f"[FAIL] success=False")
        print(f"       content : {data.get('content', '')[:300]}")
        # Print every non-empty field to surface the real error
        for k, v in data.items():
            if v and k != "content":
                print(f"       {k:20s}: {str(v)[:200]}")
        return False

    content = data.get("content", "")
    word_count = len(content.split())

    print(f"[OK]  {elapsed:.1f}s  |  {word_count} words")
    print()
    print("── Story preview (first 400 chars) ──")
    print(textwrap.fill(content[:400], width=72))
    if len(content) > 400:
        print("  [... truncated ...]")

    if show_debug and data.get("debug"):
        print()
        print("── Grounded facts ──")
        facts = data.get("grounded_facts") or []
        for i, f in enumerate(facts[:5], 1):
            print(f"  {i}. {f}")
        chunks = data.get("retrieval_chunks") or []
        print(f"\n── Retrieved {len(chunks)} chunk(s) (titles) ──")
        seen = set()
        for c in chunks:
            title = c.get("metadata", {}).get("Title", "?")
            if title not in seen:
                print(f"  • {title}")
                seen.add(title)
        attr = data.get("debug_attribution") or {}
        if attr:
            print(f"\n── Attribution gate ──")
            print(f"  novel entities : {attr.get('novel_entities', [])}")
            print(f"  truncated      : {attr.get('truncated', False)}")

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="StoryForge generation smoke tests")
    parser.add_argument("--test", type=int, default=None, help="Run a single test by ID (1-5)")
    parser.add_argument("--debug", action="store_true", help="Print grounded facts and retrieval info")
    args = parser.parse_args()

    print_divider("═")
    print("StoryForge — Generation Test Suite")
    print(f"Server: {BASE_URL}")
    print_divider("═")

    if not check_server():
        print(f"\n[ERROR] Server is not running at {BASE_URL}")
        print("        Start it first:  python main.py\n")
        sys.exit(1)

    print("[OK] Server is reachable\n")

    tests_to_run = [t for t in TESTS if args.test is None or t["id"] == args.test]
    if not tests_to_run:
        print(f"[ERROR] No test with id={args.test}")
        sys.exit(1)

    # Force debug mode on for test 3 regardless of --debug flag
    results = []
    for test in tests_to_run:
        show = args.debug or test["payload"].get("debug", False)
        ok = run_test(test, show_debug=show)
        results.append((test["id"], test["name"], ok))
        print()

    # Summary
    print_divider("═")
    print("RESULTS")
    print_divider()
    passed = sum(1 for _, _, ok in results if ok)
    for tid, name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Test {tid}: {name}")
    print_divider()
    print(f"  {passed}/{len(results)} passed")
    print_divider("═")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
