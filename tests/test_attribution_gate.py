from __future__ import annotations

import json

import pytest

import re

from storyforge.rag.attribution import (
    attribution_violations,
    extract_named_entities_heuristic,
    format_facts_for_prompt,
    parse_grounded_facts_json,
    repair_json,
)


def test_repair_json_strips_fences_and_trailing_commas():
    raw = """```json
    { "facts": [ { "type": "who", "fact": "Alana is a soldier", "source_chunk_ids": ["a_1"], }, ], }
    ```"""
    repaired = repair_json(raw)
    data = json.loads(repaired)
    assert "facts" in data


def test_parse_grounded_facts_drops_missing_sources():
    raw = """
    {
      "facts": [
        {"type":"who","fact":"Alana is a soldier","source_chunk_ids":["a_1"],"quote":"Alana, a soldier"},
        {"type":"where","fact":"In Eldoria","source_chunk_ids":[]}
      ]
    }
    """
    parsed = parse_grounded_facts_json(raw)
    assert len(parsed.facts) == 1
    assert parsed.facts[0].source_chunk_ids == ("a_1",)


def test_attribution_violations_flags_new_named_entities():
    parsed = parse_grounded_facts_json(
        """
        {"facts":[{"type":"who","fact":"Alana meets Zoruk","source_chunk_ids":["c1"],"quote":"Alana met Zoruk"}]}
        """
    )
    story = "Alana met Zoruk. Then Cassandra arrived with a sword."
    viol = attribution_violations(story=story, facts=parsed.facts)
    assert "Cassandra" in viol


# ── Threshold-based truncation logic ──────────────────────────────────────────


def _should_truncate(story: str, facts, threshold: int = 8) -> bool:
    """
    Mirror of the threshold logic in langchain_rag.generate_story_3step_langchain.

    Section headers are stripped before the check so structural words like "Setup",
    "Climax", and "Conclusion" do not inflate the violation count.
    """
    story_body = re.sub(r"^\[SECTION \d+:.*?\]\s*", "", story, flags=re.MULTILINE)
    violations = attribution_violations(story=story_body, facts=facts)
    if not violations:
        return False
    all_found = extract_named_entities_heuristic(story_body)
    violation_ratio = len(violations) / max(1, len(all_found))
    return len(violations) > threshold or violation_ratio > 0.6


def test_section_header_words_do_not_trigger_truncation():
    """
    Section headers produced by _flow_section_headers() contain capitalized words
    like 'Setup', 'Problem', 'Challenge', 'Climax', 'Moral', 'Conclusion' that are
    NOT in the grounded facts. Before the threshold fix these always caused truncation.
    With threshold=8 and only a few structural words as violations, truncation must not fire.
    """
    parsed = parse_grounded_facts_json(
        '{"facts":[{"type":"who","fact":"Alana fights Zoruk","source_chunk_ids":["c1"]}]}'
    )
    # Simulated story that includes section headers and normal creative text.
    story = (
        "[SECTION 1: WHO, WHERE, WHEN (The Setup)]\n\n"
        "Alana, a warrior, stood at the gates of Eldoria where Zoruk waited.\n\n"
        "[SECTION 2: WHAT (The Problem Starts)]\n\n"
        "Zoruk challenged Alana to a duel at dawn.\n\n"
        "[SECTION 3: TWIST/COMPLICATION (The Challenge)]\n\n"
        "Alana discovered that Zoruk wielded a cursed blade.\n\n"
        "[SECTION 4: HOW (The Big Action/Climax)]\n\n"
        "Alana fought Zoruk and disarmed the cursed blade.\n\n"
        "[SECTION 5: WHY/OUTCOME (The Moral and Conclusion)]\n\n"
        "Zoruk surrendered and Eldoria was saved."
    )
    # Section headers are stripped before attribution check, so structural words like
    # 'Setup', 'Problem', 'Challenge' do not count as novel named entities.
    assert not _should_truncate(story, parsed.facts, threshold=8)


def test_many_hallucinated_entities_trigger_truncation():
    """
    A story that invents many new named entities (> threshold) should still be truncated.
    """
    parsed = parse_grounded_facts_json(
        '{"facts":[{"type":"who","fact":"Alana fights Zoruk","source_chunk_ids":["c1"]}]}'
    )
    # 10+ entirely new named entities that are not in facts at all.
    hallucinated_story = (
        "Alana fought Zoruk. But Cassandra, Mirabel, Theodoric, Valdris, "
        "Sulenne, Orindal, Brethwyn, Caelan, Lyressa, and Dorvan all appeared "
        "uninvited at the battle of Darkhollow."
    )
    assert _should_truncate(hallucinated_story, parsed.facts, threshold=8)


# -- format_facts_for_prompt ---------------------------------------------------


def test_format_facts_for_prompt_produces_numbered_bullet_list():
    """Structured facts produce one numbered line per fact."""
    parsed = parse_grounded_facts_json(
        """
        {
          "facts": [
            {"type":"who",  "fact":"Alana is a soldier",  "source_chunk_ids":["c1"], "quote":"Alana, a soldier"},
            {"type":"where","fact":"In the city of Zorr", "source_chunk_ids":["c2"], "quote":""}
          ]
        }
        """
    )
    result = format_facts_for_prompt(parsed)

    lines = result.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("1.")
    assert "[who]" in lines[0]
    assert "Alana is a soldier" in lines[0]
    assert "c1" in lines[0]
    # quote should appear when non-empty
    assert "Alana, a soldier" in lines[0]
    assert lines[1].startswith("2.")
    assert "[where]" in lines[1]


def test_format_facts_for_prompt_returns_empty_string_for_no_facts():
    """Empty facts list must return empty string so callers fall back to raw JSON."""
    parsed = parse_grounded_facts_json("{}")
    result = format_facts_for_prompt(parsed)
    assert result == ""


def test_format_facts_for_prompt_omits_quote_when_blank():
    """A fact with an empty quote string should not include a quote section."""
    parsed = parse_grounded_facts_json(
        '{"facts":[{"type":"what","fact":"The bridge collapsed","source_chunk_ids":["c3"],"quote":""}]}'
    )
    result = format_facts_for_prompt(parsed)
    assert "The bridge collapsed" in result
    # Parenthesised quote should be absent when quote is blank
    assert '("")' not in result
