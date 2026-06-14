from __future__ import annotations

from storyforge.rag.generative_ai import clean_story_output


def test_clean_story_output_closes_quote_spacing_and_duplicate_words():
    raw = (
        '[SECTION 1: WHO]\n'
        '" Hello there," she said to them them.\n\n'
        '[SECTION 2: WHAT]\n'
        'He replied "Sure"and walked on.\n'
    )
    out = clean_story_output(raw)
    assert '"Hello there,"' in out
    assert "them them" not in out
    assert '"Sure" and' in out
