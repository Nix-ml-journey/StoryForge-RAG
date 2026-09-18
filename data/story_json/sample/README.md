# Sample story_json (safe for git)

This is the shape produced by `scripts/prepare_story_records.py` after you put a clean `.txt` in `data/stories/`.

Your real records live in `data/story_json/*.json` (gitignored).

Before ingest, open each JSON and check:

- `meta.title` / `meta.author` filled
- `Is_series` is true only for a real series chapter
- `chunks[].text` is story prose, not page numbers or a table of contents
- `chunks[].section` tags are not all the same word
- `summary` matches the actual plot (rewrite if the model invented one)

Guide: [`docs/DATA_PREP.md`](../../../docs/DATA_PREP.md).
