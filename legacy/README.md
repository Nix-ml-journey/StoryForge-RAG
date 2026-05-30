# Legacy code (pre-`src/storyforge` layout)

These packages are **deprecated**. The active application lives under `src/storyforge/` and starts from `main.py` at the repo root.

They are kept for reference while the migration finishes. Do not import them from new code.

| Folder | Replaced by |
|--------|-------------|
| `API/` | `src/storyforge/api/` |
| `Book_search/` | `src/storyforge/book_search/` |
| `Data/` | `src/storyforge/data/` |
| `Evaluation/` | `src/storyforge/evaluation/` |
| `Generative_AI/` | `src/storyforge/rag/` + `src/storyforge/generative/` |
| `Orchestrator/` | `src/storyforge/orchestrator/` |
| `Vector_Store/` | `src/storyforge/vector_store/` |
