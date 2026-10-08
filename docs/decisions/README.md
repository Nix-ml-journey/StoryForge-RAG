# Architecture decisions

Short records of *why* the pipeline looks the way it does. ADR numbers are stable; new decisions take the next
number and go into the file for their topic (or a new file if none fits).

| File | ADRs |
|---|---|
| [retrieval.md](retrieval.md) | 0002 whole-corpus BM25 candidates (tested, no gain, off); 0003 `story_type` filters retrieval |
| [evaluation-and-agentic-loop.md](evaluation-and-agentic-loop.md) | 0001 Ollama judge, no API fallback; 0006 decision order, grounded judge, refine guard; 0007 close the silent-accept gaps; 0008 guard minimum, refine-once, token floor; 0009 short judge, loop helpers, sectioned writer; 0010 story-arc method for A/B comparison; 0011 target architecture (accepted, revised: judge-driven loop stays, rule loop opt-in) |
| [platform.md](platform.md) | 0004 remove vector-store insert/update routes; 0005 model choices for 16 GB |
