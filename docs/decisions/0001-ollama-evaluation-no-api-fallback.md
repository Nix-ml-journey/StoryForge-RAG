# ADR-0001: Evaluate drafts with local Ollama, with no API fallback

## Status
Accepted (supersedes the in-process Transformers evaluator from the 2026-09 "local evaluation" session)

## Date
2026-10-05

## Context
`Evaluation_mode: "local"` loaded `Qwen3-4B-Instruct` through Transformers on CPU and, on any
failure, silently retried through HF then Gemini. On the target machine that meant a slow CPU
judge, a second model download, and API calls (and credits) the user believed were switched off.
Ollama already serves the generation model on the GPU.

## Decision
`Evaluation_mode` accepts `ollama` (alias `local`). The judge is `Ollama_evaluation_model`
(default `Generative_model`), reached through the existing `load_ollama_llm` loader. If Ollama is
unreachable or the model is not pulled, `evaluate_model()` raises `RuntimeError`; in this mode
there is **no** HF/Gemini fallback. `Evaluation_mode: "api"` is unchanged.

## Alternatives Considered
- Keep Transformers-on-CPU: slow, duplicate model, hidden fallback. Rejected.
- Transformers-on-GPU: competes with Ollama for 16 GB VRAM. Rejected.
- Keep the fallback but log it: the failure mode (silently spending credits) stays. Rejected.

## Consequences
- Fully offline pipeline; each judgement costs one Ollama call.
- The judge is usually the same model as the generator, so scores can run generous. Prompts were
  recalibrated (strict 1-10 anchors) and `Ollama_evaluation_model` lets you pick a different judge.
- If Ollama dies mid-run the agentic loop logs an ERROR and falls back to completeness-only
  decisions (it still never ACCEPTs a draft with zero grounded facts).
