from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from storyforge.config.config import load_config, load_prompts
from storyforge.data.hf_summary import get_hf_token, summarize_long_text


_ALLOWED_SECTION_TAGS = {
    "setup",
    "inciting_incident",
    "rising_action",
    "confrontation",
    "twist",
    "climax",
    "fallout",
    "resolution",
    "epilogue",
    "setting",
    "negotiation",
}


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _heuristic_section_tag(*, chunk_index: int, chunk_total: int) -> str:
    if chunk_total <= 1:
        return "setup"
    if chunk_index == 1:
        return "setup"
    if chunk_index == 2:
        return "inciting_incident" if chunk_total >= 4 else "rising_action"
    if chunk_index >= chunk_total:
        return "resolution"
    if chunk_index == chunk_total - 1:
        return "climax" if chunk_total >= 5 else "confrontation"
    return "rising_action"


def _load_local_section_labeler(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None),
    )

    pipe_kwargs: dict[str, Any] = {}
    if not torch.cuda.is_available():
        pipe_kwargs["device"] = -1

    gen = pipeline("text-generation", model=model, tokenizer=tok, return_full_text=False, **pipe_kwargs)
    return gen, tok


def _label_one_section(*, gen, tok, system: str, user: str) -> str:
    from transformers import GenerationConfig

    prompt = f"{system.strip()}\n\n{user.strip()}\n"
    gen_cfg = GenerationConfig.from_model_config(gen.model.config)
    gen_cfg.max_new_tokens = 24
    gen_cfg.max_length = None
    gen_cfg.do_sample = False
    gen_cfg.pad_token_id = tok.eos_token_id
    out = gen(prompt, generation_config=gen_cfg)
    if isinstance(out, list) and out and isinstance(out[0], dict) and out[0].get("generated_text") is not None:
        text = str(out[0].get("generated_text") or "").strip()
    else:
        text = str(out).strip()
    return (text.splitlines()[0] if text else "").strip()


def enrich_story_records(
    *,
    base_path: str | Path | None = None,
    limit: int = 0,
    overwrite_summary: bool = False,
    overwrite_sections: bool = False,
    skip_summary: bool = False,
    skip_sections: bool = False,
    dry_run: bool = False,
    enable_tqdm: bool | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    prompts = load_prompts()
    base = Path(base_path or cfg.get("BASE_PATH") or ".").resolve()
    records_dir = (base / "data" / "story_json").resolve()
    files = sorted([p for p in records_dir.glob("*.json") if p.is_file()], key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No story_json records found in {records_dir}")
    if limit and limit > 0:
        files = files[:limit]

    hf_token = ""
    summary_model = str(cfg.get("HF_summary_model") or "sshleifer/distilbart-cnn-12-6")
    if not skip_summary:
        hf_token = get_hf_token(cfg)

    ingest_prompts = (prompts or {}).get("ingest") or {}
    section_system = str(
        ingest_prompts.get("section_label_system") or "You label a story chunk with one short section tag. Output only the tag."
    )
    section_user_tpl = str(
        ingest_prompts.get("section_label_user")
        or "TITLE: {title}\nCHUNK_ID: {chunk_id}\nCHUNK:\n{chunk_text}\n\nReturn a short tag."
    )
    section_model = str(cfg.get("Section_label_model") or cfg.get("Generative_model") or "Qwen/Qwen2.5-1.5B-Instruct")
    section_bundle = None if skip_sections else _load_local_section_labeler(section_model)

    if enable_tqdm is None:
        enable_tqdm = os.getenv("STORYFORGE_TQDM", "").strip() not in {"", "0", "false", "False"}

    files_iter: Any = files
    if enable_tqdm:
        try:
            from tqdm import tqdm  # type: ignore

            files_iter = tqdm(files, desc="Enrich story records", unit="file")
        except Exception:
            files_iter = files

    updated = 0
    for fp in files_iter:
        rec = _read_json(fp)
        if not rec:
            continue
        changed = False
        title = fp.stem
        raw_text = str(rec.get("raw_text") or "")

        if not skip_summary and (overwrite_summary or _is_blank(rec.get("summary"))):
            if raw_text.strip():
                try:
                    summary = summarize_long_text(hf_token, summary_model, raw_text)
                except KeyboardInterrupt:
                    summary = ""
                except Exception:
                    summary = ""
                if summary.strip():
                    rec["summary"] = summary.strip()
                    changed = True

        if not skip_sections:
            chunks = rec.get("chunks")
            if isinstance(chunks, list) and section_bundle is not None:
                section_gen, section_tok = section_bundle
                last_label = ""
                repeat_streak = 0
                total = len(chunks)
                for idx, ch in enumerate(chunks, start=1):
                    if not isinstance(ch, dict):
                        continue
                    if (not overwrite_sections) and (not _is_blank(ch.get("section"))):
                        continue
                    chunk_text = str(ch.get("text") or "")
                    if not chunk_text.strip():
                        continue
                    user = section_user_tpl.format(
                        title=title,
                        chunk_id=str(ch.get("chunk_id") or ""),
                        chunk_index=str(idx),
                        chunk_total=str(total),
                        chunk_text=chunk_text[:2000],
                    )
                    label = _label_one_section(gen=section_gen, tok=section_tok, system=section_system, user=user).lower().strip()

                    if label == last_label and label:
                        repeat_streak += 1
                    else:
                        repeat_streak = 0
                        last_label = label
                    if (label not in _ALLOWED_SECTION_TAGS) or repeat_streak >= 3:
                        label = _heuristic_section_tag(chunk_index=idx, chunk_total=total)

                    if label:
                        ch["section"] = label
                        changed = True

        if changed:
            updated += 1
            if not dry_run:
                _write_json(fp, rec)

    return {"updated": updated, "total": len(files), "dry_run": dry_run}

