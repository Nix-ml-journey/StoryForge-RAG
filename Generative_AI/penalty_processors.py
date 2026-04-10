import logging
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import LogitsProcessor, LogitsProcessorList

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_setup_config() -> dict[str, Any]:
    root_dir = Path(__file__).parent.parent
    cfg_path = root_dir / "setup.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG: dict[str, Any] = _load_setup_config()

REPETITION_PENALTY: float = _CONFIG.get("Repetition_penalty", 1.3)
NO_REPEAT_NGRAM_SIZE: int = _CONFIG.get("No_repeat_ngram_size", 4)
FREQUENCY_PENALTY: float = _CONFIG.get("Frequency_penalty", 0.0)
PRESENCE_PENALTY: float = _CONFIG.get("Presence_penalty", 0.0)

_EXPORT_CONSTS = (
    "REPETITION_PENALTY",
    "NO_REPEAT_NGRAM_SIZE",
    "FREQUENCY_PENALTY",
    "PRESENCE_PENALTY",
)
_EXPORT_PENALTIES = (
    "build_logits_processors",
    "get_stopword_ids",
    "FrequencyPenaltyProcessor",
    "PresencePenaltyProcessor",
)
_EXPORT_CLEANING = (
    "apply_language_gate",
    "clean_generated_output",
    "extract_who_names",
    "parse_5w1h_sections",
    "normalize_character_names",
    "clean_layer2_artifacts",
    "clean_layer3_story",
    "truncate_sentences",
    "normalize_fullwidth",
    "has_unbalanced_quotes",
    "has_together_they_artifact",
    "count_ungrounded_proper_nouns",
    "story_has_proper_ending",
    "detect_junk_density",
    "check_query_context_alignment",
    "query_context_alignment_score",
    "pick_best_context_story",
    "check_how_coverage",
    "validate_layer1",
    "validate_layer2_section",
    "validate_layer3_section",
)
__all__ = [*_EXPORT_CONSTS, *_EXPORT_PENALTIES, *_EXPORT_CLEANING]

_DEFAULT_STOPWORDS = (
    "a an the this that these those "
    "i me my mine we us our ours "
    "you your yours he him his she her hers "
    "it its they them their theirs "
    "is am are was were be been being "
    "have has had do does did "
    "will would shall should can could may might must "
    "to of in on at by for with from "
    "up out off over into onto upon "
    "and but or nor so yet if then than "
    "not no as about after before between through "
    "what who whom which where when how why "
    "all each every some any few more most "
    "just also very too quite still even"
)
_stopwords_cfg = _CONFIG.get("Stopwords")
if isinstance(_stopwords_cfg, (list, tuple, set)):
    STOPWORDS = {str(w).strip().lower() for w in _stopwords_cfg if str(w).strip()}
elif isinstance(_stopwords_cfg, str) and _stopwords_cfg.strip():
    STOPWORDS = {w.strip().lower() for w in _stopwords_cfg.split() if w.strip()}
else:
    STOPWORDS = {w.strip().lower() for w in _DEFAULT_STOPWORDS.split() if w.strip()}

_stopword_ids_cache: set[int] = set()


def get_stopword_ids(tokenizer) -> set[int]:
    if _stopword_ids_cache:
        return _stopword_ids_cache
    ids: set[int] = set()
    for word in STOPWORDS:
        for variant in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            ids.update(token_ids)
    _stopword_ids_cache.update(ids)
    logging.info(f"Penalty processors: cached {len(ids)} stopword token IDs")
    return _stopword_ids_cache


class FrequencyPenaltyProcessor(LogitsProcessor):
    def __init__(self, penalty: float, prompt_length: int, stopword_ids: set):
        self.penalty = penalty
        self.prompt_length = prompt_length
        self.stopword_ids = stopword_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        for i in range(input_ids.shape[0]):
            gen_ids = input_ids[i, self.prompt_length:]
            if gen_ids.numel() == 0:
                continue
            counts = torch.bincount(gen_ids, minlength=vocab_size)[:vocab_size].float()
            if self.stopword_ids:
                mask = torch.ones(vocab_size, dtype=torch.bool, device=scores.device)
                stop_ids = [sid for sid in self.stopword_ids if sid < vocab_size]
                if stop_ids:
                    mask[torch.tensor(stop_ids, device=scores.device)] = False
                counts *= mask.float()
            scores[i] -= self.penalty * counts
        return scores


class PresencePenaltyProcessor(LogitsProcessor):
    def __init__(self, penalty: float, prompt_length: int, stopword_ids: set):
        self.penalty = penalty
        self.prompt_length = prompt_length
        self.stopword_ids = stopword_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(input_ids.shape[0]):
            gen_ids = input_ids[i, self.prompt_length:]
            if gen_ids.numel() == 0:
                continue
            unique_tokens = gen_ids.unique()
            if self.stopword_ids:
                keep = torch.tensor(
                    [t.item() not in self.stopword_ids for t in unique_tokens],
                    dtype=torch.bool, device=scores.device,
                )
                unique_tokens = unique_tokens[keep]
            scores[i, unique_tokens] -= self.penalty
        return scores


def build_logits_processors(prompt_length: int, tokenizer) -> LogitsProcessorList:
    processors = LogitsProcessorList()
    if FREQUENCY_PENALTY > 0 or PRESENCE_PENALTY > 0:
        stopword_ids = get_stopword_ids(tokenizer)
        if FREQUENCY_PENALTY > 0:
            processors.append(FrequencyPenaltyProcessor(FREQUENCY_PENALTY, prompt_length, stopword_ids))
        if PRESENCE_PENALTY > 0:
            processors.append(PresencePenaltyProcessor(PRESENCE_PENALTY, prompt_length, stopword_ids))
    return processors


# ── Text cleaning & language gate ──

def apply_language_gate(text: str) -> str:
    if not (text or "").strip():
        return (text or "").strip()
    text = text.strip()
    cjk = re.search(r"[\u4e00-\u9fff\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]", text)
    if cjk:
        text = text[: cjk.start()].strip()
    other = re.search(r"[\u0600-\u06ff\u0400-\u04ff]", text)
    if other:
        text = text[: other.start()].strip()
    return text


def _deduplicate_paragraphs(text: str, similarity_threshold: float = 0.40) -> str:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        return text

    def _word_set(p: str) -> set[str]:
        return set(re.findall(r"[a-z]+", p.lower()))

    kept: list[str] = []
    kept_word_sets: list[set[str]] = []
    for para in paragraphs:
        if re.match(r"^\[?\s*SECTION\s+\d", para.strip(), re.IGNORECASE):
            kept.append(para)
            kept_word_sets.append(set())
            continue
        ws = _word_set(para)
        if not ws:
            kept.append(para)
            kept_word_sets.append(ws)
            continue
        is_dup = False
        for prev_ws in kept_word_sets:
            if not prev_ws:
                continue
            intersection = len(ws & prev_ws)
            union = len(ws | prev_ws)
            if union > 0 and intersection / union >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(para)
            kept_word_sets.append(ws)

    return "\n\n".join(kept)


def clean_generated_output(text: str, aggressive: bool = True) -> str:
    if not (text or "").strip():
        return (text or "").strip()
    text = text.strip()
    if "\n---\n" in text:
        text = text.split("\n---\n")[0].strip()
    text = re.sub(r"\n\[.{1,80} by .{1,80}\]\s*$", "", text)

    cjk = re.search(r"[\u4e00-\u9fff\u3000-\u303f]", text)
    if cjk:
        text = text[: cjk.start()].strip()

    if aggressive:
        blob = re.search(r"\S{40,}", text)
        if blob:
            text = text[: blob.start()].strip()

    text = _deduplicate_paragraphs(text)

    if not aggressive:
        last_end = max(
            text.rfind("."),
            text.rfind("!"),
            text.rfind("?"),
            text.rfind('."'),
            text.rfind('!"'),
            text.rfind('?"'),
        )
        if last_end >= len(text) * 0.95:
            text = text[: last_end + 1]
        return text.strip()

    for para in text.split("\n\n"):
        stripped = para.strip()
        words = re.split(r"\s+", stripped) if stripped else []
        word_count = len(words)
        if word_count < 20:
            continue

        para_for_counting = re.sub(r"\.\.\.+", ".", stripped)
        sentence_enders = len(re.findall(r"[.!?]", para_for_counting))

        if stripped.endswith("...") and word_count > 40:
            cut_pos = text.find(para)
            if cut_pos > 0:
                text = text[:cut_pos].strip()
            break

        if word_count > 40 and sentence_enders == 0:
            cut_pos = text.find(para)
            if cut_pos > 0:
                text = text[:cut_pos]
            break
        if word_count > 60 and sentence_enders < max(1, word_count // 40):
            cut_pos = text.find(para)
            if cut_pos > 0:
                text = text[:cut_pos]
            break
        avg_words_per_sentence = word_count / max(sentence_enders, 1)
        if word_count > 30 and avg_words_per_sentence > 35:
            cut_pos = text.find(para)
            if cut_pos > 0:
                text = text[:cut_pos]
            break

    last_end = max(
        text.rfind("."),
        text.rfind("!"),
        text.rfind("?"),
        text.rfind('."'),
        text.rfind('!"'),
        text.rfind('?"'),
    )
    if last_end > len(text) // 3:
        text = text[: last_end + 1]

    return text.strip()


def extract_who_names(five_w_one_h: str) -> list[str]:
    if not five_w_one_h:
        return []

    _NEXT_ELEMENT = (
        "- WHAT:",
        "WHAT:",
        "- WHEN:",
        "WHEN:",
        "- WHERE:",
        "WHERE:",
        "- WHY:",
        "WHY:",
        "- HOW:",
        "HOW:",
    )

    who_block = ""
    lines = five_w_one_h.splitlines()
    capture = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^-?\s*WHO\s*:", stripped, re.IGNORECASE):
            who_block = re.split(r"WHO\s*:", stripped, maxsplit=1, flags=re.IGNORECASE)[-1].strip()
            capture = True
            continue
        if capture:
            if any(stripped.startswith(tok) for tok in _NEXT_ELEMENT):
                break
            if stripped:
                who_block += "\n" + stripped

    if not who_block:
        return []

    raw_lines = [ln.strip() for ln in who_block.splitlines() if ln.strip()]
    if not raw_lines:
        raw_lines = [p.strip() for p in re.split(r"\s{2,}", who_block) if p.strip()]

    titles = frozenset({"mr", "mrs", "ms", "dr"})
    extracted: list[str] = []
    for ln in raw_lines:
        if ln.startswith("-"):
            ln = ln.lstrip("-").strip()

        if " - " in ln:
            name_part = ln.split(" - ", 1)[0].strip()
        elif "," in ln:
            name_part = ln.split(",", 1)[0].strip()
        else:
            name_part = ln.strip()

        m = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", name_part)
        if not m:
            continue
        name = m.group(1).strip()

        parts = name.split()
        if len(parts) >= 2 and parts[0].lower() in titles and not parts[0].endswith("."):
            name = parts[0] + ". " + " ".join(parts[1:])

        extracted.append(name)

    seen: set[str] = set()
    names: list[str] = []
    for n in extracted:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            names.append(n)
    return names


def parse_5w1h_sections(layer1_text: str) -> dict[str, str]:
    if not layer1_text:
        return {}

    keys = ["who", "what", "when", "where", "why", "how"]
    results: dict[str, list[str]] = {k: [] for k in keys}

    current: str | None = None
    for raw_line in layer1_text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            if current is not None:
                results[current].append("")
            continue

        m = re.match(r"^-?\s*(WHO|WHAT|WHEN|WHERE|WHY|HOW)\s*:\s*$", stripped, flags=re.IGNORECASE)
        if m:
            current = m.group(1).lower()
            continue

        if current is None:
            continue

        results[current].append(stripped)

    joined: dict[str, str] = {}
    for k, lines in results.items():
        t = "\n".join(lines).strip()
        if t:
            joined[k] = t
    return joined


def _levenshtein(a: str, b: str) -> int:
    a = a.lower()
    b = b.lower()
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i, ca in enumerate(a, start=1):
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev
    return prev[-1]


_DEFAULT_NORMALIZE_NAME_STOPLIST = (
    "in at on amid amidst the as when where what why how it he she they we "
    "so but and or if no yes "
    "late old air mr one two first last new dark deep full long same such each both all "
    "few more most other some only own just then than too very "
    "under with during after before through between over"
)
_normalize_stop_cfg = _CONFIG.get("Normalize_name_stoplist")
if isinstance(_normalize_stop_cfg, (list, tuple, set)):
    _NORMALIZE_NAME_STOPLIST = frozenset(
        {str(w).strip().lower() for w in _normalize_stop_cfg if str(w).strip()}
    )
elif isinstance(_normalize_stop_cfg, str) and _normalize_stop_cfg.strip():
    _NORMALIZE_NAME_STOPLIST = frozenset(
        {w.strip().lower() for w in _normalize_stop_cfg.split() if w.strip()}
    )
else:
    _NORMALIZE_NAME_STOPLIST = frozenset(
        {w.strip().lower() for w in _DEFAULT_NORMALIZE_NAME_STOPLIST.split() if w.strip()}
    )


def normalize_character_names(story_summary: str, five_w_one_h: str) -> str:
    who_names = extract_who_names(five_w_one_h)
    if not who_names:
        return story_summary

    canonical = [n.strip() for n in who_names if n.strip()]
    if not canonical:
        return story_summary

    canonical_lower = {n.lower(): n for n in canonical}

    lines = story_summary.splitlines()
    candidate_counts: dict[str, int] = {}
    for line in lines:
        if line.lstrip().startswith("["):
            continue
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", line):
            key = match.strip()
            if not key:
                continue
            candidate_counts[key] = candidate_counts.get(key, 0) + 1

    if not candidate_counts:
        return story_summary

    MAX_EDIT_DISTANCE = 3

    mapping: dict[str, str] = {}
    for cand in candidate_counts:
        c_key = cand.lower()
        if c_key in canonical_lower:
            continue
        if c_key in _NORMALIZE_NAME_STOPLIST:
            continue
        if " " in cand and any(w.lower() in _NORMALIZE_NAME_STOPLIST for w in cand.split()):
            continue

        best_name = canonical[0]
        best_dist = _levenshtein(cand, best_name)
        for name in canonical[1:]:
            lname = name.lower()
            if c_key in lname or lname in c_key:
                best_name = name
                best_dist = 0
                break
            d = _levenshtein(cand, name)
            if d < best_dist:
                best_dist = d
                best_name = name

        if best_dist > MAX_EDIT_DISTANCE:
            continue

        mapping[cand] = best_name

    if not mapping:
        return story_summary

    sorted_items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    new_lines: list[str] = []
    for line in lines:
        if line.lstrip().startswith("["):
            new_lines.append(line)
            continue
        new_line = line
        for cand, target in sorted_items:
            pattern = r"\b" + re.escape(cand) + r"\b"
            new_line = re.sub(pattern, target, new_line)
        new_lines.append(new_line)

    return "\n".join(new_lines)


def clean_layer2_artifacts(text: str) -> str:
    if not text or not text.strip():
        return text
    out = text

    out = re.sub(r"Mr\.\s*Mr\.\s*", "Mr. ", out, flags=re.IGNORECASE)
    out = re.sub(r"\(period\)", ".", out, flags=re.IGNORECASE)
    out = re.sub(r"\.\s*Period\.", ".", out)
    out = re.sub(r"\s+Period\.\s*", ". ", out)

    out = re.sub(r"[,;'\"]\s*Together they\b[^.!?\n]*[.!?]?", ".", out)
    out = re.sub(
        r"\b(beside|around|behind|toward|towards|near|against)\s+and\s+he\b",
        r"\1 him",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\b(beside|around|behind|toward|towards|near|against)\s+and\b(?=\s*[,.])",
        r"\1",
        out,
        flags=re.IGNORECASE,
    )

    # Fix missing spaces after punctuation: "years,the" → "years, the"
    out = re.sub(r"([,;:])([a-zA-Z])", r"\1 \2", out)
    # Fix missing spaces after sentence-ending punctuation: "pactum.His" → "pactum. His"
    out = re.sub(r"([.!?])([a-zA-Z])", r"\1 \2", out)

    _COMMON_WORDS = frozenset({
        "the", "and", "his", "her", "its", "our", "per", "from", "with",
        "into", "upon", "near", "that", "this", "what", "when", "who",
        "how", "for", "but", "not", "was", "are", "has", "had", "will",
        "can", "may", "all", "one", "two", "out", "now", "new", "old",
        "man", "son", "returns", "meets", "instead", "recognizes",
        "challenges", "situation", "merchant", "dwarf",
    })

    def _fix_collapsed_words(line: str) -> str:
        """Break apart long all-alpha tokens by inserting spaces before common words."""
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            return line
        words = stripped.split()
        has_long = any(len(w) > 18 and w.replace(",", "").replace(".", "").isalpha() for w in words)
        if not has_long:
            return line
        fixed: list[str] = []
        for word in words:
            clean_w = word.replace(",", "").replace(".", "")
            if len(clean_w) > 18 and clean_w.isalpha():
                result = word
                for cw in sorted(_COMMON_WORDS, key=len, reverse=True):
                    pattern = rf"(?<=[a-z])({re.escape(cw)})(?=[a-z])"
                    result = re.sub(pattern, rf" \1", result, flags=re.IGNORECASE)
                fixed.append(result)
            else:
                fixed.append(word)
        indent = line[: len(line) - len(line.lstrip())]
        return indent + " ".join(fixed)

    out = "\n".join(_fix_collapsed_words(ln) for ln in out.splitlines())

    def _repair_fragmented_line(line: str) -> str:
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            return line
        alpha = [c for c in stripped if c.isalpha()]
        if len(alpha) < 10:
            return line
        upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
        if upper_ratio < 0.30:
            return line
        tokens = stripped.split()
        merged: list[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            while i + 1 < len(tokens) and (len(tok) <= 4 or len(tokens[i + 1]) <= 4):
                candidate = tok + tokens[i + 1]
                if candidate.replace(".", "").replace(",", "").isalpha() or len(tokens[i + 1]) <= 2:
                    tok = candidate
                    i += 1
                else:
                    break
            merged.append(tok)
            i += 1
        result = " ".join(merged)
        result = result.lower()
        result = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), result)
        indent = line[: len(line) - len(line.lstrip())]
        return indent + result

    out = "\n".join(_repair_fragmented_line(ln) for ln in out.splitlines())

    out = re.sub(r"([a-z])([A-Z]+)([a-z])", lambda m: m.group(1) + m.group(2).lower() + m.group(3), out)
    out = re.sub(r"([a-z])([A-Z])", r"\1 \2", out)
    out = re.sub(r"\.\.(?!\.)", ".", out)

    NAME_PATTERN = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"

    def _fix_name_and_walks(line: str) -> str:
        return re.sub(rf"\b({NAME_PATTERN})\s+and\s+walks\s+", r"\1 walks with ", line, flags=re.IGNORECASE)

    def _fix_name_and_is(line: str) -> str:
        return re.sub(rf"\b({NAME_PATTERN})\s+and\s+is\s+", r"\1 is with ", line, flags=re.IGNORECASE)

    def _fix_name_and_lumbers(line: str) -> str:
        return re.sub(rf"\b({NAME_PATTERN})\s+and\s+lumbers\s+", r"\1 lumbers ", line, flags=re.IGNORECASE)

    def _fix_name_lumbers_name(line: str) -> str:
        return re.sub(rf"\b({NAME_PATTERN})\s+lumbers\s+\1,\s*", r"\1 lumbers, ", line, flags=re.IGNORECASE)

    def _fix_name_they(line: str) -> str:
        return re.sub(rf"\b({NAME_PATTERN})\s+they\b", "Together they", line, flags=re.IGNORECASE)

    def _fix_trailing_duplicated_name(line: str) -> str:
        m = re.search(rf"\s+({NAME_PATTERN})\.\s*$", line)
        if not m:
            return line
        word = m.group(1)
        before = line[: m.start()]
        if len(word) <= 40 and re.search(rf"\b{re.escape(word)}\b", before):
            line = before.rstrip()
            if not line.endswith("."):
                line = line + "."
        return line

    def _fix_double_period(line: str) -> str:
        return re.sub(r" \.\s*$", "", line)

    def _fix_double_dot_end(line: str) -> str:
        stripped = line.rstrip()
        if stripped.endswith(".."):
            stripped = stripped[:-1]
        return stripped + line[len(line.rstrip()) :]

    def _fix_challenged_and_to(line: str) -> str:
        target = "challenged and to"
        if target in line:
            return line.replace(target, "challenged him to")
        return line

    def _fix_and_fragments(line: str) -> str:
        line = re.sub(r"\bled\s+and\s+(past|into|through|toward)\b", r"led \1", line, flags=re.IGNORECASE)
        line = line.replace("guiding and closer", "guiding him closer")
        line = line.replace("leading and deeper", "leading him deeper")
        line = line.replace("carrying and closer", "carrying him closer")
        line = line.replace("pressing down upon and once more", "pressing down upon him once more")
        line = line.replace("burned within and now", "burned within him now")
        line = line.replace("from behind and.", "from behind.")
        if "Beside and walked " in line:
            parts = line.split("Beside and walked ", 1)
            line = parts[0] + "Beside " + parts[1] + " walked"
        line = line.replace(" beside and,", " beside,")
        line = line.replace(" beside and—", " beside—")
        line = line.replace(" beside and.", " beside.")
        line = line.replace(" around and.", " around.")
        line = line.replace(" answer and except", " answer except")
        if line.lstrip().startswith("A Together they"):
            line = line.replace("A Together they", "Together they", 1)
        return line

    def _fix_together_they_conditionals(line: str) -> str:
        s = line.lstrip()
        lower = s.lower()
        for verb in ("falter", "lose", "fail", "stumble", "hesitate", "give up"):
            if lower.startswith(f"together they {verb}"):
                indent = line[: len(line) - len(s)]
                rest = s[len("Together ") :]
                return indent + "If " + rest
        return line

    _FRAGMENT_SKIP = frozenset(
        {
            "in",
            "on",
            "at",
            "as",
            "under",
            "over",
            "by",
            "with",
            "from",
            "after",
            "before",
            "during",
            "through",
            "between",
            "within",
            "into",
            "upon",
            "among",
            "across",
            "along",
            "around",
            "beyond",
            "behind",
            "beneath",
            "beside",
            "despite",
            "outside",
            "inside",
            "since",
            "until",
            "toward",
            "towards",
            "throughout",
            "without",
            "above",
            "below",
            "near",
            "past",
            "against",
            "except",
        }
    )

    def _fix_name_the_fragment(line: str) -> str:
        s = line.lstrip()
        indent = line[: len(line) - len(s)]
        m = re.match(rf"^({NAME_PATTERN})\s+the\s+", s)
        if not m:
            return line
        name = m.group(1)
        if name.lower() in _FRAGMENT_SKIP or name.lower() in _NORMALIZE_NAME_STOPLIST:
            return line
        rest = s[m.end() :]
        return indent + f"{name} felt the " + rest

    def _fix_possessive_together_they(line: str) -> str:
        s = line.lstrip()
        indent = line[: len(line) - len(s)]
        m = re.match(rf"^({NAME_PATTERN})'s\s+Together\s+they\b\s*", s)
        if not m:
            return line
        rest = s[m.end() :]
        return indent + "Together they " + rest

    def _strip_trailing_literal_period_word(line: str) -> str:
        return re.sub(r"\s+Period\.\s*$", ".", line)

    detectors = [
        _fix_name_and_walks,
        _fix_name_and_is,
        _fix_name_and_lumbers,
        _fix_name_lumbers_name,
        _fix_name_they,
        _fix_trailing_duplicated_name,
        _fix_double_period,
        _fix_and_fragments,
        _fix_possessive_together_they,
        _fix_together_they_conditionals,
        _fix_name_the_fragment,
        _fix_double_dot_end,
        _fix_challenged_and_to,
        _strip_trailing_literal_period_word,
    ]

    cleaned: list[str] = []
    for line in out.splitlines():
        if line.lstrip().startswith("["):
            cleaned.append(line)
            continue
        for detector in detectors:
            line = detector(line)
        cleaned.append(line)

    return "\n".join(cleaned)


def clean_layer3_story(full_story: str, five_w_one_h: str | None = None) -> str:
    if not full_story or not full_story.strip():
        return full_story
    out = full_story

    if five_w_one_h:
        out = normalize_character_names(out, five_w_one_h)

    out = clean_layer2_artifacts(out)

    if five_w_one_h:
        sections = parse_5w1h_sections(five_w_one_h)
        why_text = sections.get("why") or ""
        who_names = extract_who_names(five_w_one_h)
        who_display = ""
        if who_names:
            if len(who_names) == 1:
                who_display = who_names[0]
            elif len(who_names) == 2:
                who_display = f"{who_names[0]} and {who_names[1]}"
            else:
                who_display = f"{who_names[0]} and {len(who_names) - 1} others"

        if why_text and who_display:
            why_one_sentence = truncate_sentences(why_text, max_sentences=1).strip()
            why_clean = why_one_sentence.rstrip(".")

            if len(why_clean.split()) > 15:
                pass
            elif why_clean.lower().startswith("to "):
                verb_phrase = why_clean[3:].strip()
                first_word = verb_phrase.split()[0] if verb_phrase.split() else ""
                if first_word.lower() in ("prove", "find", "seek", "discover", "learn", "earn", "show", "gain"):
                    outcome_line = f"In the end, {who_display} found the courage to {verb_phrase}."
                else:
                    outcome_line = f"In the end, {who_display} learned to {verb_phrase}."
                if outcome_line not in out:
                    out = out.rstrip() + "\n\n" + outcome_line
            elif why_clean:
                outcome_line = f"In the end, {who_display} learned that {why_clean}."
                if outcome_line not in out:
                    out = out.rstrip() + "\n\n" + outcome_line

    return out


def truncate_sentences(text: str, max_sentences: int = 3) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    sentences = []
    current = []

    for ch in text:
        current.append(ch)
        if ch in ".?!":
            sent = "".join(current).strip()
            if sent:
                sentences.append(sent)
            current = []
            if len(sentences) >= max_sentences:
                break

    if not sentences:
        return text

    return " ".join(sentences)


def normalize_fullwidth(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:
            out.append(chr(cp - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def has_unbalanced_quotes(text: str) -> bool:
    if not text or not text.strip():
        return False
    count = 0
    i = 0
    while i < len(text):
        if text[i] == '"' and (i == 0 or text[i - 1] != "\\"):
            count += 1
        i += 1
    return count % 2 != 0


def has_together_they_artifact(text: str) -> bool:
    if not text or not text.strip():
        return False
    return bool(re.search(r"\bTogether\s+they\s+\w+", text, re.IGNORECASE))


_GENERIC_SENTENCE_STARTERS = frozenset({
    "Another", "After", "Before", "Then", "Now", "There", "Here", "Once",
    "First", "Second", "Finally", "Meanwhile", "However", "Still", "Yet",
    "So", "But", "And", "Or", "If", "When", "Where", "Why", "How", "What", "Who",
    "Have", "Are", "Is", "Do", "Can", "Could", "Should", "Would", "May", "Might", "Must",
    "I", "He", "She", "We", "They", "Me", "Him", "Her", "Us", "Them",
    "My", "His", "Her", "Our", "Your", "This", "That", "These", "Those",
    "Mother", "Father", "Son", "Daughter", "King", "Queen", "Count", "Princess",
    "Mr", "Mrs", "Ms", "Dr", "Lord", "Duke", "Lady", "Sir",
    "Greetings", "Together",
})


def count_ungrounded_proper_nouns(text: str, allowed: set[str] | list[str]) -> int:
    allowed_lower = {str(n).strip().lower() for n in allowed if str(n).strip()}
    if not allowed_lower:
        return 0

    expanded_allowed: set[str] = set(allowed_lower)
    for n in allowed_lower:
        if n.startswith("the "):
            expanded_allowed.add(n[4:])

    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text or "")

    ungrounded = 0
    for m in matches:
        ml = m.lower()
        if ml in expanded_allowed:
            continue

        parts = m.split()
        if len(parts) == 1:
            w = parts[0]
            wl = w.lower()

            if w in _GENERIC_SENTENCE_STARTERS or wl in STOPWORDS:
                continue

            # Count single-token names only if they're at least 4 letters.
            if len(w) < 4:
                continue

            ungrounded += 1
            continue

        # Multi-token named entities are much more likely to be real names.
        ungrounded += 1

    return ungrounded


def story_has_proper_ending(text: str) -> bool:
    if not text or not text.strip():
        return False

    s = text.strip()
    if not s:
        return False

    # Strip trailing quotes so "Yes." doesn't fail due to ending at '"'.
    s = s.rstrip().rstrip('"').rstrip("'").rstrip("”").rstrip("’")
    if not s:
        return False

    last_char = s[-1]
    # Disallow question-mark endings: many "cut off mid-dialogue" cases end on '?'.
    if last_char not in ".!":
        return False

    last_tail = s[-120:].lower()
    cutoff_cues = ("whether", "if they", "or not", "just...", "i don't know", "uncertain")
    question_start_cues = ("are we", "have we", "is it", "do we", "can we", "could we", "should we")

    if any(cue in last_tail for cue in cutoff_cues):
        return False
    if any(cue in last_tail for cue in question_start_cues):
        return False
    # Also reject if a question mark appears in the last ~25 chars.
    if "?" in last_tail[-25:]:
        return False

    return True


def detect_junk_density(text: str, window: int = 200) -> tuple[bool, str]:
    """Return (is_junk, reason) if text contains non-narrative spam."""
    if not text or len(text.split()) < 40:
        return False, ""

    words = text.split()
    total = len(words)

    comma_runs = re.findall(r"(?:\b\w+\b,\s*){5,}", text)
    if len(comma_runs) >= 2:
        return True, f"Comma-separated noun/keyword spam detected ({len(comma_runs)} runs)"

    upper_words = [w for w in words if w.isupper() and len(w) >= 2 and w.isalpha()]
    if total > 50 and len(upper_words) / total > 0.08:
        return True, f"High acronym/uppercase density ({len(upper_words)}/{total})"

    for i in range(0, total - window, window // 2):
        chunk_words = words[i : i + window]
        chunk_text = " ".join(chunk_words)
        sentences = re.findall(r"[.!?]", chunk_text)
        if len(chunk_words) > 80 and len(sentences) < 2:
            return True, f"Low sentence structure in {window}-word window (only {len(sentences)} enders)"

    _OFFTOPIC_KEYWORDS = frozenset({
        "neural", "networks", "machine", "learning", "artificial", "intelligence",
        "robotics", "algorithm", "software", "hardware", "database", "kubernetes",
        "javascript", "python", "neurotransmitters", "hormones", "adrenaline",
        "cortisol", "photosynthesis", "quantum", "blockchain", "cryptocurrency",
        "thermodynamics", "electromagnetic", "microprocessor", "semiconductor",
    })
    offtopic_count = sum(1 for w in words if w.lower().strip(",.;:!?") in _OFFTOPIC_KEYWORDS)
    if total > 50 and offtopic_count / total > 0.03:
        return True, f"Off-topic keyword density too high ({offtopic_count}/{total} technical terms)"

    return False, ""


# ── Validation ──

_ALIGNMENT_STOP = frozenset({
    "a", "an", "the", "this", "that", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "can", "may", "might", "must", "shall",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "up",
    "out", "off", "about", "into", "through", "and", "but", "or", "so",
    "if", "then", "than", "not", "no", "as", "who", "whom", "what",
    "which", "where", "when", "how", "why", "their", "they", "them",
    "its", "it", "he", "she", "his", "her", "someone", "something",
    "tries", "try", "finds", "find", "little", "strange",
})


def _extract_query_keywords(query: str) -> set[str]:
    words = {w.lower().strip(",.;:!?'\"") for w in query.split()}
    return {w for w in words if w not in _ALIGNMENT_STOP and len(w) >= 3}


def check_query_context_alignment(query: str, context: str, min_keyword_overlap: int = 1) -> tuple[bool, list[str]]:
    """Check if the retrieved context has enough overlap with the user query's key terms."""
    warnings: list[str] = []
    if not query or not context:
        return True, []

    query_keywords = _extract_query_keywords(query)
    context_lower = context.lower()
    matched = {kw for kw in query_keywords if kw in context_lower}
    unmatched = query_keywords - matched

    if len(matched) < min_keyword_overlap and query_keywords:
        warnings.append(
            f"Context may not match query: {len(matched)}/{len(query_keywords)} "
            f"query keywords found. Missing: {', '.join(sorted(unmatched)[:5])}"
        )

    if query_keywords and not matched:
        warnings.append("Zero query keywords found in context — likely misaligned retrieval")

    return len(warnings) == 0, warnings


def query_context_alignment_score(query: str, context: str) -> float:
    """Return 0.0..1.0 indicating what fraction of query keywords appear in context."""
    if not query or not context:
        return 1.0
    keywords = _extract_query_keywords(query)
    if not keywords:
        return 1.0
    ctx_lower = context.lower()
    matched = sum(1 for kw in keywords if kw in ctx_lower)
    return matched / len(keywords)


def pick_best_context_story(query: str, context_parts: list[str]) -> list[str]:
    """Given multiple '[Title by Author]\\n...' blocks, return them sorted by query relevance.
    
    The most relevant story (highest keyword overlap) comes first. Strips blocks
    with zero overlap if at least one block has overlap.
    """
    if not context_parts or not query:
        return context_parts

    query_keywords = _extract_query_keywords(query)
    if not query_keywords:
        return context_parts

    scored: list[tuple[float, int, str]] = []
    for idx, part in enumerate(context_parts):
        part_lower = part.lower()
        matched = sum(1 for kw in query_keywords if kw in part_lower)
        score = matched / len(query_keywords)
        scored.append((score, idx, part))

    scored.sort(key=lambda t: t[0], reverse=True)

    best_score = scored[0][0]
    if best_score > 0:
        filtered = [part for score, _, part in scored if score > 0]
        return filtered

    return context_parts


def check_how_coverage(layer2_output: str, how_events: list[str]) -> tuple[bool, list[str]]:
    """Check that Layer 2 sections reference the HOW events from Layer 1.
    
    Returns (all_covered, list_of_missing_event_summaries).
    """
    if not how_events or not layer2_output:
        return True, []

    l2_lower = layer2_output.lower()
    missing: list[str] = []

    for i, event in enumerate(how_events, start=1):
        event_clean = event.strip()
        if not event_clean or event_clean.lower() == "unknown":
            continue
        event_words = {
            w.lower().strip(",.;:!?'\"()")
            for w in event_clean.split()
            if len(w) > 3 and w.lower() not in _ALIGNMENT_STOP
        }
        if not event_words:
            continue
        matched = sum(1 for w in event_words if w in l2_lower)
        coverage = matched / len(event_words) if event_words else 0
        if coverage < 0.25:
            snippet = event_clean[:80] + ("..." if len(event_clean) > 80 else "")
            missing.append(f"HOW({i}): '{snippet}' ({matched}/{len(event_words)} words found)")

    return len(missing) == 0, missing


def validate_layer1(five_w_one_h: str, context: str = "") -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not five_w_one_h or not five_w_one_h.strip():
        return False, ["Layer 1 output is empty"]

    parsed = parse_5w1h_sections(five_w_one_h)
    required = {"who", "what", "where", "how"}
    for key in required:
        if key not in parsed or not parsed[key].strip():
            reasons.append(f"Missing or empty {key.upper()}")

    how_text = parsed.get("how", "")
    if how_text:
        numbered = re.findall(r"(?:^|\n)\s*(?:\(?\d+[.)]\s*)", how_text)
        events = [e.strip() for e in re.split(r"(?:^|\n)\s*\(?\d+[.)]\s*", how_text) if e.strip()]
        empty_items = re.findall(r"(?:^|\n)\s*\(?\d+[.)]\s*$", how_text, re.MULTILINE)
        if empty_items:
            reasons.append(f"HOW has {len(empty_items)} empty numbered item(s)")
        if len(events) < 3:
            reasons.append(f"HOW has only {len(events)} event(s), need at least 3")
        for ev in events:
            if len(ev.split()) < 3:
                reasons.append(f"HOW event too short: '{ev}'")

    _HALLUCINATED_RELATIONSHIPS = (
        "in-law", "in-laws", "sister-in-law", "brother-in-law",
        "mother-in-law", "father-in-law", "sisters-in-law", "brothers-in-law",
    )
    l1_lower = five_w_one_h.lower()
    ctx_lower = context.lower() if context else ""

    for term in _HALLUCINATED_RELATIONSHIPS:
        if term in l1_lower and (not context or term not in ctx_lower):
            reasons.append(f"Layer 1 introduces '{term}' not found in context (relationship hallucination)")

    when_text = parsed.get("when", "")
    if when_text:
        _SPECULATIVE_TIME_WORDS = ("winter", "summer", "spring", "autumn", "morning", "evening", "dawn", "dusk", "midnight")
        for tw in _SPECULATIVE_TIME_WORDS:
            if tw in when_text.lower() and (not context or tw not in ctx_lower):
                reasons.append(f"WHEN adds '{tw}' not found in context — use 'unspecified' if unknown")

    if context:
        if "twelve years" in ctx_lower and "twelve" not in l1_lower:
            reasons.append("Context mentions 'twelve years' but Layer 1 does not preserve it")
        if "dwarf" in ctx_lower:
            who_text = parsed.get("who", "").lower()
            how_text_lower = how_text.lower()
            if "dwarf" not in who_text and "dwarf" not in how_text_lower:
                reasons.append("Context has 'dwarf' but Layer 1 WHO/HOW does not mention it")

    return (len(reasons) == 0, reasons)


def validate_layer2_section(
    content: str,
    previous_sections: str,
    allowed_names: set[str] | None = None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not content or not content.strip():
        return False, ["Section is empty"]

    if has_together_they_artifact(content):
        reasons.append("Contains 'Together they' artifact")

    words = content.split()
    long_words = [w for w in words if len(w) > 18 and w.replace(",", "").replace(".", "").isalpha()]
    if long_words:
        reasons.append(f"Collapsed whitespace (long tokens: {', '.join(long_words[:3])})")

    if len(words) >= 10:
        long_count = sum(1 for w in words if len(w) > 25 and w.replace(",", "").replace(".", "").isalpha())
        ratio = long_count / len(words)
        if ratio > 0.05:
            reasons.append(f"High collapsed-word ratio ({long_count}/{len(words)} words >25 chars)")

    long_alpha_runs = re.findall(r"\b[a-zA-Z]{20,}\b", content)
    if len(long_alpha_runs) >= 3:
        reasons.append(f"Multiple collapsed runs ({len(long_alpha_runs)} alpha tokens >20 chars)")

    if previous_sections and previous_sections.strip():
        prev_words = set(re.findall(r"[a-z]+", previous_sections.lower()))
        cur_words = set(re.findall(r"[a-z]+", content.lower()))
        if prev_words and cur_words:
            overlap = len(cur_words & prev_words) / len(cur_words) if cur_words else 0
            if overlap > 0.85:
                reasons.append(f"Section overlaps {overlap:.0%} with previous sections (repeating same events)")

    return (len(reasons) == 0, reasons)


def validate_layer3_section(
    content: str,
    allowed_names: set[str],
    section_number: int = 0,
    total_sections: int = 5,
) -> tuple[bool, list[str]]:

    reasons: list[str] = []
    if not content or not content.strip():
        return False, ["Section is empty"]

    if has_unbalanced_quotes(content):
        reasons.append("Has unbalanced dialogue quotes")

    if has_together_they_artifact(content):
        reasons.append("Contains 'Together they' artifact")

    if allowed_names:
        ungrounded = count_ungrounded_proper_nouns(content, allowed_names)
        if ungrounded > 0:
            reasons.append(f"Section has {ungrounded} ungrounded proper nouns")

    words = content.split()
    if len(words) < 30:
        reasons.append(f"Section only has {len(words)} words (need at least 30)")

    is_junk, junk_reason = detect_junk_density(content)
    if is_junk:
        reasons.append(f"Junk/topic drift: {junk_reason}")

    if section_number == total_sections:
        if not story_has_proper_ending(content):
            reasons.append("Final section lacks a proper ending")

    return (len(reasons) == 0, reasons)
