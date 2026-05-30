import datetime
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import chromadb
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from Vector_Store.vector_store import encode_query, model as embedding_model
from Generative_AI.penalty_processors import (
    REPETITION_PENALTY, NO_REPEAT_NGRAM_SIZE, build_logits_processors,
    apply_language_gate,
    clean_generated_output,
    clean_layer2_artifacts,
    clean_layer3_story,
    extract_who_names,
    normalize_character_names,
    normalize_fullwidth,
    parse_5w1h_sections,
    truncate_sentences,
    has_unbalanced_quotes,
    has_together_they_artifact,
    count_ungrounded_proper_nouns,
    story_has_proper_ending,
    detect_junk_density,
    check_query_context_alignment,
    query_context_alignment_score,
    pick_best_context_story,
    check_how_coverage,
    validate_layer1,
    validate_layer2_section,
    validate_layer3_section,
)
from Generative_AI import choser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent.parent
config_file = ROOT_DIR / "setup.yaml"
prompts_file = ROOT_DIR / "prompts.yaml"

with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

with open(prompts_file, "r", encoding="utf-8") as file:
    prompts = yaml.safe_load(file)

BASE_PATH = config.get("BASE_PATH")
CHROMA_PATH = config.get("Chroma_path", "chroma_db")
COLLECTION_NAME = config.get("Chroma_collection_name", "English_Stories")
GENERATED_OUTPUT = config.get("Generated_story_output", "Generated_Stories")
GENERATIVE_MODEL = config.get("Generative_model")
LOAD_GENERATIVE_MODEL_IN_4BIT = config.get("Load_generative_model_in_4bit", False)
MODEL_MAX_PROMPT_TOKENS = config.get("Model_max_prompt_tokens", 4096)
STORY_GENERATION_N_RESULTS = config.get("Story_generation_n_results", 2)

GENERATION_MODE_FAST = config.get("Generation_mode_fast")
GENERATION_MODE_THINKING = config.get("Generation_mode_thinking")
GENERATION_MODE_SHORT = config.get("Generation_mode_short", "SHORT")
GENERATION_FAST_TEMPERATURE = config.get("Generation_fast_temperature", 0.5)
GENERATION_FAST_TOP_P = config.get("Generation_fast_top_p", 0.9)
GENERATION_THINKING_TEMPERATURE = config.get("Generation_thinking_temperature", 0.7)
GENERATION_THINKING_TOP_P = config.get("Generation_thinking_top_p", 0.9)

THREE_LAYER_GENERATION = config.get("Three_layer_generation", False)

SINGLE_PASS_FAST_MAX_TOKENS = config.get("Single_pass_fast_max_tokens", 768)
SINGLE_PASS_THINKING_MAX_TOKENS = config.get("Single_pass_thinking_max_tokens", 1536)
SINGLE_PASS_SHORT_MAX_TOKENS = config.get("Single_pass_short_max_tokens", 400)

LAYER1_MAX_TOKENS = config.get("Layer1_max_tokens", 600)
LAYER2_MAX_TOKENS = config.get("Layer2_max_tokens", 1024)
LAYER2_PER_SECTION = config.get("Layer2_per_section", True)
LAYER2_SECTION_MAX_TOKENS = config.get("Layer2_section_max_tokens", 280)
LAYER3_FAST_MAX_TOKENS = config.get("Layer3_fast_max_tokens", 1536)
LAYER3_THINKING_MAX_TOKENS = config.get("Layer3_thinking_max_tokens", 2048)
MIN_GENERATION_RATIO = config.get("Min_generation_ratio", 0.6)

LAYER3_SECTION_COUNT = config.get("Layer3_section_count", 5)
LAYER3_SECTION_MIN_TOKENS = config.get("Layer3_section_min_tokens", 800)
LAYER3_SECTION_MAX_TOKENS = config.get("Layer3_section_max_tokens", 1200)
LAYER3_SECTION_MIN_TOKENS_SHORT = config.get("Layer3_section_min_tokens_short", 400)
LAYER3_SECTION_MAX_TOKENS_SHORT = config.get("Layer3_section_max_tokens_short", 600)

# Layer-specific temperature/top_p (optional; fallback = get_mode_sampling(mode))
LAYER1_TEMPERATURE = config.get("Layer1_temperature")
LAYER1_TOP_P = config.get("Layer1_top_p")
LAYER2_TEMPERATURE = config.get("Layer2_temperature")
LAYER2_TOP_P = config.get("Layer2_top_p")
LAYER3_TEMPERATURE = config.get("Layer3_temperature")
LAYER3_TOP_P = config.get("Layer3_top_p")

GEN_PROMPTS = prompts.get("generation", {})

SUMMARY_SYSTEM_PROMPT = GEN_PROMPTS.get("summary_system", "")
SUMMARY_USER_PROMPT = GEN_PROMPTS.get("summary_user", "")
FULL_STORY_SYSTEM_PROMPT = GEN_PROMPTS.get("full_story_system", "")
FULL_STORY_USER_PROMPT = GEN_PROMPTS.get("full_story_user", "")

LAYER1_SYSTEM_PROMPT = GEN_PROMPTS.get("layer1_5w1h_system", "")
LAYER1_USER_PROMPT = GEN_PROMPTS.get("layer1_5w1h_user", "")
LAYER1_STYLE_ADDON = GEN_PROMPTS.get("layer1_style_addon", "")

LAYER2_SYSTEM_PROMPT = GEN_PROMPTS.get("layer2_summary_system", "")
LAYER2_USER_PROMPT = GEN_PROMPTS.get("layer2_summary_user", "")
LAYER2_ONE_SECTION_SYSTEM = GEN_PROMPTS.get("layer2_one_section_system", "")
LAYER2_ONE_SECTION_USER = GEN_PROMPTS.get("layer2_one_section_user", "")

LAYER3_SYSTEM_PROMPT = GEN_PROMPTS.get("layer3_story_system", "")
LAYER3_USER_PROMPT = GEN_PROMPTS.get("layer3_story_user", "")
LAYER3_SECTION_SYSTEM_PROMPT = GEN_PROMPTS.get("layer3_section_system", "")
LAYER3_SECTION_USER_PROMPT = GEN_PROMPTS.get("layer3_section_user", "")

class Gen_mode(str, Enum):
    FAST = GENERATION_MODE_FAST
    THINKING = GENERATION_MODE_THINKING
    SHORT = GENERATION_MODE_SHORT

class StoryType(str, Enum):
    SINGLE = "single"
    SERIES = "series"
    MIX = "mix"

class _SafeFormatDict(dict):
    def __missing__(self, key):
        return ""

def _mode_to_str(mode: Union["Gen_mode", str, None]) -> str:
    if mode is None:
        return "fast"
    if isinstance(mode, Gen_mode):
        return (mode.value or "fast").lower()
    return (mode or "fast").lower()

def parse_gen_mode(mode: Optional[str]) -> Gen_mode:
    if not mode:
        return Gen_mode.FAST
    m = str(mode).lower()
    if m == "thinking":
        return Gen_mode.THINKING
    if m == "short":
        return Gen_mode.SHORT
    return Gen_mode.FAST

def parse_story_type(story_type: Optional[str]) -> StoryType:
    if not story_type:
        return StoryType.MIX
    t = str(story_type).strip().lower()
    if t == "series":
        return StoryType.SERIES
    if t == "single":
        return StoryType.SINGLE
    return StoryType.MIX

def get_generation_mode(mode: Union[Gen_mode, str, None] = Gen_mode.FAST):
    mode_str = _mode_to_str(mode)
    if mode_str == "thinking":
        return GENERATION_MODE_THINKING
    return GENERATION_MODE_FAST

def get_mode_sampling(mode: Union[Gen_mode, str, None] = Gen_mode.FAST):
    if _mode_to_str(mode) == "thinking":
        return GENERATION_THINKING_TEMPERATURE, GENERATION_THINKING_TOP_P
    return GENERATION_FAST_TEMPERATURE, GENERATION_FAST_TOP_P

def get_layer_sampling(layer: int, mode: Union[Gen_mode, str, None] = Gen_mode.FAST):
    fallback_temp, fallback_top_p = get_mode_sampling(mode)
    if layer == 1:
        return (LAYER1_TEMPERATURE if LAYER1_TEMPERATURE is not None else fallback_temp,
                LAYER1_TOP_P if LAYER1_TOP_P is not None else fallback_top_p)
    if layer == 2:
        return (LAYER2_TEMPERATURE if LAYER2_TEMPERATURE is not None else fallback_temp,
                LAYER2_TOP_P if LAYER2_TOP_P is not None else fallback_top_p)
    if layer == 3:
        return (LAYER3_TEMPERATURE if LAYER3_TEMPERATURE is not None else fallback_temp,
                LAYER3_TOP_P if LAYER3_TOP_P is not None else fallback_top_p)
    return fallback_temp, fallback_top_p

def get_layer_max_tokens(layer: int, mode: Union[Gen_mode, str, None] = Gen_mode.FAST):
    mode_str = _mode_to_str(mode)
    if layer == 0:
        if mode_str == "thinking":
            return SINGLE_PASS_THINKING_MAX_TOKENS
        if mode_str == "short":
            return SINGLE_PASS_SHORT_MAX_TOKENS
        return SINGLE_PASS_FAST_MAX_TOKENS
    if layer == 1:
        return LAYER1_MAX_TOKENS
    if layer == 2:
        return LAYER2_MAX_TOKENS
    return LAYER3_THINKING_MAX_TOKENS if mode_str == "thinking" else LAYER3_FAST_MAX_TOKENS


def get_layer3_token_limits(mode: Union[Gen_mode, str, None] = None) -> tuple[int, int]:
    if _mode_to_str(mode) == "short":
        return LAYER3_SECTION_MIN_TOKENS_SHORT, LAYER3_SECTION_MAX_TOKENS_SHORT
    return LAYER3_SECTION_MIN_TOKENS, LAYER3_SECTION_MAX_TOKENS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
logging.info(
    f"Generation config: model={GENERATIVE_MODEL}, "
    f"max_prompt_tokens={MODEL_MAX_PROMPT_TOKENS}, "
    f"n_results={STORY_GENERATION_N_RESULTS}, "
    f"three_layer={THREE_LAYER_GENERATION}"
)

def get_chroma_client():
    try:
        chroma_full_path = Path(BASE_PATH) / CHROMA_PATH
        chroma_client = chromadb.PersistentClient(path=str(chroma_full_path))
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        logging.info(f"Connected to ChromaDB collection: {CHROMA_PATH}/{COLLECTION_NAME}")
        return chroma_client, collection
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {e}")
        raise

_model_cache: dict = {}


def get_generative_model(generative_model):
    if generative_model in _model_cache:
        return _model_cache[generative_model]
    try:
        logging.info(f"Loading generative model: {generative_model} (4bit={LOAD_GENERATIVE_MODEL_IN_4BIT})")
        use_4bit = bool(LOAD_GENERATIVE_MODEL_IN_4BIT and device.type == "cuda")
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    generative_model,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            except ImportError:
                logging.warning("bitsandbytes not installed; falling back to full precision. pip install bitsandbytes")
                use_4bit = False
        if not use_4bit:
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                generative_model,
                torch_dtype=dtype,
            )
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(generative_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Loaded generative model: {generative_model}")
        _model_cache[generative_model] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading generative model and tokenizer: {e}")
        raise

def _query_collection(query_text: str, n_results: int = 5, query_type: str = "content", story_type: Optional[StoryType] = None):
    _, collection = get_chroma_client()
    query_embedding = encode_query(embedding_model, query_text)

    if story_type == StoryType.SERIES:
        where = {"$and": [{"query_type": query_type}, {"Is_series": True}]}
    elif story_type == StoryType.SINGLE:
        where = {"$and": [{"query_type": query_type}, {"Is_series": False}]}
    else:
        where = {"query_type": query_type}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["distances", "documents", "metadatas", "embeddings"],
    )

    logging.info(f"Found {len(results['documents'][0])} results for query: {query_text} (story_type={story_type})")
    return results

def query_summary_searching(query_text: str, n_results: int = 5, query_type: str = "summary", story_type: Optional[StoryType] = None):
    return _query_collection(query_text, n_results, query_type, story_type=story_type)

def query_content_searching(query_text: str, n_results: int = 5, query_type: str = "content", story_type: Optional[StoryType] = None):
    return _query_collection(query_text, n_results, query_type, story_type=story_type)

def build_context(query_results: dict, query_type: str = "summary", query: str = ""):
    context_parts = []
    for doc, metadata in zip(query_results["documents"][0], query_results["metadatas"][0]):
        title = metadata.get('Title', 'Unknown')
        author = metadata.get('Author', 'Unknown')
        summary = metadata.get('Summary', 'Unknown')

        if query_type == "summary":
            context_parts.append(f"[{title} by {author}]\n{summary}")
        else:
            context_parts.append(f"[{title} by {author}]\n{doc}")

    if query and len(context_parts) > 1:
        original_count = len(context_parts)
        context_parts = pick_best_context_story(query, context_parts)
        if len(context_parts) < original_count:
            logging.info(
                "build_context: filtered %d → %d stories by query relevance",
                original_count, len(context_parts),
            )

    return "\n\n".join(context_parts)

def build_layer2_reference(query_results: dict, tokenizer, max_opening_tokens: int = 300):
    summaries, openings = [], []
    for doc, meta in zip(
        query_results.get("documents", [[]])[0],
        query_results.get("metadatas", [[]])[0],
    ):
        title = meta.get("Title", "Unknown")
        summary = meta.get("Summary", "")
        if summary:
            summaries.append(f"[{title}] {summary}")
        if doc:
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            if len(tokens) > max_opening_tokens:
                tokens = tokens[:max_opening_tokens]
                doc = tokenizer.decode(tokens, skip_special_tokens=True)
            openings.append(f"[{title}]\n{doc}")
    return "\n\n".join(summaries), "\n\n".join(openings)


# ── 5W1H helpers ──

def _split_how_events(how_text: str) -> list[str]:
    if not how_text:
        return []
    parts = re.split(r"\(\d+\)\s*", how_text)
    return [p.strip() for p in parts if p.strip()]


def _assign_how_slices(labels: list[str], how_events: list[str]) -> dict[int, str]:
    how_section_indices = []
    outcome_index = None
    for i, label in enumerate(labels):
        upper = label.upper()
        if "HOW" in upper or "TWIST" in upper or "COMPLICATION" in upper:
            how_section_indices.append(i)
        if ("OUTCOME" in upper or "WHY" in upper) and "HOW" not in upper:
            outcome_index = i

    if not how_section_indices or not how_events:
        return {}

    resolution_event = None
    action_events = how_events
    if outcome_index is not None and len(how_events) >= 3:
        resolution_event = how_events[-1]
        action_events = how_events[:-1]

    assignments: dict[int, str] = {}
    n_sections = len(how_section_indices)
    n_events = len(action_events)
    events_per = max(1, n_events // n_sections) if n_sections > 0 else n_events

    for idx, sec_i in enumerate(how_section_indices):
        start = idx * events_per
        if idx == n_sections - 1:
            end = n_events
        else:
            end = start + events_per
        slice_events = action_events[start:end]
        numbered = [f"({start + j + 1}) {ev}" for j, ev in enumerate(slice_events)]
        assignments[sec_i] = " ".join(numbered)

    if resolution_event is not None and outcome_index is not None:
        assignments[outcome_index] = f"(Resolution) {resolution_event}"

    return assignments


def _extract_5w1h_for_section(
    label: str,
    parsed_5w1h: dict[str, str],
    how_override: str | None = None,
) -> str:
    label_upper = label.upper()
    _KEY_MAP = {
        "WHO": "who",
        "WHERE": "where",
        "WHEN": "when",
        "WHAT": "what",
        "WHY": "why",
        "OUTCOME": "why",
        "HOW": "how",
        "TWIST": "how",
        "COMPLICATION": "how",
    }
    relevant_parts = []
    for keyword, key in _KEY_MAP.items():
        if keyword in label_upper and key in parsed_5w1h:
            if key == "how" and how_override is not None:
                relevant_parts.append(f"{keyword}: {how_override}")
            else:
                relevant_parts.append(f"{keyword}: {parsed_5w1h[key]}")

    if not relevant_parts:
        return "Use the full 5W1H to write this section."
    return "\n".join(relevant_parts)


def _character_names_for_constraints(five_w_one_h: str) -> str:
    names = extract_who_names(five_w_one_h)
    if not names:
        return ""
    return ", ".join(names)


def truncate_context(context: str, system_prompt: str, query: str, tokenizer, enable_thinking: bool = False, user_prompt_template: str = "") -> str:
    template = user_prompt_template or FULL_STORY_USER_PROMPT
    placeholder_user = template.format_map(_SafeFormatDict(context="", query=query))
    overhead_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": placeholder_user},
    ]
    try:
        overhead_text = tokenizer.apply_chat_template(
            overhead_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
    except TypeError:
        overhead_text = tokenizer.apply_chat_template(
            overhead_msgs, tokenize=False, add_generation_prompt=True,
        )
    overhead_tokens = len(tokenizer.encode(overhead_text, add_special_tokens=False))
    context_budget = MODEL_MAX_PROMPT_TOKENS - overhead_tokens - 32

    if context_budget <= 0:
        logging.warning(f"No token budget left for context (overhead={overhead_tokens}, max={MODEL_MAX_PROMPT_TOKENS})")
        return ""

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(context_ids) > context_budget:
        logging.info(f"Context truncated: {len(context_ids)} -> {context_budget} tokens (overhead={overhead_tokens})")
        context_ids = context_ids[:context_budget]
        context = tokenizer.decode(context_ids, skip_special_tokens=True)

    return context

def parse_sections(layer2_output: str):
    normalized = normalize_fullwidth(layer2_output)
    normalized = re.sub(r"SECTION\s+III\s*:", "SECTION 3:", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"SECTION\s+II\s*:", "SECTION 2:", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"SECTION\s+IV\s*:", "SECTION 4:", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"SECTION\s+V\s*:", "SECTION 5:", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"SECTION\s+I\b\s*:", "SECTION 1:", normalized, flags=re.IGNORECASE)

    patterns = [
        r'\[\s*SECTION\s*(\d+)\s*:\s*([^\]]+)\]\s*',
        r'\[\s*Section\s*(\d+)\s*:\s*([^\]]+)\]\s*',
        r'\[\s*SECTION\s+Section\s*(\d+)\s*:\s*([^\]]+)\]\s*',
        r'\*\*Section\s+(\d+):\s*([^*]+)\*\*\s*',
        r'(?m)^Section\s+(\d+):\s*(.+)$',
    ]

    for pattern in patterns:
        parts = re.split(pattern, normalized)
        if len(parts) >= 4:
            sections = []
            for i in range(1, len(parts) - 2, 3):
                label = parts[i + 1].strip()
                text = (parts[i + 2] or "").strip()
                sections.append((label, text))
            if sections:
                logging.info(f"parse_sections: matched {len(sections)} sections")
                return sections

    logging.warning(f"parse_sections: no pattern matched. Output preview: {normalized[:200]!r}")
    return []

def generate_response(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 3200,
    min_tokens: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = False,
    clean_output: bool = True,
):
    model, tokenizer = get_generative_model(GENERATIVE_MODEL)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    max_prompt = MODEL_MAX_PROMPT_TOKENS
    tokenizer.truncation_side = "right"
    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_prompt,
        truncation=True,
    ).to(device)
    if model_inputs.input_ids.shape[1] >= max_prompt:
        logging.warning(f"Prompt truncated to {max_prompt} tokens (right-side fallback)")

    prompt_length = model_inputs.input_ids.shape[1]
    logging.info(
        f"Generating response (prompt_tokens={prompt_length}, "
        f"max_new_tokens={max_tokens}, min_new_tokens={min_tokens}, "
        f"temperature={temperature})"
    )
    processors = build_logits_processors(prompt_length, tokenizer)
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": REPETITION_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "logits_processor": processors if len(processors) > 0 else None,
    }
    if min_tokens > 0:
        gen_kwargs["min_new_tokens"] = min_tokens

    try:
        generated_ids = model.generate(**model_inputs, **gen_kwargs)
    except torch.cuda.OutOfMemoryError:
        logging.warning(
            "CUDA OOM during generate (prompt_tokens=%d, max_new=%d). "
            "Halving context and retrying on CPU.",
            prompt_length, max_tokens,
        )
        torch.cuda.empty_cache()
        halved_len = max(prompt_length // 2, 128)
        model_inputs_cpu = {
            k: v[:, :halved_len].to("cpu") for k, v in model_inputs.items()
        }
        gen_kwargs.pop("logits_processor", None)
        if "min_new_tokens" in gen_kwargs:
            gen_kwargs["min_new_tokens"] = min(gen_kwargs["min_new_tokens"], max_tokens // 2)
        try:
            model_cpu = model.to("cpu")
            generated_ids = model_cpu.generate(**model_inputs_cpu, **gen_kwargs)
            model.to(device)
        except Exception as cpu_err:
            model.to(device)
            logging.error("CPU fallback also failed: %s", cpu_err)
            return ""

    output_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    raw_len = len(response.split())
    if clean_output:
        response = clean_generated_output(response)
        cleaned_len = len(response.split())
        if cleaned_len < raw_len * 0.5:
            logging.warning(f"Cleanup removed >50% of output ({raw_len} → {cleaned_len} words)")
    else:
        response = response.strip()

    logging.info("Generation complete")
    return response

def generate_full_story(query: str, n_results: Optional[int] = None, mode: Union[Gen_mode, str, None] = Gen_mode.FAST, extract_style: bool = False, story_type: StoryType = StoryType.MIX):
    if n_results is None:
        n_results = STORY_GENERATION_N_RESULTS
    results = query_content_searching(query, n_results=n_results, query_type="content", story_type=story_type)
    context = build_context(results, query_type="content", query=query)

    use_thinking = _mode_to_str(mode) == "thinking"
    _, tokenizer = get_generative_model(GENERATIVE_MODEL)

    logging.info(f"Story type filter: {story_type.value}")

    if THREE_LAYER_GENERATION:
        return generate_three_layer(query, context, mode, use_thinking, tokenizer, query_results=results, extract_style=extract_style)
    return generate_single_pass(query, context, mode, use_thinking, tokenizer)


def generate_single_pass(query: str, context: str, mode, use_thinking: bool, tokenizer):
    system_prompt = FULL_STORY_SYSTEM_PROMPT
    temperature, top_p = get_mode_sampling(mode)
    max_tokens = get_layer_max_tokens(0, mode)
    context = truncate_context(context, system_prompt, query, tokenizer, enable_thinking=use_thinking)
    user_prompt = FULL_STORY_USER_PROMPT.format(context=context, query=query)

    return generate_response(
        system_prompt, user_prompt,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        enable_thinking=use_thinking,
    )

# ── Layer 1 ──

def _run_layer1(query, context, mode, use_thinking, tokenizer, temperature, top_p, extract_style):
    l1_system = LAYER1_SYSTEM_PROMPT
    if extract_style and LAYER1_STYLE_ADDON:
        l1_system += "\n" + LAYER1_STYLE_ADDON
        logging.info("Layer 1: STYLE extraction enabled")

    alignment_score = query_context_alignment_score(query, context)
    aligned, alignment_warnings = check_query_context_alignment(query, context)
    if not aligned:
        for w in alignment_warnings:
            logging.warning("Layer 1 query-context alignment: %s", w)
    if alignment_score == 0.0:
        logging.warning(
            "Layer 1: zero query-context alignment (score=%.2f). "
            "Context does not contain any query keywords — returning insufficient-context signal.",
            alignment_score,
        )
        return (
            "WHO:\nUnknown — context does not match the query.\n"
            "WHAT:\nInsufficient context to extract events.\n"
            "WHEN:\nUnknown\n"
            "WHERE:\nUnknown\n"
            "WHY:\nUnknown\n"
            "HOW:\n(1) Unknown (2) Unknown (3) Unknown (4) Unknown"
        )

    logging.info("Layer 1: generating 5W1H from context (alignment=%.0f%%)", alignment_score * 100)
    l1_context = truncate_context(
        context, l1_system, query, tokenizer,
        enable_thinking=use_thinking, user_prompt_template=LAYER1_USER_PROMPT,
    )
    l1_user = LAYER1_USER_PROMPT.format(context=l1_context, query=query)

    five_w_one_h = generate_response(
        l1_system, l1_user,
        max_tokens=get_layer_max_tokens(1, mode), min_tokens=0,
        temperature=temperature, top_p=top_p,
        enable_thinking=use_thinking,
        clean_output=False,
    )
    five_w_one_h = apply_language_gate(five_w_one_h or "")

    is_valid, reasons = validate_layer1(five_w_one_h, context)
    if not is_valid:
        logging.warning("Layer 1 validation failed: %s — retrying with lower temperature", "; ".join(reasons))
        retry_temp = max(0.05, float(temperature) * 0.5)
        five_w_one_h_retry = generate_response(
            l1_system, l1_user,
            max_tokens=get_layer_max_tokens(1, mode), min_tokens=0,
            temperature=retry_temp, top_p=top_p,
            enable_thinking=use_thinking,
            clean_output=False,
        )
        five_w_one_h_retry = apply_language_gate(five_w_one_h_retry or "")
        is_valid_retry, reasons_retry = validate_layer1(five_w_one_h_retry, context)
        if is_valid_retry or len(reasons_retry) < len(reasons):
            five_w_one_h = five_w_one_h_retry
            if not is_valid_retry:
                logging.warning("Layer 1 retry still has issues: %s (using anyway — fewer problems)", "; ".join(reasons_retry))
            else:
                logging.info("Layer 1 retry passed validation")
        else:
            logging.warning("Layer 1 retry did not improve; keeping original output")

    if five_w_one_h and five_w_one_h.strip():
        logging.info(f"Layer 1 done ({len(tokenizer.encode(five_w_one_h, add_special_tokens=False))} tokens)")
    return five_w_one_h

# ── Layer 2 ──
def _run_layer2(query, five_w_one_h, mode, use_thinking, tokenizer, temperature, top_p, query_results, flow_structure=None):
    if flow_structure is None:
        flow_structure = choser.get_default_structure()
    section_headers, section_mapping = choser.build_section_headers_and_mapping(flow_structure)
    expected_labels = (flow_structure or {}).get("sections") or []
    if len(expected_labels) != 5:
        expected_labels = (choser.get_default_structure() or {}).get("sections", [])[:5]
    character_names = _character_names_for_constraints(five_w_one_h)
    logging.info("Layer 2: generating sectioned story summary from 5W1H (structure: %s)", flow_structure.get("name", "?"))

    if LAYER2_PER_SECTION and len(expected_labels) == 5:
        section_max = LAYER2_SECTION_MAX_TOKENS
        parsed_5w1h = parse_5w1h_sections(five_w_one_h)
        how_events = _split_how_events(parsed_5w1h.get("how", ""))
        how_slices = _assign_how_slices(expected_labels, how_events) if how_events else {}
        if how_slices:
            logging.info("Layer 2: split %d HOW events across %d sections", len(how_events), len(how_slices))
        one_fmt = _SafeFormatDict(five_w_one_h=five_w_one_h, query=query, character_names=character_names)
        blocks = []
        previous_sections_context = ""
        for i, label in enumerate(expected_labels, start=1):
            how_override = how_slices.get(i - 1)
            relevant_5w1h = _extract_5w1h_for_section(label, parsed_5w1h, how_override=how_override)
            one_fmt["section_index"] = i
            one_fmt["section_label"] = label
            one_fmt["relevant_5w1h"] = relevant_5w1h
            one_fmt["previous_sections"] = previous_sections_context or "None (this is the first section)."
            one_fmt["total_sections"] = 5
            l2_one_sys = LAYER2_ONE_SECTION_SYSTEM.format_map(one_fmt)
            l2_one_user = LAYER2_ONE_SECTION_USER.format_map(one_fmt)
            def _generate_l2_section():
                resp = generate_response(
                    l2_one_sys, l2_one_user,
                    max_tokens=section_max, min_tokens=0,
                    temperature=temperature, top_p=top_p,
                    enable_thinking=use_thinking,
                    clean_output=False,
                )
                return apply_language_gate((resp or "").strip())

            try:
                content = _generate_l2_section()
            except Exception as exc:
                logging.exception("Layer 2: section %d generation failed (%s): %s", i, label, exc)
                content = ""

            if content.strip():
                sec_valid, sec_reasons = validate_layer2_section(content, previous_sections_context)
                if not sec_valid:
                    logging.warning("Layer 2 section %d failed validation (%s) — retrying", i, "; ".join(sec_reasons))
                    try:
                        content_retry = _generate_l2_section()
                        sec_valid_r, sec_reasons_r = validate_layer2_section(content_retry, previous_sections_context)
                        if sec_valid_r or len(sec_reasons_r) < len(sec_reasons):
                            content = content_retry
                            if not sec_valid_r:
                                logging.warning("Layer 2 section %d retry still has issues: %s", i, "; ".join(sec_reasons_r))
                    except Exception:
                        logging.warning("Layer 2 section %d retry failed; keeping original", i)

            if len(content.strip()) < 20:
                logging.warning("Layer 2: section %d output too short; using placeholder", i)
                content = "To be expanded."

            lines = [ln for ln in content.splitlines() if not re.match(r"^\s*\[?\s*SECTION\s+\d+\s*:", ln, re.IGNORECASE)]
            content = " ".join(l.strip() for l in lines if l.strip()).strip() or "To be expanded."
            content = truncate_sentences(content, max_sentences=3)
            blocks.append(f"[SECTION {i}: {label}]\n\n{content}")
            previous_sections_context += f"Section {i} ({label}): {content}\n"
        story_summary = "\n\n\n".join(blocks)
        logging.info("Layer 2: generated 5 sections (per-section mode)")
    else:
        l2_max = get_layer_max_tokens(2, mode)
        l2_min = int(l2_max * MIN_GENERATION_RATIO)
        logging.info("Layer 2 single-shot: using only Layer 1 + query (raw retrieval context dropped)")
        fmt = _SafeFormatDict(
            five_w_one_h=five_w_one_h, query=query,
            reference_summaries="", reference_openings="",
            section_headers=section_headers, section_mapping=section_mapping,
            structure_name=flow_structure.get("name", "The Standard (Linear)"),
            character_names=character_names,
        )
        l2_system = LAYER2_SYSTEM_PROMPT.format_map(fmt)
        l2_user = LAYER2_USER_PROMPT.format_map(fmt)
        story_summary = generate_response(
            l2_system, l2_user,
            max_tokens=l2_max, min_tokens=l2_min,
            temperature=temperature, top_p=top_p,
            enable_thinking=use_thinking,
            clean_output=False,
        )
        story_summary = apply_language_gate(story_summary or "")
        if story_summary and story_summary.strip():
            sections = parse_sections(story_summary)
            if sections:
                n_before = len(sections)
                if len(expected_labels) == 5 and n_before < 5:
                    for i in range(n_before, 5):
                        sections.append((expected_labels[i], "To be expanded."))
                    logging.info("Layer 2: padded from %d to 5 sections", n_before)
                if len(sections) > 5:
                    sections = sections[:5]
                trimmed_sections = []
                for label, text in sections:
                    trimmed_sections.append((label, truncate_sentences(text, max_sentences=3) if text else ""))
                rebuilt_lines = []
                for idx, (label, text) in enumerate(trimmed_sections, start=1):
                    rebuilt_lines.append(f"[SECTION {idx}: {label}]")
                    rebuilt_lines.append(text or "To be expanded.")
                    rebuilt_lines.append("")
                story_summary = "\n\n".join(rebuilt_lines).strip()
                logging.info("Layer 2 post-processed into %d sections (trimmed to max 3 sentences each)", len(trimmed_sections))

    if story_summary and story_summary.strip():
        normalized = normalize_character_names(story_summary, five_w_one_h)
        if normalized != story_summary:
            logging.info("Layer 2: normalized character names based on WHO from Layer 1")
            story_summary = normalized
        story_summary = clean_layer2_artifacts(story_summary)

        parsed_5w1h_final = parse_5w1h_sections(five_w_one_h)
        how_events_final = _split_how_events(parsed_5w1h_final.get("how", ""))
        if how_events_final:
            covered, missing_events = check_how_coverage(story_summary, how_events_final)
            if not covered:
                logging.warning(
                    "Layer 2 HOW coverage gap: %d/%d HOW events not reflected in sections: %s",
                    len(missing_events), len(how_events_final),
                    "; ".join(missing_events[:3]),
                )
            else:
                logging.info("Layer 2 HOW coverage: all %d events reflected", len(how_events_final))

        logging.info("Layer 2 done (%d tokens)", len(tokenizer.encode(story_summary, add_special_tokens=False)))
    return story_summary or ""

def _clean_and_check(chunk_raw: str) -> tuple[str, int, int]:
    chunk_raw = (chunk_raw or "").strip()
    raw_words = len(chunk_raw.split())
    if not chunk_raw:
        return "", raw_words, 0
    chunk_clean = clean_generated_output(chunk_raw, aggressive=False)
    cleaned_words = len(chunk_clean.split())
    return chunk_clean.strip(), raw_words, cleaned_words


# ── Layer 3 ──
def _run_layer3(query, story_summary, five_w_one_h, use_thinking, tokenizer, temperature, top_p, mode=None):
    if "[SECTION" not in story_summary.upper():
        logging.warning(
            "Layer 3 regression guard: Layer 2 output has no [SECTION markers. "
            "This usually means parse_sections received the wrong file (e.g. structure definition). "
            "Preview: %s", story_summary[:200],
        )

    sections = parse_sections(story_summary)
    if not sections:
        logging.warning("Layer 2 output could not be parsed into sections — treating as single section")
        sections = [("Full Story", story_summary)]

    section_min_tokens, section_max_tokens = get_layer3_token_limits(mode)
    logging.info(
        "Layer 3: expanding %d sections (mode=%s, min=%d, max=%d tokens/section)",
        len(sections), _mode_to_str(mode), section_min_tokens, section_max_tokens,
    )

    story_parts = []
    previous_ending = ""
    character_names = _character_names_for_constraints(five_w_one_h)
    names_display = character_names or "the characters already established in the story"
    l3_section_system = LAYER3_SECTION_SYSTEM_PROMPT.format_map(
        _SafeFormatDict(character_names=names_display)
    )

    _allowed_names: set[str] = set()
    if five_w_one_h:
        _who = extract_who_names(five_w_one_h)
        _allowed_names = {n.strip() for n in _who if n.strip()}
    _allowed_names.update({"the merchant", "the son", "the dwarf", "the princess",
                           "merchant", "son", "dwarf", "princess"})

    for i, (label, section_text) in enumerate(sections):
        is_final_section = (i + 1 == len(sections))
        logging.info(f"Layer 3: generating section {i + 1}/{len(sections)} — {label}")
        l3_user = LAYER3_SECTION_USER_PROMPT.format_map(_SafeFormatDict(
            section_summary=section_text,
            previous_ending=previous_ending,
            section_number=str(i + 1),
            total_sections=str(len(sections)),
            query=query,
            character_names=names_display,
        ))
        if is_final_section:
            l3_user += "\n\nCRITICAL: This is the FINAL section. You MUST resolve the main conflict and end with a definitive concluding sentence. The story must feel complete."

        section_prose_raw = generate_response(
            l3_section_system, l3_user,
            max_tokens=section_max_tokens,
            min_tokens=section_min_tokens,
            temperature=temperature, top_p=top_p,
            enable_thinking=use_thinking,
            clean_output=False,
        )

        if section_prose_raw and section_prose_raw.strip():
            l3_valid, l3_reasons = validate_layer3_section(
                section_prose_raw, _allowed_names,
                section_number=i + 1, total_sections=len(sections),
            )
            if not l3_valid:
                logging.warning("Layer 3 section %d failed validation (%s) — retrying", i + 1, "; ".join(l3_reasons))
                retry_temp = max(0.1, float(temperature) * 0.7)
                section_prose_retry = generate_response(
                    l3_section_system, l3_user,
                    max_tokens=section_max_tokens,
                    min_tokens=section_min_tokens,
                    temperature=retry_temp, top_p=top_p,
                    enable_thinking=use_thinking,
                    clean_output=False,
                )
                if section_prose_retry and section_prose_retry.strip():
                    l3_valid_r, l3_reasons_r = validate_layer3_section(
                        section_prose_retry, _allowed_names,
                        section_number=i + 1, total_sections=len(sections),
                    )
                    if l3_valid_r or len(l3_reasons_r) < len(l3_reasons):
                        section_prose_raw = section_prose_retry
                        if not l3_valid_r:
                            logging.warning("Layer 3 section %d retry still has issues: %s", i + 1, "; ".join(l3_reasons_r))
                        else:
                            logging.info("Layer 3 section %d retry passed validation", i + 1)

        if is_final_section and section_prose_raw and not story_has_proper_ending(section_prose_raw):
            logging.warning("Layer 3 final section lacks proper ending — retrying once with lower temperature")
            retry_temp = max(0.1, float(temperature) * 0.7)
            section_prose_retry = generate_response(
                l3_section_system, l3_user,
                max_tokens=section_max_tokens,
                min_tokens=section_min_tokens,
                temperature=retry_temp, top_p=top_p,
                enable_thinking=use_thinking,
                clean_output=False,
            )
            if section_prose_retry and story_has_proper_ending(section_prose_retry):
                section_prose_raw = section_prose_retry

        chunk, raw_words, cleaned_words = _clean_and_check(section_prose_raw)
        if raw_words > 0 and cleaned_words < raw_words * 0.5:
            logging.warning(
                "Layer 3 cleanup cut >50%% of section %d (%d → %d words); "
                "retrying once with lower temperature",
                i + 1, raw_words, cleaned_words,
            )
            retry_temp = max(0.1, float(temperature) * 0.7)
            section_prose_retry = generate_response(
                l3_section_system, l3_user,
                max_tokens=section_max_tokens,
                min_tokens=section_min_tokens,
                temperature=retry_temp, top_p=top_p,
                enable_thinking=use_thinking,
                clean_output=False,
            )
            chunk, raw_words_retry, cleaned_words_retry = _clean_and_check(section_prose_retry)
            if raw_words_retry > 0 and cleaned_words_retry < raw_words_retry * 0.5:
                logging.warning(
                    "Layer 3 retry for section %d still degraded (>50%% cut, %d → %d words); "
                    "keeping content and continuing with remaining sections so Layer 3 is longer than Layer 2 and meet the target words per section",
                    i + 1, raw_words_retry, cleaned_words_retry,
                )

        if chunk:
            story_parts.append(chunk)
            paragraphs = [p.strip() for p in chunk.split("\n\n") if p.strip()]
            # Keep up to the last two paragraphs as transition context for the next section.
            if len(paragraphs) >= 2:
                previous_ending = "\n\n".join(paragraphs[-2:])
            else:
                previous_ending = paragraphs[-1] if paragraphs else ""
        else:
            logging.warning(f"Layer 3 section {i + 1} produced empty output after cleanup — skipping")

    return story_parts

# ── Three-layer orchestrator ──
def generate_three_layer(query: str, context: str, mode, use_thinking: bool, tokenizer, query_results=None, extract_style: bool = False):
    temp1, top_p1 = get_layer_sampling(1, mode)
    temp2, top_p2 = get_layer_sampling(2, mode)
    temp3, top_p3 = get_layer_sampling(3, mode)

    # Layer 1 (Facts) — precise
    five_w_one_h = _run_layer1(query, context, mode, use_thinking, tokenizer, temp1, top_p1, extract_style)
    if not five_w_one_h or not five_w_one_h.strip():
        logging.warning("Layer 1 produced empty output — falling back to single-pass")
        return generate_single_pass(query, context, mode, use_thinking, tokenizer)

    flow_structure = choser.choose_flow_structure(mode=mode)
    logging.info("Chosen structure: %s (flow: %s)", flow_structure.get("name"), flow_structure.get("flow"))

    # Layer 2 (Flow) — map 5W1H into the chosen section order
    story_summary = _run_layer2(query, five_w_one_h, mode, use_thinking, tokenizer, temp2, top_p2, query_results, flow_structure=flow_structure)
    if not story_summary or not story_summary.strip():
        logging.warning("Layer 2 produced empty output — falling back to single-pass")
        return generate_single_pass(query, context, mode, use_thinking, tokenizer)

    # Layer 3 (Style) — expressive
    story_parts = _run_layer3(query, story_summary, five_w_one_h, use_thinking, tokenizer, temp3, top_p3, mode=mode)
    if not story_parts:
        logging.warning("Layer 3 produced no sections — falling back to single-pass")
        return generate_single_pass(query, context, mode, use_thinking, tokenizer)

    allowed_names = set()
    if five_w_one_h:
        who_names = extract_who_names(five_w_one_h)
        allowed_names = {n.strip() for n in who_names if n.strip()}
        allowed_names.update(("the merchant", "the son", "the dwarf", "the princess", "merchant", "son", "dwarf", "princess"))

    MAX_LAYER3_RETRIES = 1
    best_story = None
    best_issues = 999
    best_story_parts = story_parts

    for attempt in range(1 + MAX_LAYER3_RETRIES):
        if attempt > 0:
            logging.warning("Layer 3 reject+retry attempt %d/%d with lower temperature", attempt, MAX_LAYER3_RETRIES)
            retry_temp = max(0.1, float(temp3) * 0.6)
            story_parts = _run_layer3(query, story_summary, five_w_one_h, use_thinking, tokenizer, retry_temp, top_p3, mode=mode)
            if not story_parts:
                logging.warning("Layer 3 retry produced no sections")
                continue

        story = "\n\n".join(story_parts)
        story = clean_layer3_story(story, five_w_one_h)

        issues: list[str] = []
        if has_unbalanced_quotes(story):
            issues.append("unbalanced quotes")
        if has_together_they_artifact(story):
            issues.append("'Together they' artifact")
        ungrounded = count_ungrounded_proper_nouns(story, allowed_names)
        if ungrounded > 3:
            issues.append(f"{ungrounded} ungrounded proper nouns")
        if not story_has_proper_ending(story):
            issues.append("no proper ending / cut-off")
        is_junk, junk_reason = detect_junk_density(story)
        if is_junk:
            issues.append(f"junk density: {junk_reason}")

        paragraphs = [p.strip() for p in story.split("\n\n") if p.strip()]
        word_count = len(story.split())
        if len(paragraphs) < 3 or word_count < 100:
            issues.append(f"too short ({len(paragraphs)} paragraphs, {word_count} words)")

        if len(issues) < best_issues:
            best_story = story
            best_issues = len(issues)
            best_story_parts = story_parts

        if not issues:
            logging.info("Layer 3 output passed all quality gates")
            break

        logging.warning("Layer 3 quality issues (attempt %d): %s", attempt + 1, "; ".join(issues))

    story = best_story
    if story is None or best_issues > 0:
        if story is None:
            logging.warning("Layer 3 produced no usable output — falling back to single-pass")
            return generate_single_pass(query, context, mode, use_thinking, tokenizer)

        paragraphs = [p.strip() for p in story.split("\n\n") if p.strip()]
        word_count = len(story.split())
        is_junk, _ = detect_junk_density(story)
        if len(paragraphs) < 3 or word_count < 100 or is_junk:
            logging.warning(
                "Layer 3 story unusable after retries (%d paragraphs, %d words, junk=%s) — "
                "falling back to single-pass",
                len(paragraphs), word_count, is_junk,
            )
            return generate_single_pass(query, context, mode, use_thinking, tokenizer)

        logging.warning("Layer 3 has %d remaining issues but output is usable — returning best attempt", best_issues)

    total_tokens = len(tokenizer.encode(story, add_special_tokens=False))
    logging.info(f"Three-layer generation complete ({len(best_story_parts)} sections, ~{total_tokens} tokens)")
    return story

def save_generated_story(content: str, filename: str = "Timestamp_generated_story.txt"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    output_folder = Path(BASE_PATH) / GENERATED_OUTPUT
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logging.info(f"Story saved to: {output_path}")
    return output_path
