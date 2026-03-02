import datetime
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import re
import chromadb
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from Vector_Store.vector_store import encode_query, model as embedding_model

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

GENERATION_FAST_TEMPERATURE = config.get("Generation_fast_temperature", 0.5)
GENERATION_FAST_TOP_P = config.get("Generation_fast_top_p", 0.9)
GENERATION_THINKING_TEMPERATURE = config.get("Generation_thinking_temperature", 0.7)
GENERATION_THINKING_TOP_P = config.get("Generation_thinking_top_p", 0.9)

GENERATION_FAST_MAX_TOKENS = config.get("Generation_fast_max_tokens", 768)
GENERATION_THINKING_MAX_TOKENS = config.get("Generation_thinking_max_tokens", 1536)

TWO_PASS_GENERATION = config.get("Two_pass_generation", False)
OUTLINE_MAX_TOKENS = config.get("Outline_max_tokens", 300)
GENERATION_PASS2_FAST_MAX_TOKENS = config.get("Generation_pass2_fast_max_tokens", 2048)
GENERATION_PASS2_THINKING_MAX_TOKENS = config.get("Generation_pass2_thinking_max_tokens", 3072)

class Gen_mode(str, Enum):
    FAST = GENERATION_MODE_FAST
    THINKING = GENERATION_MODE_THINKING

def _mode_to_str(mode: Union["Gen_mode", str, None]) -> str:
    if mode is None:
        return "fast"
    if isinstance(mode, Gen_mode):
        return (mode.value or "fast").lower()
    return (mode or "fast").lower()

GEN_PROMPTS = prompts.get("generation", {})
SUMMARY_SYSTEM_PROMPT = GEN_PROMPTS.get("summary_system", "")
SUMMARY_USER_PROMPT = GEN_PROMPTS.get("summary_user", "")
FULL_STORY_SYSTEM_PROMPT = GEN_PROMPTS.get("full_story_system", "")
FULL_STORY_USER_PROMPT = GEN_PROMPTS.get("full_story_user", "")
OUTLINE_SYSTEM_PROMPT = GEN_PROMPTS.get("outline_system", "")
OUTLINE_USER_PROMPT = GEN_PROMPTS.get("outline_user", "")
STORY_FROM_OUTLINE_SYSTEM_PROMPT = GEN_PROMPTS.get("story_from_outline_system", "")
STORY_FROM_OUTLINE_USER_PROMPT = GEN_PROMPTS.get("story_from_outline_user", "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
logging.info(
    f"Generation config: model={GENERATIVE_MODEL}, max_prompt_tokens={MODEL_MAX_PROMPT_TOKENS}, "
    f"n_results={STORY_GENERATION_N_RESULTS}, two_pass={TWO_PASS_GENERATION}, "
    f"fast_tokens={GENERATION_PASS2_FAST_MAX_TOKENS if TWO_PASS_GENERATION else GENERATION_FAST_MAX_TOKENS}, "
    f"thinking_tokens={GENERATION_PASS2_THINKING_MAX_TOKENS if TWO_PASS_GENERATION else GENERATION_THINKING_MAX_TOKENS}"
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

def get_generation_mode(mode: Union[Gen_mode, str, None] = Gen_mode.FAST):
    mode_str = _mode_to_str(mode)
    if mode_str == "thinking":
        return GENERATION_MODE_THINKING
    return GENERATION_MODE_FAST

def query_summary_searching(query_text: str, n_results: int = 5, query_type: str = "summary"):

    _, collection = get_chroma_client()
    query_embedding = encode_query(embedding_model, query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"query_type": query_type},
        include=["distances", "documents", "metadatas", "embeddings"],
    )

    logging.info(f"Found {len(results['documents'][0])} results for query: {query_text}")
    return results 

def query_content_searching(query_text: str, n_results: int = 5, query_type: str = "content"):
    _, collection = get_chroma_client()
    query_embedding = encode_query(embedding_model, query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"query_type": query_type},
        include=["distances", "documents", "metadatas", "embeddings"],
    )

    logging.info(f"Found {len(results['documents'][0])} results for query: {query_text}")
    return results 

def build_context(query_results: dict, query_type: str = "summary"):
    context_builded = []

    for i, (doc, metadata) in enumerate(zip(
        query_results["documents"][0],
        query_results["metadatas"][0],
    )):
        title = metadata.get('Title', 'Unknown')
        author = metadata.get('Author', 'Unknown')
        summary = metadata.get('Summary', 'Unknown')

        if query_type == "summary":
            context_builded.append(f"[{title} by {author}]\n{summary}")
        else:
            context_builded.append(f"[{title} by {author}]\n{doc}")

    return "\n\n".join(context_builded)

def get_generation_params(mode: Union[Gen_mode, str, None] = Gen_mode.FAST):
    """Return max_new_tokens, temperature, top_p for the given mode."""
    mode_str = _mode_to_str(mode)
    if mode_str == "thinking":
        return {
            "max_new_tokens": GENERATION_THINKING_MAX_TOKENS,
            "temperature": GENERATION_THINKING_TEMPERATURE,
            "top_p": GENERATION_THINKING_TOP_P,
        }
    return {
        "max_new_tokens": GENERATION_FAST_MAX_TOKENS,
        "temperature": GENERATION_FAST_TEMPERATURE,
        "top_p": GENERATION_FAST_TOP_P,
    }

def truncate_context(context: str, system_prompt: str, query: str, tokenizer, enable_thinking: bool = False, user_prompt_template: str = "") -> str:
    template = user_prompt_template or FULL_STORY_USER_PROMPT
    placeholder_user = template.format(context="", query=query, outline="")
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

def get_pass2_generation_params(mode: Union[Gen_mode, str, None] = Gen_mode.FAST):
    mode_str = _mode_to_str(mode)
    if mode_str == "thinking":
        return {
            "max_new_tokens": GENERATION_PASS2_THINKING_MAX_TOKENS,
            "temperature": GENERATION_THINKING_TEMPERATURE,
            "top_p": GENERATION_THINKING_TOP_P,
        }
    return {
        "max_new_tokens": GENERATION_PASS2_FAST_MAX_TOKENS,
        "temperature": GENERATION_FAST_TEMPERATURE,
        "top_p": GENERATION_FAST_TOP_P,
    }

def generate_full_story(query: str, n_results: Optional[int] = None, mode: Union[Gen_mode, str, None] = Gen_mode.FAST):
    if n_results is None:
        n_results = STORY_GENERATION_N_RESULTS
    results = query_content_searching(query, n_results=n_results, query_type="content")
    context = build_context(results, query_type="content")

    use_thinking = _mode_to_str(mode) == "thinking"
    _, tokenizer = get_generative_model(GENERATIVE_MODEL)

    if TWO_PASS_GENERATION:
        return generate_full_story_two_pass(query, context, mode, use_thinking, tokenizer)
    return generate_full_story_single_pass(query, context, mode, use_thinking, tokenizer)

def generate_full_story_single_pass(query: str, context: str, mode, use_thinking: bool, tokenizer):
    system_prompt = FULL_STORY_SYSTEM_PROMPT
    gen_params = get_generation_params(mode)
    context = truncate_context(context, system_prompt, query, tokenizer, enable_thinking=use_thinking)
    user_prompt = FULL_STORY_USER_PROMPT.format(context=context, query=query)

    return generate_response(
        system_prompt,
        user_prompt,
        max_tokens=gen_params["max_new_tokens"],
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        enable_thinking=use_thinking,
    )


def generate_full_story_two_pass(query: str, context: str, mode, use_thinking: bool, tokenizer):

    logging.info("Two-pass generation: starting Pass 1 (outline)")
    outline_system = OUTLINE_SYSTEM_PROMPT
    context = truncate_context(context, outline_system, query, tokenizer, enable_thinking=use_thinking, user_prompt_template=OUTLINE_USER_PROMPT)
    outline_user = OUTLINE_USER_PROMPT.format(context=context, query=query)

    outline = generate_response(
        outline_system,
        outline_user,
        max_tokens=OUTLINE_MAX_TOKENS,
        temperature=GENERATION_FAST_TEMPERATURE,
        top_p=GENERATION_FAST_TOP_P,
        enable_thinking=use_thinking,
    )

    if not outline or not outline.strip():
        logging.warning("Two-pass: Pass 1 produced empty outline, falling back to single-pass")
        return generate_full_story_single_pass(query, context, mode, use_thinking, tokenizer)

    logging.info(f"Two-pass: outline generated ({len(tokenizer.encode(outline, add_special_tokens=False))} tokens)")

    logging.info("Two-pass generation: starting Pass 2 (full story from outline)")
    pass2_params = get_pass2_generation_params(mode)
    story_system = STORY_FROM_OUTLINE_SYSTEM_PROMPT
    story_user = STORY_FROM_OUTLINE_USER_PROMPT.format(outline=outline, query=query)

    story = generate_response(
        story_system,
        story_user,
        max_tokens=pass2_params["max_new_tokens"],
        temperature=pass2_params["temperature"],
        top_p=pass2_params["top_p"],
        enable_thinking=use_thinking,
    )

    logging.info("Two-pass generation: complete")
    return story

def generate_response(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 3200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = False,
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

    logging.info(f"Generating response (prompt_tokens={model_inputs.input_ids.shape[1]}, max_new_tokens={max_tokens}, temperature={temperature})")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
    )

    output_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = clean_generated_output(response)

    logging.info("Generation complete")
    return response


def clean_generated_output(text: str) -> str:
    if '\n---\n' in text:
        text = text.split('\n---\n')[0]
    text = re.sub(r'\n\[.{1,80} by .{1,80}\]\s*$', '', text)

    cjk = re.search(r'[\u4e00-\u9fff\u3000-\u303f]', text)
    if cjk:
        text = text[:cjk.start()]

    blob = re.search(r'\S{40,}', text)
    if blob:
        text = text[:blob.start()]

    # Paragraph-level degeneration detection.
    for para in text.split('\n\n'):
        words = re.split(r'\s+', para.strip())
        word_count = len(words)
        if word_count < 20:
            continue
        sentence_enders = len(re.findall(r'[.!?]', para))
        if word_count > 40 and sentence_enders == 0:
            cut_pos = text.find(para)
            if cut_pos > 0:
                text = text[:cut_pos]
            break
        if word_count > 60 and sentence_enders < word_count // 40:
            cut_pos = text.find(para)
            if cut_pos > 0:
                text = text[:cut_pos]
            break
        avg_words_per_sentence = word_count / max(sentence_enders, 1)
        if word_count > 30 and avg_words_per_sentence > 45:
            cut_pos = text.find(para)
            if cut_pos > 0:
                text = text[:cut_pos]
            break

    # Trim to last complete sentence
    last_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'),
                   text.rfind('."'), text.rfind('!"'), text.rfind('?"'))
    if last_end > len(text) // 3:
        text = text[:last_end + 1]

    return text.strip()

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

