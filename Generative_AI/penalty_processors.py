import logging
from pathlib import Path

import torch
import yaml
from transformers import LogitsProcessor, LogitsProcessorList

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent.parent
config_file = ROOT_DIR / "setup.yaml"

with open(config_file, "r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f)

REPETITION_PENALTY = _config.get("Repetition_penalty", 1.3)
NO_REPEAT_NGRAM_SIZE = _config.get("No_repeat_ngram_size", 4)
FREQUENCY_PENALTY = _config.get("Frequency_penalty", 0.0)
PRESENCE_PENALTY = _config.get("Presence_penalty", 0.0)

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "to", "of", "in", "on", "at", "by", "for", "with", "from",
    "up", "out", "off", "over", "into", "onto", "upon",
    "and", "but", "or", "nor", "so", "yet", "if", "then", "than",
    "not", "no", "as", "about", "after", "before", "between", "through",
    "what", "who", "whom", "which", "where", "when", "how", "why",
    "all", "each", "every", "some", "any", "few", "more", "most",
    "just", "also", "very", "too", "quite", "still", "even",
}

_stopword_ids_cache: set = set()


def get_stopword_ids(tokenizer) -> set:
    global _stopword_ids_cache
    if _stopword_ids_cache:
        return _stopword_ids_cache
    ids = set()
    for word in STOPWORDS:
        for variant in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            ids.update(token_ids)
    _stopword_ids_cache = ids
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
