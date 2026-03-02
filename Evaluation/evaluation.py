import time
import yaml 
import logging 
import json 
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent.parent
config_file = ROOT_DIR / "setup.yaml"
prompts_file = ROOT_DIR / "prompts.yaml"

with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

with open(prompts_file, "r", encoding="utf-8") as file:
    prompts = yaml.safe_load(file)

BASE_PATH = config.get("BASE_PATH")
gemini_evaluation_model = config.get("Gemini_evaluation_model")
gemini_api_key = config.get("Gemini_api_key")
generated_story_output = config.get("Generated_story_output")
generated_summary_output = config.get("Generated_summary_output")
evaluated_stories_output = config.get("Evaluated_stories_output")

EVALUATION_PROMPTS = prompts.get("evaluation", {})
with_story_prompt = EVALUATION_PROMPTS.get("with_story", "")
with_summary_prompt = EVALUATION_PROMPTS.get("with_summary", "")

EVAL_RETRY_MAX_ATTEMPTS = 6
EVAL_RETRY_BASE_DELAY_SEC = 2
EVAL_RETRY_BACKOFF_FACTOR = 2

def _is_retryable_api_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "503" in msg or "unavailable" in msg or "high demand" in msg
        or "429" in msg or "rate limit" in msg or "resource exhausted" in msg
    )

def _invoke_with_retry(model, prompt: str):
    last_exc = None
    for attempt in range(EVAL_RETRY_MAX_ATTEMPTS):
        try:
            return model.invoke(prompt)
        except Exception as e:
            last_exc = e
            if not _is_retryable_api_error(e) or attempt == EVAL_RETRY_MAX_ATTEMPTS - 1:
                raise
            delay = EVAL_RETRY_BASE_DELAY_SEC * (EVAL_RETRY_BACKOFF_FACTOR ** attempt)
            logging.warning(
                "Evaluation API transient error (attempt %s/%s), retrying in %.1fs: %s",
                attempt + 1, EVAL_RETRY_MAX_ATTEMPTS, delay, e,
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc

def evaluate_model(temperature: float = 0.3):
    if not gemini_api_key:
        raise ValueError("Gemini API key is not set in the configuration")
    if not gemini_evaluation_model:
        raise ValueError("Gemini evaluation model is not set in the configuration")
    llm = ChatGoogleGenerativeAI(
        model = gemini_evaluation_model,
        temperature = temperature,
        api_key = gemini_api_key,
    )
    logging.info(f"Initialized StoryEvaluator with model: {gemini_evaluation_model}")
    return llm

def search_story_date(selected_date: str) -> List[Path]:
    try:
        output_folder = Path(BASE_PATH) / generated_story_output
        if not output_folder.exists():
            logging.error(f"Generated story output folder not found: {output_folder}")
            return []

        stories_generated_date = []
        for file_path in output_folder.iterdir():
            if file_path.is_file() and file_path.suffix == ".txt":
                file_name = file_path.stem
                file_date = file_name.split("_")[0]
                if file_date == selected_date:
                    stories_generated_date.append(file_path)
        return stories_generated_date
    except Exception as e:
        logging.error(f"Error searching stories for date {selected_date}: {e}")
        return []

def search_summary_date(selected_date: str) -> List[Path]:
    try:
        output_folder = Path(BASE_PATH) / generated_summary_output
        if not output_folder.exists():
            logging.error(f"Generated summary output folder not found: {output_folder}")
            return []

        summaries_generated_date = []
        for file_path in output_folder.iterdir():
            if file_path.is_file() and file_path.suffix == ".txt":
                file_name = file_path.stem
                file_date = file_name.split("_")[0]
                if file_date == selected_date:
                    summaries_generated_date.append(file_path)
        return summaries_generated_date
    except Exception as e:
        logging.error(f"Error searching summaries for date {selected_date}: {e}")
        return []

def find_story_path_for_summary(summary_path: Path) -> Optional[Path]:
    try:
        story_folder = Path(BASE_PATH) / generated_story_output
        if not story_folder.exists():
            return None
        stem = summary_path.stem
        story_stem = stem.replace("_summary", "_generated_story").replace("summary", "generated_story")
        candidate = story_folder / f"{story_stem}.txt"
        if candidate.exists():
            return candidate
        date_part = stem.split("_")[0] if "_" in stem else stem
        for p in story_folder.iterdir():
            if p.is_file() and p.suffix == ".txt" and p.stem.startswith(date_part):
                return p 
    except Exception as e:
        logging.error(f"Error finding story path for summary {summary_path}: {e}")
        return None

def _parse_json_response(response) -> dict:
    raw = response.content if hasattr(response, 'content') else response
    if isinstance(raw, list):
        parts = []
        for p in raw:
            if isinstance(p, dict):
                parts.append(p.get("text", ""))
            else:
                parts.append(str(p) if p is not None else "")
        text = "".join(parts)
    else:
        text = str(raw) if raw is not None else ""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"Evaluation response was not valid JSON: {e}")
        logging.error(f"Raw response (first 500 chars): {text[:500]}")
        return {}

def evaluate_generated_story(model, file_path) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            story = file.read()
        prompt = with_story_prompt.format(story=story)
        response = _invoke_with_retry(model, prompt)
        evaluation_data = _parse_json_response(response)
        return evaluation_data
    except Exception as e:
        logging.error(f"Error evaluating generated story: {e}")
        return {}

def evaluate_generated_summary(model, summary_path, story_path=None) -> Dict[str, Any]:
    try:
        with open(summary_path, "r", encoding="utf-8") as file:
            summary = file.read()
        story_text = "Not provided."
        if story_path and Path(story_path).exists():
            with open(story_path, "r", encoding="utf-8") as f:
                story_text = f.read()
        prompt = with_summary_prompt.format(summary=summary, story=story_text)
        response = _invoke_with_retry(model, prompt)
        evaluation_data = _parse_json_response(response)
        return evaluation_data
    except Exception as e:
        logging.error(f"Error evaluating generated summary: {e}")
        return {}

def save_evaluation_results(evaluation_data: Dict[str, Any], result_type: str, source_stem: str = None):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = source_stem if source_stem else datetime.now().strftime("%H%M%S")
        filename = f"{timestamp}_{result_type}_{unique}_evaluation_results.json"
        output_folder = Path(BASE_PATH) / evaluated_stories_output
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / filename
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(evaluation_data, file, indent=4)
        logging.info(f"Evaluation results saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving evaluation results: {e}")
        return {}

def orchestrator_evaluation(selected_date: str):
    model = evaluate_model()

    story_files_path = search_story_date(selected_date)
    summary_files_path = search_summary_date(selected_date)

    if not story_files_path and not summary_files_path:
        logging.warning(f"No files found for date {selected_date}")
        return
    
    for story_file_path in story_files_path:
        evaluation_data = evaluate_generated_story(model, story_file_path)
        if evaluation_data:
            save_evaluation_results(evaluation_data, result_type="story", source_stem=story_file_path.stem) 

    for summary_file_path in summary_files_path:
        story_path = find_story_path_for_summary(summary_file_path)
        evaluation_data = evaluate_generated_summary(model, summary_file_path, story_path=story_path)
        if evaluation_data:
            save_evaluation_results(evaluation_data, result_type="summary", source_stem=summary_file_path.stem)
