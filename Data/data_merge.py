import yaml
import logging 
import json 
from pathlib import Path 
from tqdm import tqdm 
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent.parent
config_file = ROOT_DIR / "setup.yaml"
prompts_file = ROOT_DIR / "prompts.yaml"

with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

with open(prompts_file, "r", encoding="utf-8") as file:
    prompts = yaml.safe_load(file)

BASE_PATH = config.get("BASE_PATH")
Story_input = config.get("Story_input")
Metadata_input = config.get("Metadata_input")
Merge_Data_Output = config.get("Merged_data_output")
Gemini_api_key = config.get("Gemini_api_key")
Gemini_summary_model = config.get("Gemini_summary_model")
Chroma_json_template = config.get("Chroma_json_template")

SUMMARIZATION_PROMPTS = prompts.get("summarization", {})
prompt_template = SUMMARIZATION_PROMPTS.get("chunk_summary", "")
refine_prompt = SUMMARIZATION_PROMPTS.get("refine_summary", "")
batch_summary_prompt = SUMMARIZATION_PROMPTS.get("batch_summary", "")

BATCH_SIZE = 3

from Data.metadata import check_summary_empty_count, is_summary_empty

def Load_Story_Contents(BASE_PATH, Story_input):
    story_path = Path(BASE_PATH) / Story_input
    out = {}
    for f in story_path.iterdir():
        if f.is_file() and f.suffix.lower() == ".txt":
            with open(f, "r", encoding="utf-8") as file:
                out[f.stem] = file.read()
    return out

def Load_Metadata_Contents(BASE_PATH, Metadata_input):
    metadata_path = Path(BASE_PATH) / Metadata_input
    out = []
    for f in metadata_path.iterdir():
        if f.is_file() and f.suffix.lower() == ".json":
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
            data["_stem"] = f.stem
            out.append(data)
    return out

def Merge_Metadata_and_Story(BASE_PATH, Story_input, Metadata_input, Merge_Data_Output):
    story_contents = Load_Story_Contents(BASE_PATH, Story_input)
    metadata_contents = Load_Metadata_Contents(BASE_PATH, Metadata_input)

    if not metadata_contents or not story_contents:
        logging.error("No data found or data is not complete")
        return (False, 0)

    merged_count = 0
    for metadata in metadata_contents:
        stem = metadata.get("_stem", "")
        story_content = story_contents.get(stem, "")
        if not story_content:
            logging.warning(f"No story found for {stem}")
            continue
        metadata["documents"] = story_content
        merged_count += 1
    Save_Merged_Data(BASE_PATH, Merge_Data_Output, metadata_contents)
    return (True, merged_count)

def Save_Merged_Data(BASE_PATH, Merge_Data_Output, merged_data):
    output_folder = Path(BASE_PATH) / Merge_Data_Output
    output_folder.mkdir(parents=True, exist_ok=True)
    for data in merged_data:
        stem = data.get("_stem", "")
        output_file = output_folder / f"{stem}.json"
        out_data = {k: v for k, v in data.items() if k != "_stem"}
        with open(output_file, 'w', encoding="utf-8") as file:
            json.dump(out_data, file, indent=4, ensure_ascii=False)
    logging.info(f"Data saved successfully in {output_folder}")
    return True


def _id_sort_key(ids_val):
    """Extract numeric part from ids e.g. 'id_1' -> 1 for sorting."""
    if not ids_val or not isinstance(ids_val, str):
        return 0
    try:
        return int(ids_val.replace("id_", "").strip())
    except ValueError:
        return 0


def _get_merged_files_sorted_by_id(output_folder, batch_size=3):
    output_folder = Path(output_folder)
    if not output_folder.exists():
        return []
    files_with_data = []
    for f in output_folder.iterdir():
        if not f.is_file() or f.suffix.lower() != ".json":
            continue
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            continue
        documents = data.get("documents", "") or ""
        if not documents.strip():
            continue
        ids_val = data.get("ids", "") or data.get("ID", "")
        stem = f.stem
        files_with_data.append((f, data, stem, ids_val))
    files_with_data.sort(key=lambda x: _id_sort_key(x[3]))
    batches = []
    for i in range(0, len(files_with_data), batch_size):
        batch = [(p, d, s) for p, d, s, _ in files_with_data[i : i + batch_size]]
        batches.append(batch)
    return batches


def _parse_batch_summary_response(text):
    if not text or not isinstance(text, str):
        return []
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        out = json.loads(text)
        summaries = out.get("summaries") if isinstance(out, dict) else []
        return summaries if isinstance(summaries, list) else []
    except json.JSONDecodeError:
        return []


def create_summaries_for_batch(batch_items, llm, batch_prompt_template):
    if not batch_items or not batch_prompt_template.strip():
        return []
    parts = []
    for _path, data, stem in batch_items:
        documents = data.get("documents", "") or ""
        parts.append(f"[id={stem}]\n{documents}")
    batched_stories = "\n\n".join(parts)
    prompt = batch_prompt_template.format(batched_stories=batched_stories)
    out = llm.invoke(prompt)
    text = getattr(out, "content", str(out)) or ""
    summaries = _parse_batch_summary_response(text)
    result = []
    for s in summaries:
        if isinstance(s, dict):
            sid = s.get("id")
            summary = s.get("summary")
            if sid is not None and summary is not None:
                result.append((str(sid).strip(), (summary or "").strip()))
    return result


def create_summaries_for_merged_data(BASE_PATH, Merge_Data_Output, Gemini_summary_model, Gemini_api_key, chunk_prompt_template, refine_prompt_template):
    try:
        output_folder = Path(BASE_PATH) / Merge_Data_Output
        if not output_folder.exists():
            logging.error("Merged data folder not found")
            return (False, 0)

        if batch_summary_prompt and batch_summary_prompt.strip():
            batches = _get_merged_files_sorted_by_id(output_folder, batch_size=BATCH_SIZE)
            if not batches:
                logging.warning("No merged files with documents found for batch summarization")
                return (True, 0)
            llm = ChatGoogleGenerativeAI(
                model=Gemini_summary_model,
                temperature=0.2,
                api_key=Gemini_api_key,
            )
            summaries_count = 0
            stem_to_path_and_data = {}
            for batch in batches:
                for path, data, stem in batch:
                    stem_to_path_and_data[stem] = (path, data)
            for batch in tqdm(batches, desc="Creating summaries (batch)"):
                results = create_summaries_for_batch(batch, llm, batch_summary_prompt)
                for stem, summary_text in results:
                    if stem not in stem_to_path_and_data:
                        continue
                    path, data = stem_to_path_and_data[stem]
                    if "metadatas" in data and isinstance(data["metadatas"], dict):
                        data["metadatas"]["Summary"] = summary_text
                    else:
                        data["Summary"] = summary_text
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    summaries_count += 1
            logging.info(f"Summaries created successfully (batch). Summaries created for {summaries_count} items.")
            return (True, summaries_count)

        json_files = [
            f for f in output_folder.iterdir()
            if f.is_file() and f.suffix.lower() == ".json"
        ]
        summaries_count = 0

        for json_file in tqdm(json_files, desc="Creating summaries"):
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            documents = data.get("documents", "")
            if not documents or not documents.strip():
                continue

            llm = ChatGoogleGenerativeAI(
                model=Gemini_summary_model,
                temperature=0.2,
                api_key=Gemini_api_key,
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=300,
                separators=["\n\n", "\n", ".", " ", ""],
            )
            chunks = splitter.split_text(documents)

            chunk_prompt = PromptTemplate(
                input_variables=["story_part"],
                template=chunk_prompt_template,
            )
            chunk_chain = chunk_prompt | llm
            chunk_summaries = []
            for chunk in tqdm(chunks, desc="Summarizing chunks"):
                out = chunk_chain.invoke({"story_part": chunk})
                text = getattr(out, "content", str(out))
                chunk_summaries.append(text.strip() if text else "")

            combined = "\n".join(
                f"Part {i+1}: {s}" for i, s in enumerate(chunk_summaries)
            )

            refine_prompt_tpl = PromptTemplate(
                input_variables=["summary_text"],
                template=refine_prompt_template,
            )
            refine_chain = refine_prompt_tpl | llm
            final = refine_chain.invoke({"summary_text": combined})
            summary_text = getattr(final, "content", str(final))
            if "metadatas" in data and isinstance(data["metadatas"], dict):
                data["metadatas"]["Summary"] = summary_text.strip() if summary_text else ""
            else:
                data["Summary"] = summary_text.strip() if summary_text else ""
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            summaries_count += 1

        logging.info(f"Summaries created successfully. Summaries created for {summaries_count} items.")
        return (True, summaries_count)
    except Exception as e:
        logging.error(f"Error creating summaries: {e}")
        return (False, 0)

def count_fillable_empty_summaries(BASE_PATH, Merge_Data_Output):
    output_folder = Path(BASE_PATH) / Merge_Data_Output
    if not output_folder.exists():
        return 0
    count = 0
    for f in output_folder.iterdir():
        if not f.is_file() or f.suffix.lower() != ".json":
            continue
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
            if is_summary_empty(data) and (data.get("documents") or "").strip():
                count += 1
        except Exception:
            continue
    return count

def check_number_summaries(check_summary_empty_count, create_summaries_for_merged_data):
    total, empty_count, empty_list = check_summary_empty_count(BASE_PATH, Merge_Data_Output)
    fillable_count = count_fillable_empty_summaries(BASE_PATH, Merge_Data_Output)
    success, created_count = create_summaries_for_merged_data(BASE_PATH, Merge_Data_Output, Gemini_summary_model, Gemini_api_key, prompt_template, refine_prompt)
    if not success:
        logging.error("Failed to create summaries")
        return (False, 0)
    if fillable_count != created_count:
        logging.warning(f"Count mismatch: fillable (empty+has documents) was {fillable_count}, created was {created_count}")
        return (False, fillable_count, created_count)
    logging.info(f"Counts match: fillable={fillable_count}, created={created_count}.")
    return (True, fillable_count, created_count)
