import yaml
import logging 
import json 
import sys
import re
from pathlib import Path 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config_dir = Path(__file__).parent
config_file = Path(__file__).parent.parent / "setup.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)
    story_input = config["Story_input"]
    metadata_input = config["Metadata_input"]
    database_schema = config["database_schema"]

BASE_PATH = config.get("BASE_PATH")

def load_story(BASE_PATH, story_input):
    story_path = Path(BASE_PATH) / story_input
    all_file_count = len([f for f in story_path.iterdir() if f.is_file() and f.suffix.lower() == ".txt"])
    return all_file_count

def load_metadata(BASE_PATH, metadata_input):
    metadata_path = Path(BASE_PATH) / metadata_input
    all_data = []
    for file_path in metadata_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_data.append(data)
    return all_data

def get_story_txt_files(BASE_PATH, story_input):
    story_path = Path(BASE_PATH) / story_input
    if not story_path.exists():
        return []
    return [f for f in story_path.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]

def create_json_metadata_from_stories(BASE_PATH, story_input, metadata_input, database_schema):
    story_path = Path(BASE_PATH) / story_input
    metadata_path = Path(BASE_PATH) / metadata_input
    metadata_path.mkdir(parents=True, exist_ok=True)
    count = 0
    for txt_file in story_path.iterdir():
        if not txt_file.is_file() or txt_file.suffix.lower() != ".txt":
            continue
        base_name = txt_file.stem 
        json_path = metadata_path / f"{base_name}.json"

        metadata = {
            "ids": f"id_{base_name}",
            "metadatas": {
                "Author": "",
                "Title": "",
                "Summary": ""
            },
            "chapters": {
                "Chapter_name": "",
                "Chapter_number": 0,
                "Is_series": True
            },
            "series": {
                "Series_name": "",
                "Volume_name": "",
                "Volume_number": 0
            },
            "documents": ""
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        logging.info(f"Created {json_path.name} from {txt_file.name}")
        count += 1
    return count

def is_summary_empty(data):
    try:
        summary = (data.get("metadatas") or {}).get("Summary")
        return summary is None or (isinstance(summary, str) and summary.strip() == "")
    except Exception:
        return True

def check_summary_empty_count(BASE_PATH, metadata_input):
    metadata_path = Path(BASE_PATH) / metadata_input
    if not metadata_path.exists():
        return 0,0,[]

    total_count = 0 
    empty_files = []
    for file_path in metadata_path.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".json":
            continue
        total_count += 1
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if is_summary_empty(data):
                empty_files.append(file_path.name)
        except Exception as e:
            empty_files.append(f"{file_path.name} (read error: {e})")

    return total_count, len(empty_files), empty_files

def get_user_input():
    user_input = input("Enter the filename to update: ")
    return user_input

def retrieve_and_update_metadata(user_input):
    file_path = Path(BASE_PATH) / metadata_input / user_input
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]
    with open(file_path, "r") as file:
        data = json.load(file)
    data[user_input] = input(f"Enter the value for {user_input}: ")
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    return True

def save_updated_metadata(file_path, updated_metadata):
    with open(file_path, "w") as file:
        json.dump(updated_metadata, file, indent=4)
    logging.info(f"Updated metadata saved to: {file_path}")
    return True

if __name__ == "__main__":
    create_json_metadata_from_stories(BASE_PATH, story_input, metadata_input, database_schema)

    total, empty_count, empty_list = check_summary_empty_count(BASE_PATH, metadata_input)
    print(f"Total metadata files: {total}")
    print(f"Files with empty Summary: {empty_count}")
    if empty_list:
        print("Empty summary in:", empty_list)

    input("Do manual insert for the metadata (fill the JSON files), then press Enter to run check again...")

    total2, empty_count2, empty_list2 = check_summary_empty_count(BASE_PATH, metadata_input)
    print(f"After manual insert — Total: {total2}, Still empty Summary: {empty_count2}")
    if empty_list2:
        print("Still empty summary in:", empty_list2)

