import yaml
import logging
import json
import torch
from pathlib import Path 
from tqdm import tqdm 
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

ROOT_DIR = Path(__file__).parent.parent
config_file = ROOT_DIR / "setup.yaml"

with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
    BASE_PATH = config.get("BASE_PATH")
    Merge_Data_Output = config.get("Merged_data_output")
    vector_store_model = config.get("Vector_store_model")

model = SentenceTransformer(vector_store_model, device=device)

def load_merged_data(BASE_PATH, Merge_Data_Output):
    try:
        folder_path = Path(BASE_PATH) / Merge_Data_Output
        
        if not folder_path.exists():
            logging.error(f"Folder not found: {folder_path}")
            return None
        
        if not folder_path.is_dir():
            logging.error(f"Path is not a directory: {folder_path}")
            return None
        
        all_data = []
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix == ".json":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "ID" in data:
                        all_data.append(data)
                    elif isinstance(data, list):
                        all_data.extend(data)
                except (json.JSONDecodeError, OSError) as e:
                    logging.warning(f"Skipping invalid file {file_path.name}: {e}")

        logging.info(f"Loaded {len(all_data)} items from {folder_path}")
        return all_data
    
    except Exception as e:
        logging.error(f"Error loading merged data: {e}")
        return None
    
def create_embeddings(model, data):
    try:
        if not data:
            raise ValueError("No data provided")

        embeddings = []
        for item in tqdm(data, desc="Creating embeddings"):
            summary_text = item.get("Summary") or ""
            content_text = item.get("Content") or ""
            embedding_summary = model.encode(summary_text).tolist()
            embedding_content = model.encode(content_text).tolist()
            embeddings.append({
                "ID": item.get("ID"),
                "Title": item.get("Title"),
                "Author": item.get("Author"),
                "Summary": summary_text,
                "Content": content_text,
                "embedding_summary": embedding_summary,
                "embedding_content": embedding_content,
            })

        logging.info(f"Created embeddings for {len(embeddings)} items")
        return embeddings
    
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        return None
    
def encode_query(model, query):
    return model.encode(query).tolist()