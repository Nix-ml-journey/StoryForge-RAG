import os
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

_model_instance: SentenceTransformer | None = None

def _load_embedding_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    offline = os.environ.get("OFFLINE_MODE", "").lower() in ("1", "true", "yes")
    try:
        _model_instance = SentenceTransformer(vector_store_model, device=device)
    except Exception as online_err:
        if offline:
            logging.warning(
                "Online load failed and OFFLINE_MODE is set: %s. "
                "Retrying with local_files_only=True.",
                online_err,
            )
        else:
            logging.warning(
                "Online load failed (DNS/network issue?): %s. "
                "Retrying with local_files_only=True. "
                "Set OFFLINE_MODE=true to skip online attempts.",
                online_err,
            )
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            _model_instance = SentenceTransformer(vector_store_model, device=device)
        except Exception as local_err:
            logging.error(
                "Cannot load embedding model '%s' even from cache. "
                "Ensure the model has been downloaded at least once while online. "
                "Error: %s",
                vector_store_model,
                local_err,
            )
            raise RuntimeError(
                f"Embedding model '{vector_store_model}' unavailable offline and cache is missing. "
                f"Run once while online to populate cache, or set the model path to a local directory."
            ) from local_err
        finally:
            if not offline:
                os.environ.pop("HF_HUB_OFFLINE", None)

    logging.info("Loaded embedding model: %s (device=%s)", vector_store_model, device)
    return _model_instance


class _LazyModel:
    """Proxy so that `from vector_store import model` works without immediate loading."""
    def __getattr__(self, name):
        return getattr(_load_embedding_model(), name)
    def __call__(self, *args, **kwargs):
        return _load_embedding_model()(*args, **kwargs)


model = _LazyModel()

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
    
def encode_query(model_or_lazy, query):
    m = model_or_lazy if isinstance(model_or_lazy, SentenceTransformer) else _load_embedding_model()
    return m.encode(query).tolist()