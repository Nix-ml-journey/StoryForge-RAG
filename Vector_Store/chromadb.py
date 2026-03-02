import chromadb
import yaml
import logging
import json
from pathlib import Path 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent.parent
config_file = ROOT_DIR / "setup.yaml"

with open(config_file, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

BASE_PATH = config.get("BASE_PATH")
CHROMA_PATH = config.get("Chroma_path", "chroma_db")
COLLECTION_NAME = config.get("Chroma_collection_name", "English_Stories")

chroma_full_path = Path(BASE_PATH) / CHROMA_PATH
Client = chromadb.PersistentClient(path=str(chroma_full_path))
Collection = Client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

from Vector_Store.vector_store import model as embedding_model, create_embeddings

def load_data(Merge_Data):
    try:
        with open(Merge_Data, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except Exception as e:
        logging.error(f"Error loading data from {Merge_Data}: {e}")
        return None

def store_data(Merge_Data, Collection=Collection, Client=Client):
    try:
        path = Path(Merge_Data)
        if path.is_dir():
            ids, metadatas, documents = load_merged_dir_for_chroma(path)
            if not ids:
                logging.error("No merge JSONs found in directory.")
                return False
            Collection.add(ids=ids, metadatas=metadatas, documents=documents)
            logging.info(f"Stored {len(ids)} items from directory into Chroma.")
            return True
        data = load_data(Merge_Data)
        if data is None:
            logging.error("No data found")
            return False
        items = [data] if isinstance(data, dict) and (data.get("ids") is not None or data.get("ID") is not None) else (data if isinstance(data, list) else [])
        for item in items:
            meta = item.get("metadatas") or {}
            row_id = str(item.get("ids", item.get("ID", "")))
            doc_text = item.get("documents", item.get("Document", item.get("Content", "")))
            meta_row = {
                "Author": meta.get("Author", item.get("Author", "")),
                "Title": meta.get("Title", item.get("Title", "")),
                "Summary": meta.get("Summary", item.get("Summary", "")),
                "query_type": "content",
            }
            if "series" in item and isinstance(item["series"], dict):
                s = item["series"]
                meta_row["Series_name"] = s.get("Series_name", "")
                meta_row["Volume_name"] = s.get("Volume_name", "")
                meta_row["Volume_number"] = s.get("Volume_number", 0)
            Collection.upsert(
                ids=[row_id],
                metadatas=[meta_row],
                documents=[doc_text],
            )
        return True
    except Exception as e:
        logging.error(f"Error storing data to {Collection}: {e}")
        return False

def query_data(query: str, n_results: int = 5, query_type: str = "content", threshold: float = 0.5):
    try:
        results = Collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"query_type": query_type},
            include=["distances", "documents", "metadatas"],
        )
        return results
    except Exception as e:
        logging.error(f"Error querying data: {e}")
        return None

def delete_data(id:str):
    try:
        Collection.delete(ids=[id])
        return True
    except Exception as e:
        logging.error(f"Error deleting data: {e}")
        return False

def check_data(id: str):
    try:
        results = Collection.get(ids=[id])
        return results
    except Exception as e:
        logging.error(f"Error checking data: {e}")
        return None

def get_data(id: str):
    try:
        results = Collection.get(ids=[id])
        return results
    except Exception as e:
        logging.error(f"Error getting data: {e}")
        return None

def update_data(id: str, new_data: str):
    try:
        Collection.update(ids=[id], documents=[new_data])
        return True
    except Exception as e:
        logging.error(f"Error updating data: {e}")
        return False

def get_all_data():
    try:
        results = Collection.get(include=["documents", "metadatas"])
        return results
    except Exception as e:
        logging.error(f"Error getting all data: {e}")
        return None

def load_merged_dir_for_chroma(merge_output_dir):
    merge_path = Path(merge_output_dir)
    if not merge_path.is_dir():
        return [], [], []

    ids = []
    metadatas = []
    documents = []

    for file_path in sorted(merge_path.iterdir(), key=lambda p: (not p.suffix == ".json", p.stem)):
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    item = json.load(file)
                meta = item.get("metadatas") or {}
                row_id = str(item.get("ids", item.get("ID", file_path.stem)))
                ids.append(row_id)
                meta_row = {
                    "Author": meta.get("Author", item.get("Author", "")),
                    "Title": meta.get("Title", item.get("Title", "")),
                    "Summary": meta.get("Summary", item.get("Summary", "")),
                    "query_type": "content",
                }
                if "series" in item and isinstance(item["series"], dict):
                    s = item["series"]
                    meta_row["Series_name"] = s.get("Series_name", "")
                    meta_row["Volume_name"] = s.get("Volume_name", "")
                    meta_row["Volume_number"] = s.get("Volume_number", 0)
                metadatas.append(meta_row)
                doc_text = item.get("documents", item.get("Document", item.get("Content", "")))
                documents.append(doc_text)
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Skipping {file_path.name}: {e}")
                continue

    return ids, metadatas, documents

def ingest_into_chroma(merge_output_dir=None, collection=Collection):
    return ingest_with_sentence_transformer(merge_output_dir=merge_output_dir, collection=collection)

def ingest_with_sentence_transformer(merge_output_dir=None, collection=Collection):
    if merge_output_dir is None:
        merge_output_dir = Path(BASE_PATH) / config.get("Merged_data_output", "Data_Merged")
    
    ids, metadatas, documents = load_merged_dir_for_chroma(merge_output_dir)
    if not ids:
        logging.warning("No merge JSONs found to ingest.")
        return 0 

    data = [{"ID": i, "Content": d, "Author": m.get("Author",""), "Title": m.get("Title",""), "Summary": m.get("Summary","")}
            for i, d, m in zip(ids, documents, metadatas)]

    embedded = create_embeddings(embedding_model, data)
    if not embedded:
        return 0 

    collection.add(
        ids=ids,
        metadatas=metadatas,
        documents=documents,
        embeddings=[e["embedding_content"] for e in embedded], 
    )
    logging.info(f"Ingested {len(ids)} items with SentenceTransformer embeddings.")
    return len(ids)


